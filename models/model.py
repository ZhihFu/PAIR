import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.autograd import Variable
from transformers import CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor, AutoTokenizer, CLIPTextModelWithProjection
import open_clip
import os
from typing import Callable, Optional, Sequence, Tuple
from einops import rearrange
import MHTransformer
import MHTransformer_rev

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens
        
        
class Backbone(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.5, local_token_num=8):
        super().__init__()
        clip_path = './'
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin'))
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(768,512)
        self.text_fc = nn.Linear(512,512)
        self.transformer = MHTransformer.Transformer(dim_self=hidden_dim, num_heads=8, dim_ref=hidden_dim,num_layers=1)
        self.transformer_a = MHTransformer.Transformer(dim_self=hidden_dim, num_heads=8, dim_ref=hidden_dim,num_layers=1)
        self.transformer_b = MHTransformer_rev.Transformer(dim_self=hidden_dim, num_heads=4, dim_ref=hidden_dim,num_layers=2)



        self.mlp = nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Softmax(dim=-1)
                        )
        self.mlp_img = nn.Sequential(
                        nn.LayerNorm(49),
                        nn.Linear(49, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, local_token_num),
                        nn.Softmax(dim=-1)
                        )
        
        self.mlp_text = nn.Sequential(
                        nn.LayerNorm(77),
                        nn.Linear(77, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, local_token_num),
                        nn.Softmax(dim=-1)
                        )

        self.local_token_num = local_token_num

        

    def visual_out(self, x):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.clip.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)

        x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x)
        pooled, tokens = self.clip.visual._global_pool(x)
        # print(tokens.shape)

        pooled = pooled @ self.clip.visual.proj

        
        return pooled, x
    
    
    def text_out(self, text):
        cast_dtype = self.clip.transformer.get_cast_dtype()

        x = self.clip.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        pooled, tokens = text_global_pool(x, text, self.clip.text_pool_type)
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                pooled = self.clip.text_projection(x)
            else:
                pooled = pooled @ self.clip.text_projection

        return pooled, x

    def extract_img_fea(self, x):
        img_global_fea, img_local_fea = self.visual_out(x)
        img_global_fea = img_global_fea.unsqueeze(1)
        img_local_fea = self.fc(img_local_fea.float())
        img_local_fea = img_local_fea[:, 1:, :]
        img_local_fea = self.mlp_img(img_local_fea.transpose(-2,-1)).transpose(-2,-1)

        Is = self.transformer(torch.cat([img_global_fea, img_local_fea, img_global_fea], dim=1))[:,:self.local_token_num+1, :]

        img_select_tokens = torch.cat([img_global_fea, Is], dim=1)

        return img_select_tokens
    
    def extract_img_fea_patch_selection(self, img_x, txt):
        bsz = img_x.shape[0]
        img_global_fea, img_local_fea = self.visual_out(img_x)
        img_global_fea = img_global_fea.unsqueeze(1)
        img_local_fea = self.fc(img_local_fea.float())
        img_local_fea = img_local_fea[:, 1:, :]
        
        txt_token = self.tokenizer(txt).cuda()
        text_global_fea, text_local_fea = self.text_out(txt_token)
        text_global_fea = text_global_fea.unsqueeze(1)
        text_local_fea = self.text_fc(text_local_fea.float()) 
        
        img_local_fea = self.mlp_img(img_local_fea.transpose(-2,-1)).transpose(-2,-1)
        text_local_fea = self.mlp_text(text_local_fea.transpose(-2,-1)).transpose(-2,-1)

        global_tran = self.transformer(self.mlp(img_global_fea * text_global_fea))
        
        Is = self.transformer_a(torch.cat([img_global_fea, img_local_fea, global_tran], dim=1) )
        Ts = self.transformer_a(torch.cat([text_global_fea, text_local_fea, global_tran], dim=1))
        Ix = self.transformer_b(torch.cat([img_global_fea, img_local_fea, global_tran], dim=1))
        Tx = self.transformer_b(torch.cat([text_global_fea, text_local_fea, global_tran], dim=1))
        
        global_is = F.normalize(Is[:, 0, :], dim=-1).unsqueeze(1)
        local_is = F.normalize(Is[:, 1:, :], dim=-1)
        global_tx = F.normalize(Tx[:, 0, :], dim=-1).unsqueeze(1)
        local_tx = F.normalize(Tx[:, 1:, :], dim=-1)

        img_select_tokens_con = torch.cat([img_global_fea, global_is, local_is], dim=1)
        cap_select_tokens_con = torch.cat([text_global_fea, global_tx, local_tx], dim=1)
        
        ref = torch.cat([img_global_fea, img_local_fea], dim=1)
        mod = torch.cat([text_global_fea, text_local_fea], dim=1)
        
        
        return img_select_tokens_con, cap_select_tokens_con, Is, Ts, Ix, Tx, ref, mod



class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
                        nn.LayerNorm(in_channels * 2),
                        nn.Linear(in_channels * 2, in_channels),
                        nn.GELU(),
                        nn.Linear(in_channels, out_channels)
                        )

    def forward(self, x, text_embed):
        text_embed_ = torch.cat([x, text_embed], dim=-1)
        batch = x.shape[0]
        chanel = x.shape[1] * 2
        gamma = self.MLP(text_embed_)
        x = gamma * x + (1-gamma) * text_embed
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class PAIR(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.5, local_token_num=8, t=10):
        super().__init__()
        self.backbone = Backbone(hidden_dim, dropout, local_token_num)
        self.loss_T = nn.Parameter(torch.tensor([10.]))
        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(local_token_num+2)]))
        self.local_weight_dis = nn.Parameter(torch.tensor([1.0 for _ in range(local_token_num+1)]))
        self.t = t
        
        self.affine = FeatureWiseAffine(hidden_dim, hidden_dim, use_affine_level=True)


    def target_fea(self, tag):
        tag_token = self.backbone.extract_img_fea(tag)
        return tag_token#, ref_mask
    
    def compose_feature(self, ref, mod):
        ref_token, mod_token, Is, Ts, Ix, Tx, ref, mod = self.backbone.extract_img_fea_patch_selection(ref, mod)
        fuse_local = self.affine(ref_token, mod_token)

        return fuse_local, Is, Ts, Ix, Tx, ref, mod

    def extract_retrieval_compose(self, ref, mod):

        fuse_local,_,_,_,_,_,_ = self.compose_feature(ref, mod)
        
        fuse_local = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)

        return fuse_local

    def extract_retrieval_target(self, tag):
        tag_local = self.target_fea(tag)
        tag_local = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
        return tag_local

    def compute_loss(self, ref, mod, tag):

        fuse_local, Is, Ts, Ix, Tx, ref_, mod_ = self.compose_feature(ref, mod)
        
        tag_local = self.target_fea(tag)
        loss = {}
        
        Is = F.normalize(torch.mean(Is, dim=1), p=2, dim=-1)
        Ix = F.normalize(torch.mean(Ix, dim=1), p=2, dim=-1)
        Ts = F.normalize(torch.mean(Ts, dim=1), p=2, dim=-1)
        Tx = F.normalize(torch.mean(Tx, dim=1), p=2, dim=-1)
        ref = F.normalize(torch.mean(ref_, dim=1), p=2, dim=-1)
        mod = F.normalize(torch.mean(mod_, dim=1), p=2, dim=-1)
        ref_feature = (F.normalize(ref_, p=2, dim=-1) * self.local_weight_dis.unsqueeze(0).unsqueeze(-1)).flatten(1)
        mod_feature = (F.normalize(mod_, p=2, dim=-1) * self.local_weight_dis.unsqueeze(0).unsqueeze(-1)).flatten(1)
        retrieval_query = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)
        retrieval_target = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)

        tag_feature = (F.normalize(tag_local, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)

        loss['stu_rank'] = self.info_nce(retrieval_query, retrieval_target)
        loss['kl'] = self.kl_div(retrieval_query, retrieval_target, tag_feature, tag_feature, self.t)
        loss['disen'] = self.disentangle(Is, ref, Ix, ref, ref_feature, ref_feature, self.t) + self.disentangle(Ts, mod, Tx, mod, mod_feature, mod_feature, self.t)
        
        return loss


    def info_nce(self, query, target):
        x = torch.mm(query, target.T)
        labels = torch.arange(query.shape[0]).long().cuda()
        return F.cross_entropy(x * self.loss_T, labels)

    def disentangle(self, x1, y1, x2, y2, x3, y3, t):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)
        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t
        x3_y3 = torch.mm(x3, y3.T) / t
        d1 = x1_y1 * x2_y2
        log_soft_x1 = F.log_softmax(d1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x3_y3), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
        return kl
    
    def kl_div(self, x1, y1, x2, y2, t):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)

        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t

        log_soft_x1 = F.log_softmax(x1_y1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2_y2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')

        return kl
    


