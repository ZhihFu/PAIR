
<p align="center">
  <h1 align="center">PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval</h1>
  <p align="center">
    <a href="https://zhihfu.github.io/"><strong>Zhiheng Fu</strong></a>
    &nbsp;&nbsp;
    <a href="https://lee-zixu.github.io/"><strong>Zixu Li</strong></a>
    &nbsp;&nbsp;
    <a href="https://zivchen-ty.github.io/"><strong>Zhiwei Chen</strong></a>
    &nbsp;&nbsp;
    <strong>Chunxiao Wang</strong>
    &nbsp;&nbsp;
    <a href="https://xuemengsong.github.io/"><strong>Xuemeng Song</strong></a>
    &nbsp;&nbsp;
    <strong>Yupeng Hu</strong>
    &nbsp;&nbsp;
    <a href="https://liqiangnie.github.io/"><strong>Liqiang Nie</strong></a>
  </p>
  <br>
  <p align="center">
    <a href="https://ieeexplore.ieee.org/document/10888153"><img alt='Paper' src="https://img.shields.io/badge/Paper-IEEE-green.svg?style=flat-square"></a>
    <a href="https://zhihfu.github.io/PAIR.github.io/"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>
  </p>
  <br>
</p>

This is an open-source implementation of the paper "PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval" (**PAIR**).


### Installation
1. Clone the repository

```sh
git clone https://github.com/ZhihFu/PAIR
```

2. Running Environment

```sh
Platform: NVIDIA Tesla T4 16G
Python  3.8.10
Pytorch  2.0.0
```


### Data Preparation

#### CIRR

Download the CIRR dataset following the instructions in the [**official repository**](https://github.com/Cuberick-Orion/CIRR).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRR
│   ├── train
|   |   ├── [0 | 1 | 2 | ...]
|   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

│   ├── dev
|   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

│   ├── cirr
|   |   ├── captions
|   |   |   ├── cap.rc2.[train | val | test1].json
|   |   ├── image_splits
|   |   |   ├── split.rc2.[train | val | test1].json
```

#### FashionIQ

Download the FashionIQ dataset following the instructions in
the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── FashionIQ
│   ├── captions
|   |   ├── cap.dress.[train | val | test].json
|   |   ├── cap.toptee.[train | val | test].json
|   |   ├── cap.shirt.[train | val | test].json

│   ├── image_splits
|   |   ├── split.dress.[train | val | test].json
|   |   ├── split.toptee.[train | val | test].json
|   |   ├── split.shirt.[train | val | test].json

│   ├── dress
|   |   ├── [B000ALGQSY.jpg | B000AY2892.jpg | B000AYI3L4.jpg |...]

│   ├── shirt
|   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]

│   ├── toptee
|   |   ├── [B0000DZQD6.jpg | B000A33FTU.jpg | B000AS2OVA.jpg | ...]
```

#### Shoes

Download the Shoes dataset following the instructions in
the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-retrieval/tree/master/dataset).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── Shoes
│   ├── captions_shoes.json
│   ├── eval_im_names.txt
│   ├── relative_captions_shoes.json
│   ├── train_im_names.txt
│   ├── [womens_athletic_shoes | womens_boots | ...]
|   |   ├── [0 | 1]
|   |   ├── [img_womens_athletic_shoes_375.jpg | descr_womens_athletic_shoes_734.txt | ...]

```

#### Train

Train PAIR on CIRR, FashionIQ, Shoes.

```sh
python3 train.py 
--model_dir ... 
--dataset {cirr, fashioniq, shoes}
--cirr_path ""
--fashioniq_path ""
--shoes_path ""
```

```
--dataset <str>                 Dataset to use, options: ['cirr', 'fashioniq', 'shoes']
--cirr_path <str>               Path to the CIRR dataset root folder
--fashioniq_path <str>          Path to the FashionIQ dataset root folder
--shoes_path <str>              Path to the Shoes dataset root folder
--model_dir <str>               Path to save checkpoints and logs
```


</details>

### Test Phase

To generate the predictions file for uploading on the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) using the our model, please execute the following command:

```sh
python src/cirr_test_submission.py model_path
```

```
model_path <str> : Path of the PAIR checkpoint on CIRR, e.g. "checkpoints/PAIR_CIRR.pt"
```


</details>



### Acknowledgement
This codebase is heavily inspired by and built upon [CLIP4cir](https://github.com/ABaldrati/CLIP4Cir).


