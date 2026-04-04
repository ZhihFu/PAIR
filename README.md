<a id="top"></a>
<div align="center">
  <img src="./assets/logo/pair-logo.png" width="300"> 
  <h1>PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval</h1>

  <div>
  <a target="_blank" href="https://zhihfu.github.io/">Zhiheng&#160;Fu</a><sup>1</sup>,
  <a target="_blank" href="https://lee-zixu.github.io/">Zixu&#160;Li</a><sup>1</sup>,
  <a target="_blank" href="https://zivchen-ty.github.io/">Zhiwei&#160;Chen</a><sup>1</sup>,
  Chunxiao&#160;Wang<sup>3</sup>,
  <a target="_blank" href="https://xuemengsong.github.io/">Xuemeng&#160;Song</a><sup>2</sup>,
  <a target="_blank" href="https://faculty.sdu.edu.cn/huyupeng1/zh_CN/index.htm">Yupeng&#160;Hu</a><sup>1&#9993</sup>,
  <a target="_blank" href="https://liqiangnie.github.io">Liqiang&#160;Nie</a><sup>4</sup>
  </div>
  <sup>1</sup>School of Software, Shandong University &#160&#160&#160</span>  <br>
     <sup>2</sup>School of Computer Science and Technology, Shandong University &#160&#160&#160</span> <br>
 <sup>3</sup>Key Laboratory of Computing Power Network and Information Security, Ministry of Education, Qilu University of Technology (Shandong Academy of Sciences) &#160&#160&#160</span> <br>
  <sup>4</sup>School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen) &#160&#160&#160</span> 
  <br />
  <sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
  <br/>

  <p>
      <a href="https://ieeexplore.ieee.org/document/10888153"><img alt='Paper' src="https://img.shields.io/badge/Paper-IEEE-green.svg?style=flat-square"></a>
    <a href="https://zhihfu.github.io/PAIR.github.io/"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>
    <a href="https://zhihfu.github.io"><img src="https://img.shields.io/badge/Author Page-blue.svg" alt="Author Page"></a>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?&logo=pytorch&logoColor=white"></a>
    <img src="https://img.shields.io/badge/python-≥3.8-blue?style=flat-square" alt="Python">
    <a href="https://github.com/ZhihFu/PAIR"><img alt='stars' src="https://img.shields.io/github/stars/ZhihFu/PAIR?style=social"></a>
  </p>

  <p>
    <b>Official Repository:</b> This is an open-source implementation of the paper "PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval".
  </p>
</div>

## 📌 Introduction

**PAIR** (Complementarity-guided Disentanglement for Composed Image Retrieval) is our proposed framework for Composed Image Retrieval. By exploring the inherent relationships between multimodal queries and target images, PAIR effectively disentangles the visual and textual representations guided by their complementarity. This approach achieves precise alignment and retrieval by addressing the semantic entanglement that limits existing methods.

[⬆ Back to top](#top)

## 📢 News
* **[2025-03]** 🔥 The paper *"PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval"* has been accepted by ICASSP 2025!
* **[2024-09]** 🚀 Release all codes of PAIR!

[⬆ Back to top](#top)


## 🏗️ Architecture

<p align="center">
  <img src="assets/pair.png" alt="PAIR architecture" width="1000">
  <figcaption><strong>Figure 1.</strong> The overall framework of PAIR. <em>(Update with actual architecture figure)</em></figcaption>
</p>

[⬆ Back to top](#top)

## 🏃‍♂️ Experiment Results

> 💡 <span style="color:#2980b9;">**Note:**</span> <br>
> 🎯 We evaluate PAIR extensively on three standard CIR datasets. Please refer to our main paper for detailed comparative analyses and ablation studies.

### CIR Task Performance

#### CIRR:
<p align="center">
  <img src="assets/cirr.png" alt="PAIR_cirr" width="1000">
<caption><strong>Table 1.</strong> Performance comparison on the CIRR test set in terms of R@K (%) and Rsub@K (%).</caption>
</p>


#### FashionIQ & Shoes:
<p align="center">
  <img src="assets/fiqshoes.png" alt="PAIR_fiq" width="1000">
<caption><strong>Table 2.</strong> Performance comparison on FashionIQ and Shoes validation sets in terms of R@K (%).</caption>
</p>

[⬆ Back to top](#top)

---

## Table of Contents

- [Introduction](#-introduction)
- [News](#-news)
- [Architecture](#-architecture)
- [Experiment Results](#-experiment-results)
- [Install](#-install)
- [Data Preparation](#-data-preparation)
- [Quick Start](#-quick-start)
- [Acknowledgement](#-acknowledgement)
- [Related Projects](#-related-projects)
- [Citation](#-citation)
- [Support & Contributing](#-support--contributing)

---

## 📦 Install

**1. Clone the repository**

```bash
git clone [https://github.com/ZhihFu/PAIR](https://github.com/ZhihFu/PAIR)
cd PAIR
```

**2. Setup Python Environment**
The code is evaluated with Python 3.8.10 and PyTorch 2.0.0 on an NVIDIA Tesla T4 16G platform. We recommend using Anaconda to create an isolated virtual environment:

```bash
conda create -n pair python=3.8.10
conda activate pair

# Install PyTorch
pip install torch==2.0.0 torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install core dependencies (add other requirements as needed)
pip install -r requirements.txt
```

[⬆ Back to top](#top)

---

## 📂 Data Preparation

We evaluated our framework on three standard datasets: CIRR, FashionIQ, and Shoes. Please download the datasets first.

<details>
<summary><b>Click to expand: CIRR Dataset Directory Structure</b></summary>

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
</details>

<details>
<summary><b>Click to expand: FashionIQ Dataset Directory Structure</b></summary>

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

</details>

<details>
<summary><b>Click to expand: Shoes Dataset Directory Structure</b></summary>

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

</details>

[⬆ Back to top](#top)

---

## 🚀 Quick Start
**1. Training**
You can train PAIR on CIRR, FashionIQ, or Shoes by specifying the dataset and the corresponding paths.
```bash
python3 train.py \
  --dataset cirr \
  --model_dir "./checkpoints/PAIR_CIRR" \
  --cirr_path "/path/to/CIRR" \
  --fashioniq_path "/path/to/FashionIQ" \
  --shoes_path "/path/to/Shoes"
```

**Arguments:**
```
--dataset <str>: Dataset to use, options: ['cirr', 'fashioniq', 'shoes']

--cirr_path <str>: Path to the CIRR dataset root folder

--fashioniq_path <str>: Path to the FashionIQ dataset root folder

--shoes_path <str>: Path to the Shoes dataset root folder

--model_dir <str>: Path to save checkpoints and logs
```

**2. Testing**
To generate the predictions file for uploading on the CIRR Evaluation Server using our model, please execute the following command:
```bash
python src/cirr_test_submission.py checkpoints/PAIR_CIRR.pt
```

(Where checkpoints/PAIR_CIRR.pt is the path to your trained PAIR checkpoint).

[⬆ Back to top](#top)

---


## 🤝 Acknowledgement
This codebase is heavily inspired by and built upon [CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir). We express our sincere gratitude to these open-source contributions!

[⬆ Back to top](#top)

---


## 🔗 Related Projects
Ecosystem & Other Works from our Team

<table style="width:100%; border:none; text-align:center; background-color:transparent;">
<tr style="border:none;">
<td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
<img src="./assets/logos/airknow-logo.png" alt="Air-Know" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;">
<b>Air-Know (CVPR'26)</b>


<span style="font-size: 0.9em;">
<a href="https://zhihfu.github.io/Air-Know.github.io/" target="_blank">Web</a> |
<a href="https://github.com/zhihfu/Air-Know" target="_blank">Code</a>
</span>
</td>
<td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
<img src="./assets/logos/habit-logo.png" alt="HABIT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;">


<b>HABIT (AAAI'26)</b>


<span style="font-size: 0.9em;">
<a href="https://lee-zixu.github.io/HABIT.github.io/" target="_blank">Web</a> |
<a href="https://github.com/Lee-zixu/HABIT" target="_blank">Code</a> |
<a href="https://ojs.aaai.org/index.php/AAAI/article/view/37608" target="_blank">Paper</a>
</span>
</td>
<td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
<img src="./assets/logos/retrack-logo.png" alt="ReTrack" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;">


<b>ReTrack (AAAI'26)</b>


<span style="font-size: 0.9em;">
<a href="https://lee-zixu.github.io/ReTrack.github.io/" target="_blank">Web</a> |
<a href="https://github.com/Lee-zixu/ReTrack" target="_blank">Code</a> |
<a href="https://ojs.aaai.org/index.php/AAAI/article/view/39507" target="_blank">Paper</a>
</span>
</td>
<td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
<img src="./assets/logos/intent-logo.png" alt="INTENT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;">


<b>INTENT (AAAI'26)</b>


<span style="font-size: 0.9em;">
<a href="https://zivchen-ty.github.io/INTENT.github.io/" target="_blank">Web</a> |
<a href="https://github.com/ZivChen-Ty/INTENT" target="_blank">Code</a> |
<a href="https://ojs.aaai.org/index.php/AAAI/article/view/39181" target="_blank">Paper</a>
</span>
</td>

</tr>

<tr style="border:none;">

<td style="width:30%; border:none; vertical-align:top; padding-top:30px;">

<img src="./assets/logos/hud-logo.png" alt="HUD" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;">


<b>HUD (ACM MM'25)</b>


<span style="font-size: 0.9em;">

<a href="https://zivchen-ty.github.io/HUD.github.io/" target="_blank">Web</a> |
<a href="https://github.com/ZivChen-Ty/HUD" target="_blank">Code</a> |
<a href="https://dl.acm.org/doi/10.1145/3746027.3755445" target="_blank">Paper</a>
</span>
</td>
<td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
<img src="./assets/logos/offset-logo.png" alt="OFFSET" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;">


<b>OFFSET (ACM MM'25)</b>


<span style="font-size: 0.9em;">
<a href="https://zivchen-ty.github.io/OFFSET.github.io/" target="_blank">Web</a> |
<a href="https://github.com/ZivChen-Ty/OFFSET" target="_blank">Code</a> |
<a href="https://dl.acm.org/doi/10.1145/3746027.3755366" target="_blank">Paper</a>
</span>
</td>
<td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
<img src="./assets/logos/encoder-logo.png" alt="ENCODER" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;">


<b>ENCODER (AAAI'25)</b>


<span style="font-size: 0.9em;">
<a href="https://sdu-l.github.io/ENCODER.github.io/" target="_blank">Web</a> |
<a href="https://github.com/Lee-zixu/ENCODER" target="_blank">Code</a> |
<a href="https://ojs.aaai.org/index.php/AAAI/article/view/32541" target="_blank">Paper</a>
</span>
</td>
</tr>
</table>

---

## 📝⭐️ Citation

If you find our work or this code useful in your research, please consider leaving a Star⭐️ or Citing📝 our paper 🥰. Your support is our greatest motivation!

```
@article{PAIR2025,
    title={PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval},
    author={Fu, Zhiheng and Li, Zixu and Chen, Zhiwei and Wang, Chunxiao and Song, Xuemeng and Hu, Yupeng and Nie, Liqiang},
    journal={IEEE},
    year = {2025}
}
```
[⬆ Back to top](#top)

---
## 🫡 Support & Contributing

For any questions, issues, or feedback, please open an [issue](https://github.com/ZhihFu/PAIR/issues) on GitHub or reach out to us at fuzhiheng8@gmail.com

[⬆ Back to top](#top)

---

<div align="center">

**If this project helps you, please leave a Star!**

[![GitHub stars](https://img.shields.io/github/stars/ZhihFu/Air-Know?style=social)](https://github.com/ZhihFu/PAIR)


</div>



