# Awesome Continual Learning of Vision-Language Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
> A curated list of papers, codebases, and datasets for continual learning in vision-language models (VLMs).  
> *Last updated: [2025-03-11]*

---

## 📜 Overview  
This repository aims to **systematically organize** research advancements, and discussions on continual learning for vision-language models (VLMs). Vision-language models (e.g., CLIP, ALIGN) have shown remarkable progress, but adapting them to evolving data streams without catastrophic forgetting remains a critical challenge. This repo serves as a **community-driven hub** for:  
- 🎯 **Tracking SOTA methods**: Papers, preprints, and codebases for continual VLM learning.  
- 🌍 **Fostering collaboration**: Encouraging open discussions and contributions.  

## 📄 Papers  
*Sorted chronologically (newest first).*  

### 2025
* (CVPR) Synthetic Data is an Elegant GIFT for Continual Vision-Language Models. [[PDF](https://arxiv.org/pdf/2503.04229)][[CODE](https://github.com/Luo-Jiaming/GIFT_CL)]
* (TPAMI) Learning without Forgetting for Vision-Language Models. [[PDF](https://arxiv.org/pdf/2305.19270)][[CODE](https://github.com/zhoudw-zdw/PROOF/)]
* (arXiv) No Images, No Problem: Retaining Knowledge in Continual VQA with Questions-Only Memory. [[PDF](https://arxiv.org/pdf/2502.04469.pdf)][[CODE](https://github.com/IemProg/QUAD)]

### 2024
* (CVPR) AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning. [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_AttriCLIP_A_Non-Incremental_Learner_for_Incremental_Knowledge_Learning_CVPR_2023_paper.pdf)][[CODE](https://gitee.com/mindspore/models/tree/master/research/)]
* (CVPR) Generative Multi-modal Models are Good Class-Incremental Learners. [[PDF](https://arxiv.org/pdf/2403.18383)][[CODE](https://github.com/DoubleClass/GMM)]
* (CVPR) Pre-trained Vision and Language Transformers Are Few-Shot Incremental Learners. [[PDF](https://arxiv.org/pdf/2404.02117)][[CODE](https://github.com/KU-VGI/PriViLege)]
* (AAAI) Continual Vision-Language Retrieval via Dynamic Knowledge Rectification. [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/29054/29997)]
* (AAAI) Learning Task-Aware Language-Image Representation for Class-Incremental Object Detection. [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/28537)]
* (ECCV) Mind the Interference: Retaining Pre-trained Knowledge in Parameter Efficient Continual Learning of Vision-Language Models (DIKI). [[PDF](https://arxiv.org/pdf/2407.05342)] [[CODE](https://github.com/lloongx/DIKI)]
* (ECCV) Adapt without Forgetting: Distill Proximity from Dual Teachers in Vision-Language Models. [[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07052.pdf)]
* (ECCV) Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models. [[PDF](https://arxiv.org/pdf/2403.09296)]
* (ECCV) Class-Incremental Learning with CLIP: Adaptive Representation Adjustment and Parameter Fusion. [[PDF](https://arxiv.org/pdf/2407.14143)] [[CODE](https://github.com/linlany/RAPF)]
* (NeurIPS) Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models (RAIL). [[PDF](https://arxiv.org/pdf/2406.18868)][[CODE](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning)]
* (ACM MM) Low-rank Prompt Interaction for Continual Vision-language Retrieval. [[PDF](https://arxiv.org/pdf/2501.14369)][[CODE](https://github.com/Kelvin-ywc/LPI)]
* (IJCAI) Continual Multimodal Knowledge Graph Construction. [[PDF](https://arxiv.org/pdf/2305.08698)] [[CODE](https://github.com/zjunlp/ContinueMKGC)]
* (arXiv) CLAP4CLIP: Continual Learning with Probabilistic Finetuning for Vision-Language Models. [[PDF](https://arxiv.org/pdf/2403.19137.pdf)][[CODE](https://github.com/srvCodes/clap4clip)]
* (arXiv) LW2G: Learning Whether to Grow for Prompt-based Continual Learning. [[PDF](https://arxiv.org/pdf/2409.18860.pdf)][[CODE](https://github.com/raian08/lw2g)]
* (arXiv) ATLAS: Adapter-Based Multi-Modal Continual Learning with a Two-Stage Learning Strategy. [[PDF](https://arxiv.org/pdf/2410.10923.pdf)][[CODE](https://github.com/lihong2303/ATLAS)]


### 2023
* (AAAI) Symbolic Replay: Scene Graph as Prompt for Continual Learning on VQA Task. [[PDF](https://arxiv.org/pdf/2208.12037)] [[CODE](https://github.com/showlab/CLVQA)]
* (CVPR) Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models (ZSCL). [[PDF](https://arxiv.org/abs/2303.06628)] [[CODE](https://github.com/Thunderbeee/ZSCL/tree/main)]
* (CVPR) Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters. [[PDF](https://arxiv.org/abs/2403.11549)] [[CODE](https://github.com/JiazuoYu/MoE-Adapters4CL)]
* (CVPR) VQACL: A Novel Visual Question Answering Continual Learning Setting. [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_VQACL_A_Novel_Visual_Question_Answering_Continual_Learning_Setting_CVPR_2023_paper.pdf)] [[CODE](https://github.com/zhangxi1997/VQACL)]
* (ICCV) CTP: Towards Vision-Language Continual Pretraining via Compatible Momentum Contrast and Topology Preservation [[PDF](https://arxiv.org/pdf/2308.07146)] [[CODE](https://github.com/KevinLight831/CTP)]
* (ICCV) Decouple Before Interact: Multi-Modal Prompt Learning for Continual Visual Question Answering [[PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Qian_Decouple_Before_Interact_Multi-Modal_Prompt_Learning_for_Continual_Visual_Question_ICCV_2023_paper.pdf)]
* (ICCVW) Multimodal Parameter-Efficient Few-Shot Class Incremental Learning. [[PDF](https://arxiv.org/pdf/2303.04751)]
* (ICML) Continual vision-language representation learning with off-diagonal information. [[PDF](https://proceedings.mlr.press/v202/ni23c/ni23c.pdf)] [[CODE](https://github.com/Thunderbeee/ZSCL/tree/main)]
* (ACM MM) Multi-Domain Lifelong Visual Question Answering via Self-Critical Distillation. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3581783.3612121)]
* (arXiv) Class Incremental Learning with Pre-trained Vision-Language Models. [[PDF](https://arxiv.org/pdf/2310.20348.pdf)]

### 2022
* (Neurips) S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning. [[PDF](https://arxiv.org/abs/2207.12819)] [[CODE](https://github.com/iamwangyabin/S-Prompts)]
* (Neurips) Climb: A continual learning benchmark for vision-and-language tasks. [[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/bd3611971089d466ab4ca96a20f7ab13-Paper-Datasets_and_Benchmarks.pdf)] [[CODE](https://github.com/GLAMOR-USC/CLiMB)]
* (ECCV) Generative Negative Text Replay for Continual Vision-Language Pretraining. [[PDF](https://arxiv.org/pdf/2210.17322)] 
* (arxiv) Continual-CLIP: CLIP is an Efficient Continual Learner. [[PDF](https://arxiv.org/abs/2210.03114)] [[CODE](https://github.com/vgthengane/Continual-CLIP/tree/master)]

---

## 🗂️ Datasets  
*Datasets are categorized by their application tasks and continual learning scenarios. Key attributes include:*  
- **Task Type**: Classification, VQA, Detection, Segmentation, etc.  
- **CL Scenario**: Domain-Incremental (DIL), Task-Incremental (TIL), Class-Incremental (CIL).  
- **# Tasks**: Number of sequential tasks.  
- **Scale**: Approximate data size (images/text pairs).

| Dataset       | Task Type       | CL Scenario | # Tasks | Scale       | Link         |  
|---------------|-----------------|-------------|---------|-------------|--------------|  
| **COCO-CL**   | Detection       | CIL         | 10      | 120K images | [Download](#)|  
| **Flickr30K-Stream** | Retrieval | TIL         | 5       | 30K pairs   | [Download](#)|  
| **VQA-CL**    | VQA             | DIL         | 8       | 200K QA     | [Download](#)|  
| **ADE20K-CL** | Segmentation    | CIL         | 6       | 25K images  | [Download](#)|  
| **CIFAR-100S**| Classification  | CIL         | 20      | 60K images  | [Download](#)|  
| **CLiMB**     | VQA             | DIL         | 4       | *          | [Download](https://github.com/GLAMOR-USC/CLiMB)|  

Dataset	Task Type	CL Scenario	Modality	Domain	# Tasks	Scale	Metrics	Link
CDDB	Classification	DIL (Domain-Incremental)	Image	Multi-Domain	10	~50K images	Accuracy	Link
CORe50	Classification	DIL	Video	Robotics	10	50K video clips	Accuracy	Link
DomainNet	Classification	DIL	Image	6 Domains (e.g., Sketch, Painting)	6	600K images	Accuracy	Link
Climb	Multimodal Understanding	TIL (Task-Incremental)	Image+Text	General	4	VQAv2 (200K QA), NLVR2 (100K pairs), SNLI-VE (500K pairs), VCR (290K QA)	VQA Accuracy, NLVR2 F1	Link
Conceptual 12M	Segmentation	CIL / Instance-Incremental	Image+Text	Open-World	N/A	12M image-text pairs	mIoU, mAP	Link
ImageNet-100/1K	Classification	CIL / DIL / Task-Agnostic	Image	Natural Images	10-100	130K-1.3M images	Top-1/5 Accuracy	Link
CIFAR-100	Classification	CIL	Image	Natural Images	20	60K images	Accuracy	Link
TinyImageNet	Classification	CIL	Image	Natural Images	200	100K images	Accuracy	Link
MDL-VQA	VQA	DIL	Image+Text	5 Visual Domains (Art, Abstract, etc.)	5	~150K QA	VQA Accuracy	Link
P9D	Multimodal Retrieval	DIL / TIL	Image+Text	9 Industries	9	1M+ image-text pairs	Retrieval mAP	Link
CLVQA	VQA	Scene/Function-Incremental	Image+Text	Scenes & Functions	2	~100K QA	Scene/Function Accuracy	Link
COCO-CL	Detection/Segmentation	CIL	Image+Annotations	Natural Scenes	80	200K+ instances	mAP (Detection), mIoU (Segmentation)	Link
ADE20K-CL	Segmentation	CIL	Image+Annotations	Indoor/Outdoor	150	25K images	mIoU	Link

 ### Notes:  
- **CIL (Class-Incremental Learning)**: New classes are added incrementally.  
- **TIL (Task-Incremental Learning)**: Tasks are disjoint, and task IDs are provided during inference.  
- **DIL (Domain-Incremental Learning)**: Data distribution shifts across domains (e.g., weather, lighting).  

---

## 🌟 Related Resources  
- [Awesome Continual Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)  
- [Awesome Vision-Language Models](https://github.com/jingyi0000/VLM_survey)  

---

## 🛠️ Maintenance  
*Maintainer*: [Yuyang Liu](https://github.com/YuyangSunshine), Qiuhe Hong and Dipam Goswami
*Contact*: sunshineliuyuyang@gmail.com  
*Star the repo to show support!* ⭐  

*Acknowledgments: This repo builds on the efforts of the open-source community.*  


