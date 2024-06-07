# Comprehensive-Advances-in-Deepfake-Detection-Spanning-Diverse-Modalities

![Awesome](https://awesome.re/badge.svg)

This repository provides a comprehensive investigation of advanced single-modal and multi-modal deepfake detection.

## Relevant Surveys

### Deepfake/AIGC Generation and Detection
\[arxiv 2024\] Deepfake Generation and Detection: A Benchmark and Survey [Project](https://arxiv.org/abs/2403.17881) [Project](https://github.com/flyingby/Awesome-Deepfake-Generation-and-Detection)

\[arxiv 2024\] Detecting Multimedia Generated by Large AI Models: A Survey [Paper](https://arxiv.org/abs/2402.00045) [Project](https://github.com/Purdue-M2/Detect-LAIM-generated-Multimedia-Survey)

\[ECAI 2023\] GAN-generated Faces Detection: A Survey and New Perspectives [Paper](https://arxiv.org/abs/2202.07145)

\[NeurIPS 2023\] DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection [Paper](https://arxiv.org/abs/2307.01426) [Project](https://github.com/SCLBD/DeepfakeBench)

\[arxiv 2023\] Deepfake detection: A comprehensive study from the reliability perspective [Paper](https://arxiv.org/abs/2211.10881)

\[IJCV 2022\] Countering Malicious DeepFakes: Survey, Battleground, and Horizon [Paper](https://arxiv.org/abs/2103.00218) [Project](https://www.xujuefei.com/dfsurvey)

### Multi-modal Fact-checking
\[EMNLP 2023\] Multimodal automated fact-checking: A survey [Paper](https://arxiv.org/abs/2305.13507)

## Facial Deepfake Datasets
### Single-modal GAN-generated Datasets
|Dataset|Year|Task|Manipulated Modality|\# of real videos|\# of fake videos|Paper|Link|
|:-:|:-:|:-:|:-:|:-:|:-:|-|-|
|FaceForensics++(FF++)|2019|Classification|Visual|1,000|4,000|[FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)|[Download](https://github.com/ondyari/FaceForensics)|
|DFD|2019|Classification|Visual|363|3,068|[Contributing data to deepfake detection research](https://blog.research.google/2019/09/contributing-datato-deepfake-detection.html)|[Download](https://github.com/ondyari/FaceForensics)|
|DFFD|2020|Classification|Visual|1,000|3,000|[On the Detection of Digital Face Manipulation](https://arxiv.org/abs/1910.01717)|[Download](https://cvlab.cse.msu.edu/dffd-dataset.html)|
|FaceShifter|2020|Classification|Visual|-|1,000|[FaceShifter: Towards High Fidelity And Occlusion Aware Face Swapping](https://arxiv.org/abs/1912.13457)|[Download](https://github.com/ondyari/FaceForensics)|
|DFDC|2020|Classification|Visual|23,654|104,500|[The DeepFake Detection Challenge (DFDC) Dataset](https://arxiv.org/abs/2006.07397)|[Download](https://ai.meta.com/datasets/dfdc/)|
|Celeb-DF|2020|Classification|Visual|590|5,639|[Celeb-df: A large-scale challenging dataset for deepfake forensics](https://arxiv.org/abs/1909.12962)|[Download](https://github.com/yuezunli/celeb-deepfakeforensics)|
|DeeperForensics-1.0|2020|Classification|Visual|50,000|10,000|[DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection](https://www.arxiv.org/abs/2001.03024)|[Download](https://github.com/EndlessSora/DeeperForensics-1.0)|
|WildDeepfake|2020|Classification|Visual|3,805|3,509|[WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection](https://dl.acm.org/doi/10.1145/3394171.3413769)|[Download](https://github.com/deepfakeinthewild/deepfake-in-the-wild)|
|KoDF|2020|Classification|Visual|62,166|175,776|[KoDF: A Large-scale Korean DeepFake Detection Dataset](https://arxiv.org/abs/2103.10094)|[Download](https://deepbrainai-research.github.io/kodf/)|
|FFIW_10k|2021|Classification & Spatial Grounding|Visual|10,000|10,000|[Face Forensics in the Wild](https://arxiv.org/abs/2103.16076)|[Download](https://github.com/tfzhou/FFIW)|
|ForgeryNet|2021|Classification & Spatial Grounding & Temporal Grounding|Visual|99,630|12,1617|[Forgerynet: A versatile benchmark for comprehensive forgery analysis](https://arxiv.org/abs/2103.05630)|[Download](https://yinanhe.github.io/projects/forgerynet.html)|
|DF-Platter|2023|Classification|Visual|133,260|132,496|[DF-Platter: Multi-Face Heterogeneous Deepfake Dataset](https://openaccess.thecvf.com/content/CVPR2023/papers/Narayan_DF-Platter_Multi-Face_Heterogeneous_Deepfake_Dataset_CVPR_2023_paper.pdf)|[Download](https://iab-rubric.org/df-platter-database)|
### Single-modal Diffusion-generated Datasets
|Dataset|Year|Task|Manipulated Modality|\# of real images|\# of fake images|Paper|Link|
|:-:|:-:|:-:|:-:|:-:|:-:|-|-|
|DeepFakeFace|2023|Classification|Visual|30,000|90,000|[Robustness and Generalizability of Deepfake Detection: A Study with Diffusion Models](https://arxiv.org/abs/2309.02218)|[Download](https://github.com/OpenRL-Lab/DeepFakeFace)|
|DiFF|2024|Classification|Visual|23,661|537,466|[Diffusion Facial Forgery Detection](https://arxiv.org/abs/2401.15859)|[Download](https://github.com/xaCheng1996/DiFF)|
|DiffusionFace|2024|Classification|Visual|30,000|600,000|[DiffusionFace: Towards a Comprehensive Dataset for Diffusion-Based Face Forgery Analysis](https://arxiv.org/abs/2403.18471)|[Download](https://github.com/Rapisurazurite/DiffFace)|
|DiffusionDB-Face|2024|Classification|Visual|94,120|24,794|[Diffusion Deepfake](https://arxiv.org/abs/2404.01579)|[Download](https://surrey-uplab.github.io/research/diffusion_deepfake/)|
|JourneyDB-Face|2024|Classification|Visual|94,120|87,833|[Diffusion Deepfake](https://arxiv.org/abs/2404.01579)|[Download](https://surrey-uplab.github.io/research/diffusion_deepfake/)|
### Multi-modal Audio-Visual Datasets
|Dataset|Year|Task|Manipulated Modality|\# of real videos|\# of fake videos|Paper|Link|
|:-:|:-:|:-:|:-:|:-:|:-:|-|-|
|FakeAVCeleb|2021|Classification|Visual & Audio|500|19,500|[FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset](https://arxiv.org/abs/2108.05080)|[Download](https://sites.google.com/view/fakeavcelebdash-lab/)|
|TMC|2022|Classification & Temporal Grounding|Visual & Audio|2,563|4,380|[Trusted Media Challenge Dataset and User Study](https://arxiv.org/abs/2201.04788)|-|
|LAV-DF|2022|Classification & Temporal Grounding|Visual & Audio|36,431|99,873|[Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization](https://arxiv.org/abs/2204.06228)|[Download](https://github.com/ControlNet/LAV-DF)|
|DefakeAVMiT|2023|Classification|Visual & Audio|540|6,480|[AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake](https://doi.org/10.1109/TIFS.2023.3262148)|-|
|AV-Deepfake1M|2023|Classification & Temporal Grounding|Visual & Audio|286,721|860,039|[AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset](https://arxiv.org/abs/2311.15308)|[Download](https://github.com/ControlNet/AV-Deepfake1M)|
|MMDFD|2023|Classification|Visual & Audio & Text|1,500|5,000|[MMDFD- A Multimodal Custom Dataset for Deepfake Detection](https://doi.org/10.1145/3607947.3608013)|-|

### Multi-modal Text-Visual Datasets
|Dataset|Year|Task|Manipulated Modality|\# of real image-text pairs|\# of fake image-text pairs|Paper|Link|
|:-:|:-:|:-:|:-:|:-:|:-:|-|-|
|DGM4|2023|Classification & Spatial Grounding & Text Grounding|Visual & Text|77,426|152,574|[DGM4: Detecting and Grounding Multi-Modal Media Manipulation and Beyond](https://arxiv.org/abs/2309.14203)|[Download](https://github.com/rshaojimmy/MultiModal-DeepFake)|
## Single-modal (Visual) Deepfake Detection
### Passive Detection
#### Naive Detection
##### Visual Artifacts
1. **Non-facial** \[CVPR 2024\] Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection [Paper](https://arxiv.org/abs/2312.10461)
2. \[CVPR 2024\] LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection [Paper](https://arxiv.org/abs/2401.13856)
3. **Non-facial** \[arXiv 2024\] Data-Independent Operator: A Training-Free Artifact Representation Extractor for Generalizable Deepfake Detection [Paper](https://arxiv.org/abs/2403.06803)
4. **Non-facial** \[arXiv 2024\] A Single Simple Patch is All You Need for AI-generated Image Detection [Paper](https://arxiv.org/abs/2402.01123)
5. \[arXiv 2024\] GenFace: A Large-Scale Fine-Grained Face Forgery Benchmark and Cross Appearance-Edge Learning [Paper](https://arxiv.org/abs/2402.02003)
6. \[TMM 2023\] GLFF: Global and Local Feature Fusion for AI-synthesized Image Detection [Paper](https://arxiv.org/abs/2211.08615)
7. Non-facial \[CVPRW 2023\] Intriguing properties of synthetic images: from generative adversarial networks to diffusion models [Paper](https://arxiv.org/abs/2304.06408)
8. **Non-facial**, \[arXiv 2023\] Diffusion Noise Feature: Accurate and Fast Generated Image Detection [Paper](https://arxiv.org/pdf/2312.02625)
9. \[CVPR 2021\] Multi-attentional Deepfake Detection [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Multi-Attentional_Deepfake_Detection_CVPR_2021_paper.pdf) 
10. \[CVPR 2020\] Global Texture Enhancement for Fake Face Detection In the Wild [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Global_Texture_Enhancement_for_Fake_Face_Detection_in_the_Wild_CVPR_2020_paper.pdf)
11. \[ICCV 2019\] FaceForensics++: Learning to Detect Manipulated Facial Images [Paper](https://arxiv.org/abs/1901.08971)
12. \[WIFS 2018\] Mesonet: a compact facial video forgery detection network [Paper](https://ieeexplore.ieee.org/document/8630761/)
##### Consistency-based
1. \[IJCV 2024\] Learning Spatiotemporal Inconsistency via Thumbnail Layout for Face Deepfake Detection [Paper](https://arxiv.org/abs/2403.10261)
2. \[ICCV 2023\] TALL: Thumbnail Layout for Deepfake Video Detection [Paper](https://arxiv.org/abs/2307.07494)
3. \[\] [Paper]()
4. \[\] [Paper]()
5. \[\] [Paper]()
6. \[\] [Paper]()
7. \[\] [Paper]()
8. \[ACM MM Asia 2022\] Latent pattern sensing: Deepfake video detection via predictive representation learning [Paper](https://doi.org/10.1145/3469877.3490586)
9. \[CVPR 2021\] Lips don’t lie: A generalisable and robust approach to face forgery detection [Paper](https://arxiv.org/abs/2012.07657)
10. \[ACM MM 2020\] DeepRhythm: Exposing DeepFakes with Attentional Visual Heartbeat Rhythms [Paper](https://arxiv.org/abs/2006.07634)
11. \[WIFS 2018\] In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking [Paper](https://ieeexplore.ieee.org/document/8630787/)
#### Advanced Detection
##### Input Level: Data Augmentation
##### Input Level: Frequency-based
##### Model Level: Transformer-based
##### Model Level: VLM-based
##### Learning Level: Advanced Loss
##### Learning Level: Disentangled Representation
##### Learning Level: Reconstruction
##### Learning Level: Manipulation Localization
##### Learning Level: Identity Discrepancy
##### Learning Level: Knowledge Distillation
##### Learning Level: Fine-grained Prediction
##### Learning Level: Reasoning


##### Diffusion Model Detection
### Proactive Detection
#### Proactive Methods for GANs
#### Proactive Methods for Diffusion Models

## Multi-modal Audio-Visual Deepfake Detection
### Audio-Visual Detection
#### Independent Learning
#### Joint Learning
##### Intermediate Fusion: Cross-Attention
##### Late Fusion: Concatenation & Addition
##### Late Fusion: Attention
##### Late Fusion: MLP Mixer Layer
##### Multi-task Strategy
##### Regularization
#### Matching-based Learning
#### Others

## Multi-modal Visual-Text Deepfake Detection

## Trustworthy Deepfake Detection
### Adversarial Attack
### Backdoor Attack
### Discrepancy Minimization
### Defense Strategies

## Other Useful Sources



