# Comprehensive Advances in Deepfake Detection Spanning Diverse Modalities

![Awesome](https://awesome.re/badge.svg)

This repository provides a comprehensive investigation of advanced single-modal and multi-modal deepfake detection.

## Content
---
- [Relevant Surveys](#relevant-surveys)
     - [Deepfake/AIGC Generation and Detection](#deepfakeaigc-generation-and-detection)
     - [Multi-modal Fact-checking](#multi-modal-fact-checking)
- [Facial Deepfake Datasets](#facial-deepfake-datasets)
     - [Single-modal GAN-generated Datasets](#single-modal-gan-generated-datasets)
     - [Single-modal Diffusion-generated Datasets](#single-modal-diffusion-generated-Datasets)
     - [Multi-modal Audio-Visual Datasets](#multi-modal-audio-visual-datasets)
     - [Multi-modal Text-Visual Datasets](#multi-modal-text-visual-Datasets)
- [Single-modal (Visual) Deepfake Detection](#single-modal-visual-deepfake-detection)
     - [Passive Detection](#passive-detection)
          - [Naive Detection](#naive-detection)
               - [Visual Artifacts](#visual-artifacts)
               - [Consistency-based](#consistency-based)
          - [Advanced Detection](#advanced-detection)
               - [Input Level](#input-level)
                    - [Data Augmentation](#data-augmentation)
                    - [Frequency-based](#frequency-based)
               - [Model Level](#model-level)
                    - [Transformer-based](#transformer-based)
                    - [VLM-based](#vlm-based)
               - [Learning Level](#learning-level)
     - [Proactive Detection](#proactive-detection)
          - [Proactive Methods for GANs](#proactive-methods-for-gans)
          - [Proactive Methods for Diffusion Models](#proactive-methods-for-diffusion-models)
## Relevant Surveys

### Deepfake/AIGC Generation and Detection
\[arxiv 2024\] Deepfake Generation and Detection: A Benchmark and Survey [Paper](https://arxiv.org/abs/2403.17881) [Project](https://github.com/flyingby/Awesome-Deepfake-Generation-and-Detection)

\[arxiv 2024\] Detecting Multimedia Generated by Large AI Models: A Survey [Paper](https://arxiv.org/abs/2402.00045) [Project](https://github.com/Purdue-M2/Detect-LAIM-generated-Multimedia-Survey)

\[ECAI 2023\] GAN-generated Faces Detection: A Survey and New Perspectives [Paper](https://arxiv.org/abs/2202.07145)

\[NeurIPS 2023\] DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection [Paper](https://arxiv.org/abs/2307.01426) [Project](https://github.com/SCLBD/DeepfakeBench)

\[arxiv 2023\] Deepfake detection: A comprehensive study from the reliability perspective [Paper](https://arxiv.org/abs/2211.10881)

\[IJCV 2022\] Countering Malicious DeepFakes: Survey, Battleground, and Horizon [Paper](https://arxiv.org/abs/2103.00218) [Project](https://www.xujuefei.com/dfsurvey)

### Multi-modal Fact-checking
\[EMNLP 2023\] Multimodal automated fact-checking: A survey [Paper](https://arxiv.org/abs/2305.13507)

---
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

---
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
2. \[arxiv 2024\] Compressed Deepfake Video Detection Based on 3D Spatiotemporal Trajectories [Paper](https://arxiv.org/abs/2404.18149)
3. \[ICCV 2023\] TALL: Thumbnail Layout for Deepfake Video Detection [Paper](https://arxiv.org/abs/2307.07494)
4. \[CVPR 2023\] AltFreezing for More General Video Face Forgery Detection [Paper](https://arxiv.org/abs/2307.08317)
5. \[TCSVT 2023\] MRE-Net: Multi-Rate Excitation Network for Deepfake Video Detection [Paper](https://ieeexplore.ieee.org/document/10025759)
6. \[WACV 2023\] TI2Net: Temporal Identity Inconsistency Network for Deepfake Detection [Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_TI2Net_Temporal_Identity_Inconsistency_Network_for_Deepfake_Detection_WACV_2023_paper.pdf)
7. \[ACM MM Asia 2022\] Latent pattern sensing: Deepfake video detection via predictive representation learning [Paper](https://doi.org/10.1145/3469877.3490586)
8. \[CVPR 2021\] Lips don’t lie: A generalisable and robust approach to face forgery detection [Paper](https://arxiv.org/abs/2012.07657)
9. \[ACM MM 2020\] DeepRhythm: Exposing DeepFakes with Attentional Visual Heartbeat Rhythms [Paper](https://arxiv.org/abs/2006.07634)
10. \[WIFS 2018\] In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking [Paper](https://ieeexplore.ieee.org/document/8630787/)
#### Advanced Detection
##### Input Level
###### Data Augmentation
1. \[arXiv 2024\] FreqBlender: Enhancing DeepFake Detection by Blending Frequency Knowledge [Paper](https://arxiv.org/abs/2404.13872)
2. \[ICCV 2023\] SeeABLE: Soft Discrepancies and Bounded Contrastive Learning for Exposing Deepfakes [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Larue_SeeABLE_Soft_Discrepancies_and_Bounded_Contrastive_Learning_for_Exposing_Deepfakes_ICCV_2023_paper.pdf)
3. \[arXiv 2023\] Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection [Paper](https://arxiv.org/abs/2311.11278)
4. \[CVPR 2022\] Detecting Deepfakes with Self-Blended Images [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.pdf)
5. \[CVPR 2022\] Self-supervised Learning of Adversarial Example: Towards Good Generalizations for Deepfake Detection [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Self-Supervised_Learning_of_Adversarial_Example_Towards_Good_Generalizations_for_Deepfake_CVPR_2022_paper.pdf)
6. \[CVPR 2021\] Representative Forgery Mining for Fake Face Detecti [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Representative_Forgery_Mining_for_Fake_Face_Detection_CVPR_2021_paper.pdf)
7. \[ICCV 2021\] Learning Self-Consistency for Deepfake Detection [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Learning_Self-Consistency_for_Deepfake_Detection_ICCV_2021_paper.pdf)
###### Frequency-based
1. \[AAAI 2024\]Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning [Paper](https://arxiv.org/abs/2403.07240)
2. \[ICASSP 2024\] Frequency Masking for Universal Deepfake Detection [Paper](https://arxiv.org/abs/2401.06506)
3. \[CVPR 2023\] Dynamic Graph Learning with Content-guided Spatial-Frequency Relation Reasoning for Deepfake Detection [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Dynamic_Graph_Learning_With_Content-Guided_Spatial-Frequency_Relation_Reasoning_for_Deepfake_CVPR_2023_paper.pdf)
4. \[AAAI 2022\] FrePGAN: Robust Deepfake Detection Using Frequency-level Perturbations [Paper](https://arxiv.org/abs/2202.03347)
5. \[AAAI 2022\] ADD: Frequency Attention and Multi-View based Knowledge Distillation to Detect Low-Quality Compressed Deepfake Images [Paper](https://arxiv.org/abs/2112.03553)
6. \[CVPR 2021\] Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain [Paper](https://arxiv.org/abs/2103.01856)
7. \[CVPR 2021\] Generalizing Face Forgery Detection with High-frequency Features [Paper](https://arxiv.org/abs/2103.12376)
8. \[CVPR 2021\] Frequency-aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection [Paper](https://arxiv.org/abs/2103.09096)
9. \[AAAI 2021\] Local Relation Learning for Face Forgery Detection [Paper](https://arxiv.org/pdf/2105.02577)
10. \[ECCV 2020\] Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues [Paper](https://arxiv.org/abs/2007.09355)
##### Model Level
###### Transformer-based
1. \[arXiv 2024\] A Timely Survey on Vision Transformer for Deepfake Detection [Paper](https://arxiv.org/abs/2405.08463)
2. \[arXiv 2024\] Exploring Self-Supervised Vision Transformers for Deepfake Detection: A Comparative Analysis [Paper](https://arxiv.org/abs/2405.00355)
###### VLM-based
1. \[arXive 2024\] Towards More General Video-based Deepfake Detection through Facial Feature Guided Adaptation for Foundation Model [Paper](https://arxiv.org/abs/2404.05583)
2. \[arxiv 2023\] Forgery-aware Adaptive Vision Transformer for Face Forgery Detection [Paper](https://arxiv.org/abs/2309.11092)
3. \[arXiv 2024\] Mixture of Low-rank Experts for Transferable AI-Generated Image Detection [Paper](https://arxiv.org/abs/2404.04883)
4. \[arXiv 2024\] MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection [Paper](https://arxiv.org/abs/2404.08452)
5. \[CVPR 2023\] AUNet: Learning Relations Between Action Units for Face Forgery Detection [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Bai_AUNet_Learning_Relations_Between_Action_Units_for_Face_Forgery_Detection_CVPR_2023_paper.pdf)
6. \[ICCVW 2023\] Undercover Deepfakes: Detecting Fake Segments in Video [Paper](https://openaccess.thecvf.com/content/ICCV2023W/DFAD/papers/Saha_Undercover_Deepfakes_Detecting_Fake_Segments_in_Videos_ICCVW_2023_paper.pdf)
7. \[arXiv 2023\] DeepFake-Adapter: Dual-Level Adapter for DeepFake Detection [Paper](https://arxiv.org/abs/2306.00863)
##### Learning Level
###### Advanced Loss
1. \[ToMM 2024\] Domain-invariant and Patch-discriminative Feature Learning for General Deepfake Detection [Paper](https://dl.acm.org/doi/10.1145/3657297)
2. \[ICME 2023\] Domain-Invariant Feature Learning for General Face Forgery Detection [Paper](https://ieeexplore.ieee.org/document/10219778/)
3. \[ICDM 2023\] Concentric Ring Loss for Face Forgery Detection [Paper](https://www.computer.org/csdl/proceedings-article/icdm/2023/078800b505/1Ui3cRpq3ug)
###### Disentangled Representation
1. \[CVPR 2024\] Preserving Fairness Generalization in Deepfake Detection [Paper](https://arxiv.org/abs/2402.17229)
2. \[ICCV 2023\] UCF: Uncovering Common Features for Generalizable Deepfake Detection [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_UCF_Uncovering_Common_Features_for_Generalizable_Deepfake_Detection_ICCV_2023_paper.pdf)
3. \[ECCV 2022\] Exploring Disentangled Content Information for Face Forgery Detection [Paper](https://arxiv.org/abs/2207.09202)
###### Reconstruction
1. \[CVPR 2023\] MARLIN: Masked Autoencoder for facial video Representation LearnINg [Paper](https://arxiv.org/abs/2211.06627)
2. \[CVPR 2022\] End-to-End Reconstruction-Classification Learning for Face Forgery Detection [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_End-to-End_Reconstruction-Classification_Learning_for_Face_Forgery_Detection_CVPR_2022_paper.pdf)
3. \[IJCAI 2021\] Beyond the Spectrum: Detecting Deepfakes via Re-Synthesis [Paper](https://arxiv.org/abs/2105.14376)
4. \[CVPRW 2020\] OC-FakeDect: Classifying Deepfakes Using One-class Variational Autoencoder [Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Khalid_OC-FakeDect_Classifying_Deepfakes_Using_One-Class_Variational_Autoencoder_CVPRW_2020_paper.pdf)
###### Manipulation Localization
1. \[CVPR 2024\] **Non-facial, seems proactive**EditGuard: Versatile Image Watermarking for Tamper Localization and Copyright Protection [Paper](https://arxiv.org/pdf/2312.08883)
2. \[WACV 2024\] Weakly-supervised deepfake localization in diffusion-generated images [Paper](https://arxiv.org/pdf/2311.04584)
3. \[arXiv 2024\] Delocate: Detection and Localization for Deepfake Videos with Randomly-Located Tampered Traces [Paper](https://arxiv.org/abs/2401.13516)
4. \[CVPR 2023\] **Non-facial, seems proactive** MaLP: Manipulation Localization Using a Proactive Scheme [Paper](https://arxiv.org/abs/2303.16976)
5. \[CVPR 2023\] Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.pdf)
6. \[ACM MM 2023\] Locate and Verify: A Two-Stream Network for Improved Deepfake Detection [Paper](https://arxiv.org/abs/2309.11131)
7. \[CVPR 2020\] Face X-ray for More General Face Forgery Detection [Paper](https://arxiv.org/abs/1912.13458)
8. \[CVPR 2020\] On the Detection of Digital Face Manipulation [Paper](https://arxiv.org/abs/1910.01717)
###### Identity Discrepancy
1. \[CVPR 2023\] Implicit Identity Driven Deepfake Face Swapping Detection [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Implicit_Identity_Driven_Deepfake_Face_Swapping_Detection_CVPR_2023_paper.pdf)
2. \[CVPR 2022\] Protecting Celebrities from DeepFake with Identity Consistency Transformer [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Protecting_Celebrities_From_DeepFake_With_Identity_Consistency_Transformer_CVPR_2022_paper.pdf)
3. \[TPAMI 2021\] DeepFake Detection Based on Discrepancies Between Faces and Their Context [Paper](https://ieeexplore.ieee.org/document/9468380)
4. \[ICCV 2021\] ID-Reveal: Identity-aware DeepFake Video Detection [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cozzolino_ID-Reveal_Identity-Aware_DeepFake_Video_Detection_ICCV_2021_paper.pdf)
###### Knowledge Distillation
1. \[arXiv 2023\] DomainForensics: Exposing Face Forgery across Domains via Bi-directional Adaptation [Paper](https://arxiv.org/pdf/2312.10680)
2. \[AAAI 2022\] ADD: Frequency Attention and Multi-View based Knowledge Distillation to Detect Low-Quality Compressed Deepfake Images [Paper[(https://arxiv.org/abs/2112.03553)
3. \[ACM MM 2021\] CoReD: Generalizing Fake Media Detection with Continual Representation using Distillation [Paper](https://arxiv.org/abs/2107.02408)
4. \[CVPRW 2021\] FReTAL: Generalizing Deepfake Detection using Knowledge Distillation and Representation Learning [Paper](https://arxiv.org/abs/2105.13617)
5. \[Journal of Mathematical Imaging and Vision 2015\] Sliced and Radon Wasserstein Barycenters of Measures [Paper](https://link.springer.com/article/10.1007/s10851-014-0506-3)
###### Fine-grained Prediction
1. \[ToMM 2024\] **Include facial data, no focused on facial detection**Mastering Deepfake Detection: A Cutting-Edge Approach to Distinguish GAN and Diffusion-Model Images [Paper](https://dl.acm.org/doi/10.1145/3652027)
2. \[CVPR 2023\] Hierarchical Fine-Grained Image Forgery Detection and Localization [Paper](https://arxiv.org/abs/2303.17111)
3. \[ICCV 2023\] Controllable Guide-Space for Generalizable Face Forgery Detection [Paper](https://arxiv.org/abs/2307.14039)
###### Reasoning
1. \[arXiv 2024\] FakeBench: Uncover the Achilles' Heels of Fake Images with Large Multimodal Models [Paper](https://arxiv.org/abs/2404.13306)
2. \[arXiv 2024\] Can ChatGPT Detect DeepFakes? A Study of Using Multimodal Large Language Models for Media Forensics [Paper](https://arxiv.org/abs/2403.14077)
3. \[arXiv 2024\] SHIELD: An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models [Paper](https://arxiv.org/abs/2402.04178)
4. \[arXiv 2024\] Common Sense Reasoning for Deep Fake Detection [Paper](https://arxiv.org/abs/2402.00126)
5. \[ACM ICMRW 2024\] Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection [Paper](https://arxiv.org/abs/2404.18649)
6. \[arXiv 2023\] Towards General Visual-Linguistic Face Forgery Detection [Paper](https://arxiv.org/abs/2307.16545)

##### Diffusion Model Detection
1. \[CVPR 2024\] **Non-facial**LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection [Paper](https://arxiv.org/abs/2403.17465)
2. \[VISAPP 2024\] **Non-facial**Towards the Detection of Diffusion Model Deepfakes [Paper](https://arxiv.org/abs/2210.14571)
3. \[arXiv 2024\] Diffusion Facial Forgery Detection [Paper](https://arxiv.org/abs/2401.15859)
4. \[ICCV 2023\] **Non-facial**DIRE for Diffusion-Generated Image Detection [Paper](https://arxiv.org/pdf/2303.09295)
5. \[ICASSP 2023\] **Non-facial**On the detection of synthetic images generated by diffusion models [Paper](https://arxiv.org/abs/2211.00680)
6. \[ICCVW 2023\] **Non-facial and withdrawn on arxiv**Detecting Images Generated by Deep Diffusion Models using their Local Intrinsic Dimensionality [Paper](https://openaccess.thecvf.com/content/ICCV2023W/DFAD/papers/Lorenz_Detecting_Images_Generated_by_Deep_Diffusion_Models_Using_Their_Local_ICCVW_2023_paper.pdf)
7. \[ICMLW 2023\] Exposing the Fake: Effective Diffusion-Generated Images Detection [Paper](https://arxiv.org/abs/2307.06272)
### Proactive Detection
#### Proactive Methods for GANs
1. \[IJCAI 2024\] Are Watermarks Bugs for Deepfake Detectors? Rethinking Proactive Forensics [Paper](https://arxiv.org/abs/2404.17867)
2. \[TIFS 2024\] Dual Defense: Adversarial, Traceable, and Invisible Robust Watermarking against Face Swapping [Paper](https://arxiv.org/abs/2310.16540)
3. \[CVPR 2023\] **Non-facial, seems proactive** MaLP: Manipulation Localization Using a Proactive Scheme [Paper](https://arxiv.org/abs/2303.16976)
4. \[ACM MM 2023\] SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection [Paper](https://arxiv.org/abs/2305.06321)
5. \[arXiv 2023\] Feature Extraction Matters More: Universal Deepfake Disruption through Attacking Ensemble Feature Extractors [Paper](https://arxiv.org/abs/2303.00200)
6. \[arXiv 2023\] Robust Identity Perceptual Watermark Against Deepfake Face Swapping [Paper](https://arxiv.org/abs/2311.01357)
7. \[CVPR 2022\] **Non-facial**Proactive Image Manipulation Detection [Paper](https://arxiv.org/abs/2203.15880)
8. \[ICLR 2022\] Responsible Disclosure of Generative Models Using Scalable Fingerprinting [Paper](https://arxiv.org/abs/2012.08726)
9. \[ECCV 2022\] TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations [Paper](https://arxiv.org/abs/2112.09151)
10. \[AAAI 2022\] CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfake [Paper](https://arxiv.org/pdf/2105.10872)
11. \[IJCAI 2022\] Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations [Paper](https://arxiv.org/abs/2206.00477)
12. \[AAAI 2021\] Initiative Defense against Facial Manipulation [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16254)
13. \[CVPRW 2020\] Disrupting Deepfakes: Adversarial Attacks Against Conditional Image Translation Networks and Facial Manipulation Systems [Paper](https://arxiv.org/abs/2003.01279)
14. \[WACVW 2020\] Disrupting Image-Translation-Based DeepFake Algorithms with Adversarial Attacks [Paper](https://openaccess.thecvf.com/content_WACVW_2020/papers/w4/Yeh_Disrupting_Image-Translation-Based_DeepFake_Algorithms_with_Adversarial_Attacks_WACVW_2020_paper.pdf)
#### Proactive Methods for Diffusion Models
1. \[ICLR 2024\] **Non-facial**DIAGNOSIS: Detecting Unauthorized Data Usages in Text-to-image Diffusion Models [Paper](https://arxiv.org/abs/2307.03108)
2. \[NeurIPSW 2024\] **Non-facial** DiffusionShield: A Watermark for Data Copyright Protection against Generative Diffusion Models [Paper](https://arxiv.org/pdf/2306.04642)
3. \[ICCV 2023\] **Non-facial** The Stable Signature: Rooting Watermarks in Latent Diffusion Models [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.pdf)
4. \[TPS-ISA 2023\] **Audio generation**Invisible Watermarking for Audio Generation Diffusion Models [Paper](https://arxiv.org/abs/2309.13166)
5. \[arXiv 2023\] **Non-facial**A Recipe for Watermarking Diffusion Models [Paper](https://arxiv.org/abs/2303.10137)
6. \[arXiv 2023\] LEAT: Towards Robust Deepfake Disruption in Real-World Scenarios via Latent Ensemble Attack [Paper](https://arxiv.org/abs/2307.01520)

## Multi-modal Audio-Visual Deepfake Detection
### Audio-Visual Detection
#### Independent Learning
1. \[Applied Soft Computing 2023\] AVFakeNet: A unified end-to-end Dense Swin Transformer deep learning model for audio–visual​ deepfakes detection [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1568494623001424)
2. \[APSIPA ASC 2022\] Multimodal Forgery Detection Using Ensemble Learning [Paper](https://ieeexplore-ieee-org/document/9980255)
3. \[ICCV 2021\] Joint Audio-Visual Deepfake Detection [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Joint_Audio-Visual_Deepfake_Detection_ICCV_2021_paper.pdf)
4. \[ACM MMW 2021\] Evaluation of an Audio-Video Multimodal Deepfake Dataset using Unimodal and Multimodal Detectors [Paper](https://arxiv.org/abs/2109.02993)
#### Joint Learning
##### Intermediate Fusion: Cross-Attention
1. \[\] [Paper]()
2. \[\] [Paper]()
3. \[\] [Paper]()
##### Late Fusion: Concatenation & Addition
1. \[\] [Paper]()
2. \[\] [Paper]()
3. \[\] [Paper]()
4. \[\] [Paper]()
5. \[\] [Paper]()
6. \[\] [Paper]()
7. \[\] [Paper]()
##### Late Fusion: Attention
1. \[\] [Paper]()
2. \[\] [Paper]()
3. \[\] [Paper]()
##### Late Fusion: MLP Mixer Layer
1. \[\] [Paper]()
##### Multi-task Strategy
1. \[\] [Paper]()
2. \[\] [Paper]()
3. \[\] [Paper]()
4. \[\] [Paper]()
5. \[\] [Paper]()
##### Regularization
1. \[\] [Paper]()
2. \[\] [Paper]()
3. \[\] [Paper]()
4. \[\] [Paper]()
#### Matching-based Learning
1. \[\] [Paper]()
2. \[\] [Paper]()
#### Others
1. \[\] [Paper]()
2. \[\] [Paper]()
3. \[\] [Paper]()
4. \[\] [Paper]()

## Multi-modal Visual-Text Deepfake Detection
1. \[\] [Paper]()
2. \[\] [Paper]()
3. \[\] [Paper]()
## Trustworthy Deepfake Detection
### Adversarial Attack
### Backdoor Attack
### Discrepancy Minimization
### Defense Strategies

## Other Useful Sources



