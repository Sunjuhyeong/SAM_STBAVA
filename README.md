# Extend Segment Anything Model into Auditory and Temporal Dimensions

This repository provides the PyTorch implementation for the ICIP2024 paper "Extend Segment, Anything Model, into Auditory and Temporal Dimensions.” This paper proposes the spatio-temporal, bidirectional audio-visual attention (ST-BAVA) module to enable SAM to utilize the audio-visual relationship across multiple frames. [Project Page] [arXiv]

This repository is based on the [repo](https://github.com/OpenNLPLab/AVSBench) of the ECCV 2022 paper “Audio-visual segmentation” and the [repo](https://github.com/jinxiang-liu/anno-free-AVS) of the WACV 2024 paper “Annotation-free audio-visual segmentation.”

### Data preparation

### 1. AVSBench dataset

The AVSBench dataset was first proposed in the [ECCV paper](https://arxiv.org/abs/2207.05042). It contains a Single-source and a Multi-Source subset. The ground truths of these two subsets are binary segmentation maps indicating pixels of the sounding objects. These downloaded data should be placed in the directory `avsbench_data`.

### 2. pre trained backbones

The pretrained VGGish (audio) backbones can be downloaded from [here](https://drive.google.com/drive/folders/1386rcFHJ1QEQQMF6bV1rXJTzy8v26RTV?usp=sharing) (from original AVS repo). Place it to the directory `pretrained_backbones`.

The pretrained SAMA encoder can be downloaded from [here](https://drive.google.com/drive/folders/1386rcFHJ1QEQQMF6bV1rXJTzy8v26RTV?usp=sharing) (from the original SAMA-AVS repo). Place it in the directory `avs_scripts/avs_ms3/sam_sandbox` and `avs_scripts/avs_s4/sam_sandbox`. 

**Notice:** Please update the data path and pre-trained backbone in `avs_s4/config.py`, `avs_ms3/config.py`, and `avss/config.py` accordingly.

---

### S4 setting

- Train AVS Model

```
cd avs_scripts/avs_s4
bash train.sh
```

- Test AVS Model

```
cd avs_scripts/avs_s4
bash test.sh
```

---

### MS3 setting

- Train AVS Model

```
cd avs_scripts/avs_ms3
bash train.sh
```

- Test AVS Model

```
cd avs_scripts/avs_ms3
bash test.sh
```

---

### License

This project is released under the Apache 2.0 license as found in the [LICENSE](https://www.notion.so/LICENSE) file.