# TS-Foundation
A general large language model capable of hybrid forecasting.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

## Overview

TS-Foundation is a unified large-model framework for cross-domain time-series forecasting.  
Instead of training a separate model for each dataset or domain, TS-Foundation formulates heterogeneous forecasting problems under a shared instruction-driven interface and optimizes a single forecasting backbone across multiple temporal regimes.

The framework is designed to handle heterogeneous time-series data from different domains, such as energy, finance, and weather, within one unified modeling pipeline. It further supports target-domain refinement through preference alignment, enabling the shared backbone to retain broad cross-domain generalization while achieving domain-specific improvement when necessary.

---

## Key Features

- **Unified cross-domain forecasting interface**  
  Heterogeneous time-series tasks are converted into a shared instruction-based representation, allowing a single model to process multiple forecasting scenarios.

- **Mixed-domain supervised training**  
  A unified forecasting backbone is first trained on mixed-domain data to learn transferable temporal patterns across domains.

- **Task-sensitive fast adaptation**  
  A lightweight task-adaptive mechanism is introduced to improve adaptation efficiency under low-resource or newly introduced target domains.

- **Temporal-aware dynamic regularization**  
  The training pipeline incorporates time-aware dynamic regularization to mitigate overfitting and improve robustness under heterogeneous temporal regimes.

- **Target-domain DPO alignment for regression**  
  After mixed-domain supervised fine-tuning, the model can be further refined on a selected target domain via preference-based alignment, improving domain-specific forecasting quality without rebuilding the backbone.

- **PEFT- and quantization-friendly design**  
  The framework is compatible with efficient fine-tuning strategies such as LoRA/PEFT and can be deployed under resource-constrained settings.

---

## Motivation

Most existing time-series forecasting methods are developed in a domain-specific manner.  
They often require separate architectures, hyperparameter tuning strategies, and retraining procedures for different datasets. Such a fragmented paradigm limits scalability, increases deployment complexity, and weakens transferability across domains.

TS-Foundation addresses this problem by introducing a unified large-model paradigm for forecasting. The central idea is simple:

1. **learn a shared forecasting backbone from mixed-domain supervision**, and then  
2. **refine the model toward a target domain when deeper specialization is needed**.

This design allows the model to first obtain **breadth** through cross-domain learning and then gain **depth** through target-domain preference alignment.

---

## Framework

The full pipeline consists of two stages:

### Stage 1: Mixed-domain supervised forecasting

In the first stage, heterogeneous time-series datasets are serialized into a shared instruction-driven format.  
The model is trained under a common supervised objective across mixed domains, enabling it to learn reusable temporal representations and general forecasting behavior.

### Stage 2: Target-domain preference alignment

In the second stage, the pretrained shared backbone is further refined on a selected target domain using preference pairs constructed from chosen and rejected forecasting trajectories.  
This stage improves target-domain regression quality while preserving the unified modeling framework.

---

## Method Highlights

### 1. Unified instruction-based forecasting interface
All datasets are reformulated into a common input-output structure.  
This eliminates the need for domain-specific heads or dataset-specific model redesign.

### 2. Shared forecasting backbone
A single backbone is used to process diverse temporal signals.  
This supports unified training, simplified deployment, and transferable representation learning.

### 3. Task-sensitive fast adaptation
To improve adaptation under domain shift, limited samples, or emerging tasks, we introduce a lightweight task-sensitive adaptation mechanism that can be efficiently combined with PEFT-style optimization.

### 4. Temporal-aware dynamic regularization
A time-aware regularization strategy is adopted during training to stabilize optimization across heterogeneous temporal dynamics and improve robustness under noisy or complex regimes.

### 5. Target-domain DPO alignment
To further improve domain-specific performance, we formulate regression refinement as a preference alignment problem.  
This allows the shared model to better capture target-domain forecasting preferences after mixed-domain training.

---

## Why This Repository

This repository is intended to provide:

- a **unified benchmark-oriented implementation** for cross-domain forecasting;
- an extensible **large-model training pipeline** for mixed-domain supervised learning;
- an efficient **target-domain refinement strategy** based on preference alignment;
- a practical framework for studying **generalization, robustness, and specialization** in time-series foundation models.

---

## Experimental Scope

The model is designed to support the following evaluation settings:

- **overall performance on cross-domain mixed forecasting**
- **generalization across heterogeneous domains**
- **ablation study on task-sensitive adaptation and temporal-aware regularization**
- **domain-specific improvement via DPO alignment**
- **robustness under noisy, missing, or limited-data settings**
- **qualitative visualization of representative forecasting cases**

---

## Repository Structure

```bash
TS-Foundation/
├── data/                   # datasets or dataset preprocessing scripts
├── configs/                # training and evaluation configurations
├── src/
│   ├── models/             # backbone and model modules
│   ├── data/               # dataset loading and instruction construction
│   ├── training/           # SFT / DPO / adaptation modules
│   ├── evaluation/         # metrics and test scripts
│   └── utils/              # utility functions
├── scripts/                # runnable shell scripts
├── checkpoints/            # saved checkpoints (optional)
├── figures/                # paper figures / visualizations
├── requirements.txt
└── README.md
````

---

## Installation

```bash
git clone https://github.com/yourname/TS-Foundation.git
cd TS-Foundation
pip install -r requirements.txt
```

---

## Quick Start

### 1. Mixed-domain supervised training

```bash
python train_sft.py \
    --config configs/sft.yaml
```

### 2. Target-domain DPO refinement

```bash
python train_dpo.py \
    --config configs/dpo.yaml
```

### 3. Evaluation

```bash
python evaluate.py \
    --config configs/eval.yaml
```

---

## Planned Support

* [ ] mixed-domain training scripts
* [ ] target-domain DPO refinement scripts
* [ ] PEFT / LoRA support
* [ ] quantized training support
* [ ] visualization tools for forecasting curves
* [ ] robustness evaluation under noise and missing values
* [ ] benchmark configuration files

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{yourname2026tsfoundation,
  title={TS-Foundation: A Unified Large Language Model Framework for Cross-Domain Time-Series Forecasting},
  author={Your Name and Coauthors},
  journal={Under Review},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions, suggestions, or collaboration, please open an issue or contact:

* **Name**: FangXin
* **Email**: fangx6531@gmail.com
```
