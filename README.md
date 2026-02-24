# 🚀 Multi-Task Targeted Learning  
## A Multi-Task Framework for Lithium-Ion Battery SOH and RUL Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()
[![IEEE TTE](https://img.shields.io/badge/Published-IEEE%20Transactions%20on%20Transportation%20Electrification-005BAC.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 📖 Paper

**A Multi-Task Targeted Learning Framework for Lithium-Ion Battery State-of-Health and Remaining Useful Life**

Chenhan Wang, Zhengyi Bao, Huipin Lin, Jiahao Nie, Chunxiang Zhu  

Published in **IEEE Transactions on Transportation Electrification (TTE), 2025**  
DOI: 10.1109/TTE.2025.3610457  

This repository provides the official implementation of the proposed **Multi-Task Targeted Learning (MTTL)** framework for simultaneous prediction of battery **State-of-Health (SOH)** and **Remaining Useful Life (RUL)** from raw voltage time-series data.

---

## 🔬 Background & Motivation

Accurate estimation of SOH and RUL is essential for:

- Electric vehicle operational safety  
- Battery health diagnostics  
- Predictive maintenance strategies  

Conventional approaches often treat SOH and RUL as independent tasks or adopt a two-step pipeline, which leads to:

1. Error propagation between intermediate predictions  
2. Inconsistent degradation representation  
3. Inefficient feature utilization  

Battery degradation exhibits strong temporal dependency and shared underlying mechanisms between SOH and RUL. Therefore, joint modeling is necessary for consistent and robust prediction.

---

## 🧠 Proposed Method: Multi-Task Targeted Learning (MTTL)

The proposed framework enables end-to-end joint learning of SOH and RUL with task-specific feature extraction.

### 1️⃣ Shared Temporal Representation Learning

Raw voltage time-series data are encoded through a shared backbone network to capture:

- Long-term degradation dynamics  
- Cross-cycle temporal dependencies  
- Intrinsic battery aging patterns  

This shared representation preserves global degradation information.


### 2️⃣ Targeted Task-Specific Learning

To address the distinct characteristics of SOH and RUL, the framework introduces:

- **SOH-specific prediction head** for capacity health estimation  
- **RUL-specific prediction head** for lifetime forecasting  

Targeted feature learning reduces task interference and improves estimation accuracy.


### 3️⃣ End-to-End Joint Optimization

The framework jointly optimizes SOH and RUL objectives through multi-task loss formulation, which:

- Eliminates error accumulation from two-stage methods  
- Improves learning efficiency  
- Enhances generalization across datasets  

The model supports simultaneous SOH estimation and RUL prediction within a unified architecture.

---

## 💻 Environment

- torch==2.1.0+cu118  
- numpy>=1.24.4  
- scipy>=1.10.1  
- scikit-learn>=1.3.2  
- matplotlib>=3.7.5  
- tqdm>=4.66.4
 ```
pip install -r requirement.txt
```

---

## 🏃 Training & Testing
```
python3.8 main.py
```


Hyperparameters such as learning rate, sequence length, and loss weighting coefficients can be modified in the configuration settings within `main.py`.

---

## 📌 Citation

If you find this framework useful for your research, please consider citing:

```bibtex
@article{wang2025multi,
  title={A Multi-Task Targeted Learning Framework for Lithium-Ion Battery State-of-Health and Remaining Useful Life},
  author={Wang, Chenhan and Bao, Zhengyi and Lin, Huipin and Nie, Jiahao and Zhu, Chunxiang},
  journal={IEEE Transactions on Transportation Electrification},
  year={2025},
  publisher={IEEE}
}
