# Multi-Task Targeted Learning for Lithium-Ion Battery SOH and RUL Prediction

This repository provides the official implementation of the paper:

> **A Multi-Task Targeted Learning Framework for Lithium-Ion Battery State-of-Health and Remaining Useful Life**  
> Chenhan Wang, **Zhengyi Bao**, Huipin Lin, Jiahao Nie, Chunxiang Zhu  
> *IEEE Transactions on Transportation Electrification*, 2025  
> DOI: 10.1109/TTE.2025.3610457

The proposed framework enables **simultaneous and end-to-end prediction** of lithium-ion battery **State-of-Health (SOH)** and **Remaining Useful Life (RUL)** using raw voltage time-series data.

---

## 🔍 Overview

Accurate estimation of battery SOH and RUL is critical for the safety and reliability of electric vehicles.  
This project proposes a **multi-task targeted learning framework** that jointly models SOH and RUL while addressing:

- Long-term temporal dependency modeling
- Selective feature learning for different degradation indicators
- Error propagation in traditional two-step prediction methods

---

## 📊 Datasets

The experiments in the paper are conducted on five public battery aging datasets:

- **NASA**  
  https://data.nasa.gov/

- **CALCE (University of Maryland)**  
  https://calce.umd.edu/battery-data

- **XJTU**  
  https://github.com/wang-fujin/PINN4SOH

- **Oxford**  
  https://ora.ox.ac.uk/

- **MIT**  
  https://data.matr.io/1/

> ⚠️ **Note:** Due to dataset licenses, this repository does **not** redistribute raw data.  
Please download the datasets from their official sources and follow the preprocessing steps described in the paper.

---

## ⚙️ Environment Requirements

- Python >= 3.8  
- PyTorch >= 1.10  
- NumPy  
- SciPy  
- scikit-learn  
- Hyperopt  

Example installation:

```bash
pip install torch numpy scipy scikit-learn hyperopt


