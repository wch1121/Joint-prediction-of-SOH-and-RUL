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

### Key features

- **Multi-scale Feature Extraction Module (FEM)**  
  Captures local degradation patterns from raw voltage sequences using multi-scale 1D CNNs.

- **Improved Extended LSTM (IE-LSTM)**  
  Enhances long-term memory with exponential gating and stabilized internal states.

- **Dual-Stream Attention Module (DSAM)**  
  - Polarized Attention for SOH prediction  
  - Sparse Attention for RUL prediction  

- **End-to-end Multi-task Learning**  
  Direct many-to-two mapping from voltage sequences to SOH and RUL.

---

## 🧠 Network Architecture

The overall framework consists of four main components:

