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

## ⚙️ Environment Requirements

- Python >= 3.8  
- PyTorch >= 2.1.0+cu118 
- NumPy >=1.24.4
- SciPy >=1.10.1 
- scikit-learn >=1.3.2
- matplotlib >=3.7.5
- tqdm >=4.66.4

Install requirements
```
pip install -r requirement.txt
```

# Training and Testing
```
python3.8 main.py
```

# citation
If you find ETSformer useful, please consider citing:
```javascript
@article{wang2025multi,
  title={A Multi-Task Targeted Learning Framework for Lithium-Ion Battery State-of-Health and Remaining Useful Life},
  author={Wang, Chenhan and Bao, Zhengyi and Lin, Huipin and Nie, Jiahao and Zhu, Chunxiang},
  journal={IEEE Transactions on Transportation Electrification},
  year={2025},
  publisher={IEEE}
}
```


