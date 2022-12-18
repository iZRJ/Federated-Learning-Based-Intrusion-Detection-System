# Federated-Learning-Based-Intrusion-Detection-System
FL-based intrusion detection system development using model averaging.

# Requirements
- Python 3.6
- NumPy 1.21.5
- Scikit-learn 1.0.2
- Keras 2.3.1

# Dataset
UNSW-NB15 dataset, a real-world traffic dataset for intrusion detection problems.
Publicly available at: https://research.unsw.edu.au/projects/unsw-nb15-dataset

# Usage
1. create directory
```
- main
  - data
  - Server
  - CentralServer
```

2. train FL-NIDS model
```
python FL-Based_NIDS.py
```

# Contact-Info
Email: ruijiezhao@sjtu.edu.cn

# About
Link to our laboratory: [SJTU-NSSL](https://github.com/NSSL-SJTU "SJTU-NSSL")

# Reference
Hope this repo is useful for your research. In addition, we proposed a federated learning method based on knowledge distillation for intrusion detection ([SSFL-IDS](https://github.com/iZRJ/SSFL-IDS)):

R. Zhao, Y. Wang, Z. Xue, T. Ohtsuki, B. Adebisi, and G. Gui, ``Semi-Supervised Federated Learning Based Intrusion Detection Method for Internet of Things,'' IEEE Internet Things J., early access, doi: 10.1109/JIOT.2022.3175918.

```
@ARTICLE{SSFL_IDS,
  author    = {Zhao, Ruijie and Wang, Yijun and Xue, Zhi and Ohtsuki, Tomoaki and Adebisi, Bamidele and Gui, Guan},
  title     = {Semi-Supervised Federated Learning Based Intrusion Detection Method for Internet of Things},
  booktitle = {IEEE Internet of Things Journal},
  pages     = {1--14},
  doi={10.1109/JIOT.2022.3175918}},
  year      = {2022}}
```
