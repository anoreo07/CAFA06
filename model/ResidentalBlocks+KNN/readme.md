# CAFA 6 Protein Function Prediction - Multi-modal Ensemble System

This repository contains the source code for our solution to the CAFA 6 Protein Function Prediction competition. The system utilizes a multi-modal ensemble strategy combining a **Hybrid CNN-ResNet** architecture and a **Residual MLP + KNN** approach based on ProtT5 embeddings.

## 📂 Repository Structure

```text
CAFA06/
├── model/
│   ├── Ensemble/
│   │   ├── ensemble.ipynb       # Experimentation notebook
│   │   ├── ensemble.py          # Script for blending logic
│   │   ├── final.ipynb          # Final submission generation (Notebook)
│   │   └── final.py             # Final submission generation (Script)
│   │
│   ├── HybridCNNRestNet/
│   │   ├── HybridCNNRestNet.ipynb
│   │   └── HybridCNNRestNet.py  # 1D-CNN and SE-ResNet Model
│   │
│   └── ResidentalBlocks+KNN/
│       ├── ResidentalBlocks+KNN.ipynb
│       └── ResidentalBlock+KNN.py # Residual MLP and Weighted KNN Model
│
└── README.md
## 🛠️ Prerequisites

```text

Ensure you have Python 3.8+ installed along with the following libraries:

Bash

pip install torch torchvision torchaudio numpy pandas scikit-learn biopython
(Note: If you are using specific GPU versions, please install the appropriate PyTorch version).
