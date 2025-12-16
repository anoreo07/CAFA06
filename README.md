# CAFA 6 Protein Function Prediction - Multi-modal Ensemble System

This repository contains the source code for our solution to the CAFA 6 Protein Function Prediction competition. The system utilizes a multi-modal ensemble strategy combining a **Hybrid CNN-ResNet** architecture and a **Residual MLP + KNN** approach based on ProtT5 embeddings.

## Repository Structure

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
````
## Prerequisites

Ensure you have Python 3.8+ installed along with the following libraries:

```text
pip install torch torchvision torchaudio numpy pandas scikit-learn biopython
````
(Note: If you are using specific GPU versions, please install the appropriate PyTorch version).

## Usage Instructions

> **IMPORTANT:** Before running any script, please open the files and **update the file paths** (e.g., `DATA_DIR`, `embeddings_path`, `save_path`) to match your local directory structure or Kaggle environment.

### Step 1: Run Component Models
You need to generate the prediction scores from the individual models first. These scripts can be run in any order.

#### Option A: Residual MLP + KNN
This module uses ProtT5 embeddings and combines a Deep Residual Network with a Weighted KNN search.

1. Navigate to `model/ResidentalBlocks+KNN/`.
2. Open `ResidentalBlock+KNN.py` (or the `.ipynb` version).
3. **Update the input paths** for training data and embeddings.
4. Run the script:
   ```bash
   python model/ResidentalBlocks+KNN/ResidentalBlock+KNN.py
   ````
   Output: This will save the model weights and/or prediction scores (e.g., res_knn_preds.npy).

#### Option B: Hybrid CNN-ResNet
This module extracts local features using 1D-CNN and global features using SE-ResNet.

1. Navigate to `model/HybridCNNRestNet/`.
2. Open `HybridCNNRestNet.py` (or the `.ipynb` version).
3. **Update the input paths** for training data.
4. Run the script:
   ```bash
   python model/HybridCNNRestNet/HybridCNNRestNet.py
   ````
   Output: This will save the prediction scores (e.g., cnn_resnet_preds.npy).
   
### Step 2: Generate Final Ensemble Submission
Once you have the predictions from both models above, run the final ensemble script to blend the results, apply hierarchical consistency (True Path Rule), and inject external knowledge.

1. Navigate to `model/Ensemble/`.
2. Open `final.py` (or `final.ipynb`).
3. **Update the paths** to point to the output files generated in Step 1.
4. Run the script:
   ```bash
   python model/Ensemble/final.py
   ````
   Output: The script will generate the final submission.tsv file ready for evaluation.

## Methodology Summary

* **Feature Extraction:** ProtT5 Embeddings (1024-dim) + One-hot Taxonomy.
* **Model 1:** Residual MLP combined with Homology-based Weighted KNN.
* **Model 2:** Hybrid Architecture (1D-CNN + SE-ResNet).
* **Ensemble:** Linear weighted blending ($\alpha \cdot M_1 + (1-\alpha) \cdot M_2$).
* **Post-processing:** Hierarchical Consistency enforcement and Knowledge Injection (UniProt-GOA).

## Authors

* **Nguyen Hai An** 
* **Ngo Thi Thao Linh**

## Acknowledgements & Contact

We would like to express our sincere gratitude to the **CAFA 6 Challenge organizers** and the **Kaggle community** for providing the dataset and the evaluation platform. We also acknowledge the authors of **ProtTrans (ProtT5)** and **ESM-2** for their open-source protein language models, which served as the backbone of our feature engineering pipeline.

We actively welcome any feedback, constructive criticism, or suggestions to improve this project. If you have questions about the code or ideas for optimization, please feel free to:

* **Report a bug:** Open a [GitHub Issue](../../issues) in this repository.
* **Contact us:** Reach out via email at `linhandan@gmail.com` or connect with us on Kaggle.

If you find this repository helpful, please consider giving it a ⭐ **Star**!
