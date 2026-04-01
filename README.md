# ML-FINAL-PROJECT-TEAM-18
# GCN-Transformer Hybrid for Multi-Disease Diagnosis from Simulated EHR Data

A deep learning system that combines **Graph Convolutional Networks (GCN)** and a **Time-Aware Transformer** to classify 141 diseases from simulated Electronic Health Record (EHR) data derived from binary symptom profiles.

---

## Project Overview

Traditional disease classifiers treat patients as static feature vectors, ignoring two important signals:
- **Symptom co-occurrence structure** — fever and chills together are more diagnostic than either alone
- **Temporal visit patterns** — symptoms revealed across multiple doctor visits over time

This project addresses both by:
1. Building a **symptom co-occurrence graph** and learning node embeddings via GCNConv layers
2. Simulating **multi-visit EHR sequences** with timestamps per patient
3. Encoding visit sequences through a **Time-Aware Transformer** with attention pooling
4. Classifying **141 disease classes** with 77.43% test accuracy and Macro F1 of 0.773

### Model Architecture
```
Binary Symptom Matrix
        ↓
GCN (2x GCNConv layers, EMB_DIM=64)
        ↓
Symptom Node Embeddings
        ↓
Per-Visit Mean Pooling  +  Time Embedding
        ↓
Transformer Encoder (2 layers, 4 heads)
        ↓
Attention Pooling → Context Vector (64,)
        ↓
Dropout → Linear → Softmax
        ↓
Disease Prediction (141 classes)
```

### Results

| Metric | Value |
|---|---|
| Test Accuracy | 77.43% |
| Macro Precision | 0.7895 |
| Macro Recall | 0.7813 |
| Macro F1 | 0.7734 |
| Disease Classes | 141 |
| Symptom Features | 405 |

---

## Repository Structure
```
├── ML FINAL PROJECT.ipynb                  # Main notebook — full pipeline
├── README.md                    # This file
├── data/
│   └── fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv
├── outputs/
│   ├── text_accuracy_over_epochs.png
│   ├── fig2_symptom_cooccurrence_subgraph.png
│   ├── fig3_ehr_simulation_diagram.png
│   └── confusion_matrix.png
|   |_  per_class_f1score
```

---

## Setup Instructions

### 1. Clone or download the repository
```bash
git clone https://github.com/yourusername/gcn-transformer-disease.git
cd gcn-transformer-disease
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio
pip install torch_geometric
pip install scikit-learn pandas numpy matplotlib seaborn networkx
```

> **Note:** If `torch_geometric` fails, just run:
> ```bash
> pip install torch_geometric
> ```
> The optional dependencies (`pyg_lib`, `torch_scatter` etc.) are not required for this project.

### 4. Place the dataset

Put the CSV file in the project root or update the path in the notebook:
```python
df = pd.read_csv('fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv')
```

---

## How to Run

### Option A — Jupyter Notebook (recommended)
```bash
jupyter notebook model.ipynb
```

Run all cells top to bottom. The notebook is divided into these sections:

| Section | Description |
|---|---|
| Cell 1 | Imports |
| Cell 2 | Data loading, symptom parsing, oversampling |
| Cell 3 | Graph construction (symptom co-occurrence) |
| Cell 4 | EHR simulation |
| Cell 5 | Model definition (GCN + Transformer) |
| Cell 6 | Training loop (40 epochs, Stage 1) |
| Cell 7 | Resume training (20 epochs, Stage 2) |
| Cell 8 | Resume training (40 epochs, Stage 3) |
| Cell 9 | Evaluation + classification report |
| Cell 10 | Result graphs |

### Option B — Google Colab

1. Upload the notebook and CSV to Colab
2. Run this install cell first:
```python
!pip install torch_geometric -q
```

3. Run all remaining cells

### Option C — Kaggle

The environment already has most dependencies. Just run:
```python
!pip install torch_geometric -q
```

Then run all cells.

---

## Key Hyperparameters

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 64 |
| GCN layers | 2 |
| Transformer heads | 4 |
| Transformer layers | 2 |
| Batch size | 128 |
| Total epochs | 100 (staged) |
| Learning rate (Stage 1) | 5×10⁻³ |
| Learning rate (Stage 2) | 5×10⁻⁴ |
| Learning rate (Stage 3) | 1×10⁻⁴ |
| Dropout | 0.3 |
| Label smoothing | 0.1 |
| Max EHR visits per patient | 3 |

---

## Requirements
```
torch>=2.0.0
torch_geometric>=2.3.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
networkx>=3.0
```

Save this as `requirements.txt` and install with:
```bash
pip install -r requirements.txt
```

---

## Notes

- The EHR visit sequences are **simulated** from static symptom data — not real patient records
- Accuracy varies slightly between runs due to random visit simulation — set `SEED = 42` for reproducibility
- Training is staged across 3 runs to avoid restarting from scratch when extending epochs
- Best model checkpoint is saved automatically during training as `best_state`

---

