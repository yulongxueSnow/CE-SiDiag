# CE-SiDiag: Bearing Fault Diagnosis Based on Chaos-Enhanced Ensemble Learning and Similarity-Based Diagnosis

## 1. Project Overview

This repository provides the official implementation of **CE-SiDiag (Chaos-Ensemble Similarity Diagnosis)** for the bearing fault diagnosis experiments reported in the corresponding paper. The method is designed for bearing fault diagnosis in rotating machinery.

The core idea of CE-SiDiag is to coordinate **historical-sample similarity assessment** with **chaos-enhanced heterogeneous ensemble classification**, providing an adaptive diagnostic process for different input samples.

The main procedure consists of the following steps:

1. **Time-Series Feature Extraction**

   The raw bearing vibration signals are segmented using a sliding window, and multidimensional time-domain and frequency-domain statistical features are extracted using `TSFEL`.

2. **Redundant Feature Removal and PCA-Based Dimensionality Reduction**

   Highly redundant features are first removed according to the Pearson correlation coefficient. The remaining features are standardized using `StandardScaler` and subsequently projected into a **13-dimensional principal-component space** using Principal Component Analysis (PCA).

3. **Chaos-Enhanced Feature Transformation and Heterogeneous Ensemble Classification**

   A Sine chaotic map is iteratively applied to the reduced feature representation to introduce nonlinear dynamical perturbations and enhance the nonlinear characterization of fault classes. The transformed features are then processed by a heterogeneous ensemble consisting of **chaos-enhanced Naive Bayes, Support Vector Machine (SVM), and k-Nearest Neighbors (KNN)**. The individual classification results are integrated through **majority voting**.

4. **Similarity-Based Diagnosis and Decision Correction**

   During diagnosis, cosine similarity is used to measure the correspondence between an incoming sample and historical diagnostic samples. The most similar samples in the training set are retrieved, and when a sufficiently high similarity is established, the historical-sample information is used to correct the diagnostic result, thereby improving the consistency between the final decision and historical diagnostic evidence.

The diagnostic task contains four bearing states:

| Label | Bearing State              |
| ----- | -------------------------- |
| 0     | Normal (NS)                |
| 1     | Inner-Race Fault (IB)      |
| 2     | Outer-Race Fault (OB)      |
| 3     | Rolling-Element Fault (TB) |

---

## 2. Dataset

The experiments use the **Jiangnan University (JNU) Bearing Dataset**, with vibration signals collected under the 600 rpm operating condition from data channel 2.

| File           | Bearing State              |
| -------------- | -------------------------- |
| `ib600_2.csv`  | Inner-Race Fault (IB)      |
| `ob600_2.csv`  | Outer-Race Fault (OB)      |
| `tb600_2.csv`  | Rolling-Element Fault (TB) |
| `n600_3_2.csv` | Normal State (NS)          |

### Signal Segmentation

A sliding-window strategy is used for signal segmentation:

* **Window length:** 2000 samples
* **Step size:** 600 samples
* **Samples per class:** 832
* **Number of classes:** 4
* **Total feature samples:** 3328

---

## 3. Repository Structure

```text
.
├── pre_process_data.py        # Raw vibration signals → feature extraction
├── decomposition_features.py  # Correlation-based feature removal + PCA
├── chaos_classfier.py         # Chaos-enhanced ensemble classification
├── similar_vector.py          # Cosine similarity retrieval and diagnosis
├── main_merge_classfier.py    # Main experimental pipeline
├── test_results.py            # Evaluation and visualization
├── tools/                     # Auxiliary visualization tools
├── data/                      # Raw data and intermediate features
├── requirements.txt           # Python dependencies
├── Makefile                   # Automated execution scripts
└── LICENSE                    # License
```

---

## 4. Environment Requirements

* **Python:** ≥ 3.9
* **Tested environment:** Python 3.11

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Alternatively, use the provided Makefile:

```bash
make install
```

The visualization modules use the `Agg` backend by default, allowing the complete pipeline to run in headless or offline environments without a graphical interface.

---

## 5. Reproduction

The repository provides the intermediate feature files and trained model files required for reproducing the reported experiments, including:

* `*_features.csv`
* `*_features_pca13.csv`
* scaler/PCA model files (`*.pkl`)
* trained model files (`*_model.pkl`)

Therefore, the experimental results can be reproduced directly without re-extracting all features. Alternatively, the complete pipeline can be executed from the raw vibration signals.

### 5.1 One-Command Execution

Run the complete preprocessing, feature extraction, dimensionality reduction, and training pipeline:

```bash
make all
```

The overall workflow is:

```text
Raw vibration signals
        ↓
Feature extraction
        ↓
Redundant feature removal
        ↓
Standardization + PCA
        ↓
Chaos-enhanced feature transformation
        ↓
Heterogeneous ensemble classification
        ↓
Similarity-based diagnosis
        ↓
Performance evaluation
```

### 5.2 Step-by-Step Execution

#### Step 1: Raw Signals → Feature Extraction

```bash
python pre_process_data.py
```

This step segments the raw vibration signals using the specified sliding window and extracts multidimensional vibration features using TSFEL.

If the corresponding feature files are already available in the repository, this step can be skipped.

#### Step 2: Redundant Feature Removal + PCA

```bash
python decomposition_features.py
```

This step removes highly correlated redundant features, standardizes the remaining features, and performs PCA dimensionality reduction.

The resulting feature representation contains **13 principal components**, together with the corresponding preprocessing models.

#### Step 3: CE-SiDiag Main Experiment

```bash
python main_merge_classfier.py
```

This script executes the complete CE-SiDiag diagnostic pipeline, including:

* chaos-enhanced feature transformation;
* Naive Bayes, SVM, and KNN classification;
* majority-voting ensemble decision;
* cosine-similarity-based historical-sample retrieval;
* similarity-based diagnostic correction;
* final performance evaluation.

### Reproducibility Note

The `train_test_split` procedure does not use a fixed random seed. Therefore, the results may exhibit small variations between independent runs. A variation of approximately **±0.5%** in the reported accuracy is considered normal.

---

## 6. Key Hyperparameters

| Parameter                              |                    Setting | Implementation              |
| -------------------------------------- | -------------------------: | --------------------------- |
| Sliding-window length                  |               2000 samples | `pre_process_data.py`       |
| Sliding-window step                    |                600 samples | `pre_process_data.py`       |
| Pearson correlation threshold          |                        0.8 | `decomposition_features.py` |
| PCA dimensions                         |                         13 | `n_components`              |
| Feature scaling                        |             StandardScaler | `decomposition_features.py` |
| Chaotic mapping                        |                   Sine map | `chaos_maps`                |
| Chaos control parameter                |                 `r = 0.99` | `chaos_maps`                |
| Chaos iterations                       |                         10 | `chaos_iterations`          |
| Random perturbation standard deviation |                       0.01 | `chaos_maps`                |
| Similarity metric                      |          Cosine similarity | `main_merge_classfier.py`   |
| Similarity threshold                   |                       0.90 | `main_merge_classfier.py`   |
| Similarity retrieval range             |                    Top-100 | `similar_vector.py`         |
| Naive Bayes                            | Chaos-enhanced Gaussian NB | `chaos_classfier.py`        |
| SVM kernel                             |                        RBF | `chaos_classfier.py`        |
| SVM parameter                          |                  `C = 1.0` | `chaos_classfier.py`        |
| SVM gamma                              |                    `scale` | `chaos_classfier.py`        |
| KNN neighbors                          |                    `k = 5` | `chaos_classfier.py`        |
| Ensemble strategy                      |            Majority voting | `chaos_classfier.py`        |

---

## 7. Model Configuration

The CE-SiDiag implementation uses a 13-dimensional feature representation after feature preprocessing and PCA dimensionality reduction.

The chaos-enhanced Naive Bayes classifier uses the transformed feature representation for four-class classification. The SVM classifier uses an RBF kernel, while KNN performs distance-based classification in the same feature space. Their individual predictions are integrated using majority voting.

For similarity-based diagnosis, cosine similarity is calculated between the incoming sample and historical training samples. The Top-100 most similar samples are retrieved, and a similarity threshold of **0.90** is used for similarity-based decision processing.

The main implementation configuration is summarized below:

| Component               | Configuration        |
| ----------------------- | -------------------- |
| Input feature dimension | 13                   |
| Number of classes       | 4                    |
| Chaos mapping           | Sine map             |
| Chaos control parameter | 0.99                 |
| Chaos iterations        | 10                   |
| Naive Bayes             | Gaussian Naive Bayes |
| SVM                     | RBF kernel           |
| KNN                     | `k = 5`              |
| Ensemble decision       | Majority voting      |
| Similarity metric       | Cosine similarity    |
| Retrieval range         | Top-100              |
| Similarity threshold    | 0.90                 |

---

## 8. Experimental Evaluation

The repository provides scripts for evaluating the diagnostic performance of CE-SiDiag, including:

* Accuracy
* F1-score
* Confusion matrix
* Training time
* Inference time

The implementation can therefore be used to reproduce the main diagnostic performance and computational-efficiency experiments reported in the paper.

---

## 9. Citation

If you use this repository or the CE-SiDiag method in your research, please cite the corresponding paper using the BibTeX information provided in the manuscript.

```bibtex
@article{
    ...
}
```

---

## 10. License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Contact

For questions regarding the implementation or experimental reproduction, please refer to the contact information provided in the corresponding paper.
