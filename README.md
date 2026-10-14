# Drug Classification - End-to-End ML CI/CD Pipeline

[![ML CI/CD](https://github.com/GAGAN1005/Drug-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/GAGAN1005/Drug-Classification/actions/workflows/ci.yml)

A production-grade, end-to-end Machine Learning CI/CD pipeline implementing automated code formatting verification, model training, CML (Continuous Machine Learning) performance reporting in pull requests, and continuous deployment of a client-side inference application to **Hugging Face Static Spaces** using **ONNX Runtime Web**.

---

## 🏛 Architecture Overview

```
                                      +------------------------------------+
                                      |            GitHub Push             |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      |     GitHub Actions Runner (CI)     |
                                      | - Check formatting (Black)         |
                                      | - Train Random Forest Model        |
                                      | - Export ONNX & Joblib Pipelines   |
                                      | - Generate Metrics & Confusion Mat |
                                      +-----------------+------------------+
                                                        |
                         +------------------------------+------------------------------+
                         | (on pull_request)                                           | (on push to main)
                         v                                                             v
       +------------------------------------+                        +------------------------------------+
       |          CML PR Comment            |                        |     Hugging Face Static Space      |
       | - Model accuracy & F1 score        |                        | - app/index.html & app/README.md   |
       | - Inline Confusion Matrix plot     |                        | - model/drug_model.onnx            |
       +------------------------------------+                        | - results/ (metrics & plots)       |
                                                                     +-----------------+------------------+
                                                                                       |
                                                                                       v
                                                                     +------------------------------------+
                                                                     |     Client Browser (Inference)     |
                                                                     | - Runs via onnxruntime-web (Wasm)  |
                                                                     | - Zero backend server compute costs|
                                                                     | - Real-time client-side predictions|
                                                                     +------------------------------------+
```

---

## 📂 Project Structure

```text
Drug-Classification/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI & CD GitHub Actions workflow
├── app/
│   ├── index.html               # Responsive web UI with ONNX Runtime Web inference
│   └── README.md                # Hugging Face Static Space metadata & overview
├── data/
│   └── drug200.csv              # Patient clinical dataset (200 records)
├── model/
│   ├── drug_model.joblib        # Scikit-learn trained pipeline artifact
│   └── drug_model.onnx          # Exported ONNX graph for browser runtime
├── results/
│   ├── confusion_matrix.png     # Evaluation confusion matrix visualization
│   └── metrics.txt              # Classification report, accuracy & F1 scores
├── train.py                     # Training, evaluation & ONNX export script
├── verify_inference.py          # Validation script comparing Sklearn vs ONNX
├── requirements.txt             # Python dependencies
├── Makefile                     # Make commands for local and CI workflows
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

---

## 📊 Dataset & Features

The model classifies patient data into 5 target drug classes (`drugA`, `drugB`, `drugC`, `drugX`, `drugY`).

| Feature | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `Age` | Numerical | Age of patient | 15 - 74 |
| `Sex` | Categorical | Gender | `F`, `M` |
| `BP` | Categorical | Blood Pressure | `HIGH`, `NORMAL`, `LOW` |
| `Cholesterol` | Categorical | Cholesterol Level | `HIGH`, `NORMAL` |
| `Na_to_K` | Numerical | Sodium to Potassium ratio | 6.2 - 38.2 |
| **`Drug`** | Target | Recommended Drug | `drugA`, `drugB`, `drugC`, `drugX`, `drugY` |

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Setup Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
make install
```

### 3. Check Formatting
```bash
make format
```
*(To reformat automatically: `black train.py`)*

### 4. Train Model & Export ONNX
```bash
make train
```

### 5. Inspect Evaluation Metrics
```bash
make eval
```

### 6. Verify Local Inference (Sklearn vs ONNX Runtime)
```bash
python verify_inference.py
```

### 7. Run Web App Locally
Because the browser loads `model/drug_model.onnx` via HTTP `fetch()`, run a local static web server from the repository root:
```bash
python -m http.server 8000
```
Open [http://localhost:8000/app/](http://localhost:8000/app/) in your browser.

---

## ⚙️ CI/CD Setup with GitHub Actions & Hugging Face

### 1. Create a Hugging Face Static Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Space Name: `drug-classification-lite`.
4. License: choose `mit` or `apache-2.0`.
5. SDK: Select **Static** (Static HTML).

### 2. Configure GitHub Secrets
In your GitHub repository:
Navigate to **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name | Value |
| :--- | :--- |
| `HF_TOKEN` | Hugging Face Access Token with **write** permissions ([Generate Token](https://huggingface.co/settings/tokens)) |
| `HF_SPACE_ID` | Your Hugging Face Space ID (`YOUR_USERNAME/drug-classification-lite`) |

*Note: `GITHUB_TOKEN` is automatically provided by GitHub Actions for CML pull request commenting.*

### 3. Push to GitHub
```bash
git add .
git commit -m "Add ML CI pipeline and ONNX Runtime Web app"
git push origin main
```

---

## 🛡 Disclaimer
This application is strictly for educational and demonstration purposes. It is **not** a medical diagnostic or prescription tool.
