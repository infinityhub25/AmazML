# AmazML — Business Entity Resolution

A machine learning project for identifying and matching records that refer to the same real-world business across multiple data sources.

# Project Overview

Business information is often stored across different databases, where the same business may appear under different names, address formats, or other variations. These inconsistencies make it difficult to identify duplicate records and build a unified view of business entities.
AmazML addresses this problem using intelligent candidate generation, text normalization, fuzzy similarity matching, and machine learning to identify potential matches between business records.

# Objectives

- Identify potential duplicate business records across multiple data sources.
- Reduce unnecessary comparisons through candidate generation and blocking.
- Extract useful similarity features from business names and addresses.
- Use a trained LightGBM model to support business matching.
- Build a modular and reusable entity-resolution pipeline.

# Methodology

The project follows a multi-stage workflow:

1. Data ingestion: Load business records from the configured data sources.
2. Normalization: Standardize text fields to improve consistency between records.
3. Candidate generation and blocking: Narrow down the records that need to be compared.
4. Feature engineering: Calculate similarity features, including fuzzy name and address similarities.
5. Model training: Train a LightGBM classifier to distinguish likely matches from non-matches.
6. Prediction and evaluation: Use the trained model to score candidate pairs and evaluate matching performance using the implemented evaluation utilities.

# Tech Stack

- Programming language: Python
- Machine learning: LightGBM
- Text similarity: RapidFuzz
- Data processing: Libraries specified in `requirements.txt`
- Version control: Git and GitHub

# Project Structure

```text
AmazML/
├── models/
│   ├── lgbm_matcher.txt
│   ├── feature_names.json
│   └── threshold.json
├── src/
│   ├── __init__.py
│   ├── blocking.py
│   ├── config.py
│   ├── features.py
│   ├── io_utils.py
│   ├── metrics.py
│   ├── normalize.py
│   ├── predict.py
│   └── train.py
├── run_step1_blocking.py
├── requirements.txt
└── .gitignore
```

# Key Components

- `blocking.py` — Candidate generation and blocking logic.
- `normalize.py` — Text normalization utilities.
- `features.py` — Similarity feature generation.
- `train.py` — Model training functionality.
- `predict.py` — Prediction functionality.
- `metrics.py` — Evaluation utilities.
- `io_utils.py` — Input/output utilities.
- `config.py` — Project configuration.
- `models/` — Trained model and supporting metadata.
- `run_step1_blocking.py` — Blocking pipeline entry-point script.

# Dataset

The project uses business records from three data sources for entity resolution, with separate training and test datasets.

Training Data
The training dataset contains:
- `train_source1` — Business records from source 1.
- `train_source2` — Business records from source 2.
- `train_source3` — Business records from source 3.
- `train_ground_truth` — Ground-truth matching information used for supervised training and/or evaluation.

Test Data
The test dataset contains:
- `test_source1` — Test records from source 1.
- `test_source2` — Test records from source 2.
- `test_source3` — Test records from source 3.

# Data Availability

The original dataset files are stored locally and are not included in this GitHub repository because of their large size and potential data redistribution restrictions.

To reproduce the pipeline, obtain the data from its original authorized source and configure the input paths according to the project code.

# Trained Model

The repository includes the following model artifacts:

- `lgbm_matcher.txt` — Trained LightGBM model.
- `feature_names.json` — Feature metadata.
- `threshold.json` — Threshold configuration.

These files are provided to support the project's prediction workflow. Successful inference requires compatible input data, preprocessing, feature ordering, and configuration.

# Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/infinityhub25/AmazML.git
cd AmazML
```

# 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

# 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

# 4. Configure the input data

Obtain the required datasets from an authorized source and configure their paths according to `src/config.py` and the pipeline implementation.

# 5. Run the pipeline

Review `run_step1_blocking.py` and the project configuration before executing the pipeline. Ensure the required input files are available and that the configured paths are correct.

```bash
python run_step1_blocking.py
```

# Future Improvements

- Add a small, permission-cleared sample dataset for demonstration.
- Document the exact input schema and prediction commands.
- Add verified evaluation metrics and example predictions.
- Improve reproducibility and automated testing.
- Add a visual representation of the complete matching pipeline.

# Author

**Project:** AmazML — Business Entity Resolution  
**GitHub:** [infinityhub25](https://github.com/infinityhub25)

---

*This project demonstrates an approach to large-scale entity matching using data processing, fuzzy similarity features, candidate generation, and machine learning.*
