# Project Title: AI-Generated Arabic Text Detection
Dataset: https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts
# Project Objective

This project aims to build a system capable of distinguishing between human-written Arabic text and AI-generated Arabic text. The goal is to support content authenticity and reliability in Arabic digital platforms.

#  Project Phases
1️⃣ Data Acquisition

2️⃣ Preprocessing & Exploratory Data Analysis (EDA)

3️⃣ Feature Engineering

4️⃣ Modeling

5️⃣ Evaluation & Comparison

6️⃣ Best Model Selection


# Project Structure

```text
AI-Generated-Arabic-Text-Detection/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train_features.csv
│       ├── val_features.csv
│       └── test_features.csv
│
├── notebooks/
│   ├── phase_1_data_acquisition.ipynb
│   ├── phase_2_preprocessing_eda.ipynb
│   ├── phase_3_feature_engineering.ipynb
│   └── phase_4_modeling.ipynb
│
├── models/
│
├── reports/
│   ├── final_model_results.csv
│   ├── best_model_summary.csv
│   └── figures/
│
├── src/
│   ├── data_preparation.py
│   ├── modeling.py
│   ├── utils.py
│   └── visualization.py
│
├── requirements.txt
├── .gitignore
└── README.md
```


# Project Outputs

flowchart LR
    A[Dataset Collection<br/>Human & AI Texts] --> 
    B[Data Preprocessing<br/>Cleaning · Normalization · Tokenization] --> 
    C[Feature Engineering<br/>Linguistic + Statistical]

    C --> D[Traditional ML Models<br/>Logistic Regression<br/>Random Forest<br/>XGBoost]
    C --> E[Deep Learning Model<br/>AraBERT Fine-Tuning]

    D --> F[Model Evaluation<br/>Accuracy · Precision · Recall · F1 · ROC-AUC]
    E --> F

    F --> G[Confusion Matrix<br/>& Error Analysis]
    G --> H[Best Model Selection<br/>AraBERT]


- Evaluation reports and visualizations

- A structured and reproducible project workflow
  

  # Report#
