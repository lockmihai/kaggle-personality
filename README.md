# Kaggle Customer Personality Analysis

This project aims to perform customer segmentation and personality analysis using the Kaggle dataset: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).

## Project Overview

Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors, and concerns of different customer segments.

The primary objective is to group the customer base into distinct segments (clusters) to help target marketing campaigns more effectively.

### Dataset Features

The dataset (`marketing_campaign.csv`) consists of 2,240 records with 29 features:

- **Demographics**: `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer`, `Recency`, `Complain`
- **Product Spend**: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- **Purchasing Channels**: `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`
- **Campaign Acceptance**: `AcceptedCmp1` - `AcceptedCmp5`, `Response` (final campaign)

---

## Directory Structure

```
kaggle-personality/
├── .gitignore
├── requirements.txt
├── README.md
├── download_dataset.sh
├── data/                      # Excluded from Git, place marketing_campaign.csv here
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Preprocessing & Feature Engineering
│   └── clustering.py          # PCA & K-Means Clustering
└── notebooks/
    └── eda_and_clustering.ipynb  # Interactive analysis & visualization
```

---

## Getting Started

### 1. Set Up Environment

Use `uv` or standard Python `venv` to create a virtual environment and install packages:

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the Dataset

You can download the dataset directly from Kaggle:
1. Go to [Kaggle Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).
2. Download the ZIP file, extract `marketing_campaign.csv`, and place it in the `data/` directory.

Alternatively, if you have the Kaggle CLI installed and configured (`~/.kaggle/kaggle.json`), run:
```bash
./download_dataset.sh
```

### 3. Run the Analysis

Start Jupyter Lab to open and run the notebook:
```bash
jupyter lab
```
Or run the Python modules directly:
```bash
python3 src/data_processing.py
python3 src/clustering.py
```
