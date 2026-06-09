import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads the Customer Personality Analysis dataset.
    """
    try:
        # The dataset is tab-separated (sep='\t') in Kaggle
        df = pd.read_csv(filepath, sep='\t')
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Cleans the raw customer dataset.
    """
    df_clean = df.copy()
    
    # 1. Handle missing values (Income has ~24 missing rows)
    # Drop rows where Income is missing
    df_clean = df_clean.dropna(subset=['Income'])
    
    # 2. Convert Dt_Customer to datetime
    df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    
    # 3. Drop outliers
    # Year_Birth: remove customers born before 1900 (age > 120 is likely erroneous)
    df_clean = df_clean[df_clean['Year_Birth'] > 1900]
    
    # Income: remove extreme outliers (e.g. Income > $600,000)
    df_clean = df_clean[df_clean['Income'] < 600000]
    
    print(f"After cleaning: {df_clean.shape[0]} rows.")
    return df_clean

def engineer_features(df):
    """
    Engineers new features for clustering and modeling.
    """
    df_feat = df.copy()
    
    # 1. Customer Age (Assume reference year is 2021, when dataset was published/analyzed)
    df_feat['Age'] = 2021 - df_feat['Year_Birth']
    
    # 2. Total Spend across all categories
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df_feat['Total_Spend'] = df_feat[spend_cols].sum(axis=1)
    
    # 3. Family Structure / Children
    df_feat['Children'] = df_feat['Kidhome'] + df_feat['Teenhome']
    df_feat['Is_Parent'] = (df_feat['Children'] > 0).astype(int)
    
    # 4. Total Purchases
    purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df_feat['Total_Purchases'] = df_feat[purchase_cols].sum(axis=1)
    
    # 5. Simplify Marital Status
    # Group into 'Partner' vs 'Alone'
    alone_statuses = ['Single', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO']
    partner_statuses = ['Married', 'Together']
    
    def group_marital(status):
        if status in partner_statuses:
            return 'Partner'
        elif status in alone_statuses:
            return 'Alone'
        else:
            return 'Alone' # Default fallback
            
    df_feat['Living_Status'] = df_feat['Marital_Status'].apply(group_marital)
    
    # 6. Simplify Education level
    # Group into Undergraduate, Graduate, Postgraduate
    edu_mapping = {
        'Basic': 'Undergraduate',
        '2n Cycle': 'Undergraduate',
        'Graduation': 'Graduate',
        'Master': 'Postgraduate',
        'PhD': 'Postgraduate'
    }
    df_feat['Education_Level'] = df_feat['Education'].map(edu_mapping)
    
    # 7. Customer tenure (days enrolled as customer)
    # Using the max date in the dataset as the reference point
    max_date = df_feat['Dt_Customer'].max()
    df_feat['Customer_Tenure_Days'] = (max_date - df_feat['Dt_Customer']).dt.days
    
    # Drop redundant or documentation-lacking columns
    cols_to_drop = ['ID', 'Year_Birth', 'Dt_Customer', 'Marital_Status', 'Education', 'Z_CostContact', 'Z_Revenue']
    df_feat = df_feat.drop(columns=[col for col in cols_to_drop if col in df_feat.columns])
    
    print(f"Features engineered. Final dataframe has {df_feat.shape[1]} columns.")
    return df_feat

def preprocess_pipeline(filepath):
    """
    Runs the full loading, cleaning, and feature engineering pipeline.
    """
    df = load_data(filepath)
    if df is not None:
        df_clean = clean_data(df)
        df_final = engineer_features(df_clean)
        return df_final
    return None

if __name__ == "__main__":
    import os
    # For testing when executed directly
    data_path = os.path.join("data", "marketing_campaign.csv")
    if os.path.exists(data_path):
        print("Preprocessing test data...")
        df_processed = preprocess_pipeline(data_path)
        print(df_processed.head())
    else:
        print(f"Test run skipped: '{data_path}' not found. Place the dataset there to run this file directly.")
