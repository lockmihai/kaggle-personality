import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_classif, mutual_info_classif
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import preprocess_pipeline

def prepare_classification_data(filepath, random_state=42):
    """
    Loads raw data, applies basic cleaning and feature engineering,
    and splits into Train (70%), Test (20%), and Application (10%) sets.
    Returns:
        df_train, df_test, df_app (raw feature dataframes)
        y_train, y_test, y_app (target arrays)
    """
    # 1. Load, clean, and engineer features using the core pipeline
    df = preprocess_pipeline(filepath)
    if df is None:
        raise ValueError(f"Could not load data from {filepath}")
    
    # 2. Separate target 'Response'
    if 'Response' not in df.columns:
        raise ValueError("Target column 'Response' not found in dataset.")
    
    y = df['Response'].values
    X = df.drop(columns=['Response'])
    
    # 3. Create the Application Set (10%) and the remaining data (90%)
    # Stratify by y to preserve target distribution
    X_temp, X_app, y_temp, y_app = train_test_split(
        X, y, test_size=0.10, random_state=random_state, stratify=y
    )
    
    # 4. Split the remaining 90% into Train (70% of total) and Test (20% of total)
    # 20/90 ~ 0.2222 test size of the temporary split
    test_size_temp = 0.20 / 0.90
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size_temp, random_state=random_state, stratify=y_temp
    )
    
    print(f"Data split sizes:")
    print(f"  Train:       {X_train.shape[0]} rows")
    print(f"  Test:        {X_test.shape[0]} rows")
    print(f"  Application: {X_app.shape[0]} rows")
    
    return X_train, X_test, X_app, y_train, y_test, y_app

def evaluate_and_rank_predictors(X_train, y_train):
    """
    Evaluates predictors using statistical methods:
    - ANOVA F-value (for numerical features)
    - Mutual Information (for all features)
    Returns a ranking DataFrame.
    """
    # Identify numerical and categorical columns
    cat_cols = ['Living_Status', 'Education_Level']
    num_cols = [col for col in X_train.columns if col not in cat_cols]
    
    # For ANOVA, we need scaled/filled numerical values
    X_num = X_train[num_cols].fillna(X_train[num_cols].mean())
    f_vals, p_vals = f_classif(X_num, y_train)
    
    # Create temp dataframe with encoded variables to calculate mutual info
    # Simple label encoding for categorical features just for MI computation
    X_mi = X_train.copy()
    for col in cat_cols:
        X_mi[col] = X_mi[col].astype('category').cat.codes
    X_mi = X_mi.fillna(X_mi.mean())
    
    mi_scores = mutual_info_classif(X_mi, y_train, random_state=42)
    
    # Build ranking table
    rankings = []
    for idx, col in enumerate(X_train.columns):
        mi = mi_scores[idx]
        if col in num_cols:
            num_idx = num_cols.index(col)
            f_val = f_vals[num_idx]
            p_val = p_vals[num_idx]
            rankings.append({
                'Feature': col,
                'Type': 'Numerical',
                'ANOVA_F_Score': f_val,
                'ANOVA_p_value': p_val,
                'Mutual_Info': mi
            })
        else:
            rankings.append({
                'Feature': col,
                'Type': 'Categorical',
                'ANOVA_F_Score': np.nan,
                'ANOVA_p_value': np.nan,
                'Mutual_Info': mi
            })
            
    df_rankings = pd.DataFrame(rankings)
    df_rankings = df_rankings.sort_values(by='Mutual_Info', ascending=False).reset_index(drop=True)
    return df_rankings

def build_preprocessing_pipeline(X_train):
    """
    Creates and fits a column transformer on the training features.
    """
    cat_cols = ['Living_Status', 'Education_Level']
    num_cols = [col for col in X_train.columns if col not in cat_cols]
    
    # Numeric pipeline scales features
    # Categorical pipeline one-hot encodes features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    preprocessor.fit(X_train)
    return preprocessor

def get_feature_names(preprocessor, X_train):
    """
    Extracts feature names after ColumnTransformer preprocessing.
    """
    cat_cols = ['Living_Status', 'Education_Level']
    num_cols = [col for col in X_train.columns if col not in cat_cols]
    
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
    
    return num_cols + cat_features

if __name__ == "__main__":
    # Test script execution
    filepath = os.path.join("data", "marketing_campaign.csv")
    if os.path.exists(filepath):
        X_train, X_test, X_app, y_train, y_test, y_app = prepare_classification_data(filepath)
        rankings = evaluate_and_rank_predictors(X_train, y_train)
        print("\nPredictor Rankings by Mutual Information:")
        print(rankings.head(15))
        
        preprocessor = build_preprocessing_pipeline(X_train)
        X_train_trans = preprocessor.transform(X_train)
        feature_names = get_feature_names(preprocessor, X_train)
        print(f"\nProcessed features shape: {X_train_trans.shape}")
        print("Feature Names:", feature_names[:10], "... total:", len(feature_names))
    else:
        print("Raw data not found. Please download it first.")
