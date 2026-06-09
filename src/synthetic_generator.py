import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_prep import prepare_classification_data
from src.classification_runner import calculate_metrics

def filter_synthetic_samples(model, X_generated, y_generated, threshold=0.55):
    """
    Uses the trained classifier to filter synthetic samples.
    Keeps only those samples where the classifier agrees with the synthetic label with confidence >= threshold.
    """
    if len(X_generated) == 0:
        return X_generated, y_generated
        
    if not hasattr(model, "predict_proba"):
        # Fallback to hard predictions if probability is not available
        preds = model.predict(X_generated)
        mask = (preds == y_generated)
        return X_generated[mask], y_generated[mask]
        
    probs = model.predict_proba(X_generated)[:, 1]
    
    # For class 1, predicted prob of class 1 must be >= threshold
    # For class 0, predicted prob of class 1 must be <= (1 - threshold)
    mask = np.zeros(len(y_generated), dtype=bool)
    mask[y_generated == 1] = (probs[y_generated == 1] >= threshold)
    mask[y_generated == 0] = (probs[y_generated == 0] <= (1 - threshold))
    
    return X_generated[mask], y_generated[mask]

def generate_synthetic_data(X_train, y_train, model, method='smote', target_ratio=0.5, filter_threshold=0.55, random_state=42):
    """
    Generates synthetic samples to balance the minority class (Response = 1) using different methods.
    Returns:
        X_augmented, y_augmented, X_only_synthetic, y_only_synthetic
    """
    np.random.seed(random_state)
    classes, counts = np.unique(y_train, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]
    n_majority = counts[np.argmax(counts)]
    n_minority = counts[np.argmin(counts)]
    
    # Target number of total minority samples
    n_target = int(n_majority * target_ratio)
    n_to_generate = n_target - n_minority
    
    if n_to_generate <= 0:
        print(f"Minority class is already at or above target ratio ({n_minority}/{n_majority}). No generation needed.")
        return X_train, y_train, np.empty((0, X_train.shape[1])), np.empty(0)
        
    synthetic_samples = []
    synthetic_labels = []
    
    max_iters = 15
    iter_count = 0
    
    X_min = X_train[y_train == minority_class]
    
    if method == 'smote':
        # Custom SMOTE implementation
        k_neighbors = min(5, len(X_min) - 1)
        if k_neighbors < 1:
            # Fallback to simple duplication with noise if not enough neighbors
            method = 'jitter'
        else:
            nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_min)
            distances, indices = nbrs.kneighbors(X_min)
            
            while len(synthetic_samples) < n_to_generate and iter_count < max_iters:
                needed = n_to_generate - len(synthetic_samples)
                chunk_size = needed * 2  # Generate extra for filtering
                chunk_samples = []
                
                for _ in range(chunk_size):
                    idx = np.random.randint(0, len(X_min))
                    # Pick a random neighbor (excluding itself at index 0)
                    nn_idx = np.random.randint(1, k_neighbors + 1)
                    neighbor_idx = indices[idx, nn_idx]
                    
                    diff = X_min[neighbor_idx] - X_min[idx]
                    gap = np.random.rand()
                    chunk_samples.append(X_min[idx] + gap * diff)
                    
                chunk_samples = np.vstack(chunk_samples)
                chunk_labels = np.full(len(chunk_samples), minority_class)
                
                if filter_threshold > 0.5:
                    chunk_samples, chunk_labels = filter_synthetic_samples(model, chunk_samples, chunk_labels, filter_threshold)
                    
                if len(chunk_samples) > 0:
                    synthetic_samples.extend(chunk_samples)
                    synthetic_labels.extend(chunk_labels)
                iter_count += 1
                
    if method == 'gmm':
        # Gaussian Mixture Model sampling
        gmm = GaussianMixture(n_components=min(4, len(X_min) // 10 + 1), covariance_type='full', random_state=random_state)
        gmm.fit(X_min)
        
        while len(synthetic_samples) < n_to_generate and iter_count < max_iters:
            needed = n_to_generate - len(synthetic_samples)
            chunk_size = needed * 2
            chunk_samples, _ = gmm.sample(chunk_size)
            chunk_labels = np.full(len(chunk_samples), minority_class)
            
            if filter_threshold > 0.5:
                chunk_samples, chunk_labels = filter_synthetic_samples(model, chunk_samples, chunk_labels, filter_threshold)
                
            if len(chunk_samples) > 0:
                synthetic_samples.extend(chunk_samples)
                synthetic_labels.extend(chunk_labels)
            iter_count += 1
            
    if method == 'jitter':
        # Add Gaussian noise (jittering)
        feature_stds = np.std(X_train, axis=0)
        # Prevent 0 std features from blowing up or remaining 0
        feature_stds[feature_stds < 1e-5] = 1e-5
        
        while len(synthetic_samples) < n_to_generate and iter_count < max_iters:
            needed = n_to_generate - len(synthetic_samples)
            chunk_size = needed * 2
            
            base_indices = np.random.choice(len(X_min), size=chunk_size, replace=True)
            X_base = X_min[base_indices]
            
            # Add small noise (10% of feature std dev)
            noise = np.random.normal(0, 0.1, size=X_base.shape) * feature_stds
            chunk_samples = X_base + noise
            chunk_labels = np.full(len(chunk_samples), minority_class)
            
            if filter_threshold > 0.5:
                chunk_samples, chunk_labels = filter_synthetic_samples(model, chunk_samples, chunk_labels, filter_threshold)
                
            if len(chunk_samples) > 0:
                synthetic_samples.extend(chunk_samples)
                synthetic_labels.extend(chunk_labels)
            iter_count += 1
            
    # Trim to exactly the needed count
    if len(synthetic_samples) > n_to_generate:
        synthetic_samples = synthetic_samples[:n_to_generate]
        synthetic_labels = synthetic_labels[:n_to_generate]
        
    if len(synthetic_samples) > 0:
        X_only_syn = np.vstack(synthetic_samples)
        y_only_syn = np.array(synthetic_labels)
        X_augmented = np.vstack([X_train, X_only_syn])
        y_augmented = np.concatenate([y_train, y_only_syn])
        return X_augmented, y_augmented, X_only_syn, y_only_syn
    else:
        return X_train, y_train, np.empty((0, X_train.shape[1])), np.empty(0)

def postprocess_synthetic_data(df_syn, train_cols):
    """
    Applies logic constraints and rounds integer features to ensure the synthetic raw data
    is consistent and realistic.
    """
    df_post = df_syn.copy()
    
    # 1. Reorder columns to match original training features
    df_post = df_post[train_cols]
    
    # 2. Identify numeric columns that are counts or identifiers and round/clip them
    # Define integer columns that should not be floats
    int_cols = [
        'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
        'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 
        'Age', 'Customer_Tenure_Days'
    ]
    
    binary_cols = [
        'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 
        'Complain', 'Is_Parent'
    ]
    
    # Clip continuous and integer numerical columns to be non-negative
    numeric_cols = int_cols + ['Income']
    for col in numeric_cols:
        if col in df_post.columns:
            df_post[col] = df_post[col].clip(lower=0)
            
    # Round integer columns
    for col in int_cols:
        if col in df_post.columns:
            df_post[col] = np.round(df_post[col]).astype(int)
            
    # Clip and round binary columns
    for col in binary_cols:
        if col in df_post.columns:
            df_post[col] = np.round(df_post[col].clip(0, 1)).astype(int)
            
    # 3. Recalculate derived features to guarantee perfect logical consistency
    df_post['Children'] = df_post['Kidhome'] + df_post['Teenhome']
    df_post['Is_Parent'] = (df_post['Children'] > 0).astype(int)
    
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df_post['Total_Spend'] = df_post[spend_cols].sum(axis=1)
    
    purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df_post['Total_Purchases'] = df_post[purchase_cols].sum(axis=1)
    
    # Floor values of Age and Tenure to reasonable caps if they went too high
    df_post['Age'] = df_post['Age'].clip(18, 100)
    
    return df_post

def run_synthetic_pipeline(data_path="data/marketing_campaign.csv", output_dir="output"):
    print("--- PIPELINE GENERARE DATE ARTIFICIALE ---")
    
    # 1. Load data
    print("\n[Step 1] Încărcare date originale...")
    X_train_raw, X_test_raw, X_app_raw, y_train, y_test, y_app = prepare_classification_data(data_path)
    
    # 2. Load preprocessor and best model
    print("\n[Step 2] Încărcare preprocesor și model antrenat de referință...")
    try:
        preprocessor = joblib.load(os.path.join(output_dir, "preprocessor.joblib"))
        best_model = joblib.load(os.path.join(output_dir, "best_model.joblib"))
        train_cols = joblib.load(os.path.join(output_dir, "train_cols.joblib"))
    except FileNotFoundError as e:
        print(f"Error loading trained assets: {e}")
        print("Please run 'src/run_pipeline.py' first to train and save the model assets.")
        return
        
    # Transform data
    X_train_trans = preprocessor.transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)
    
    # Get details on class distribution
    classes, counts = np.unique(y_train, return_counts=True)
    print(f"Distribuție originală Train set:")
    print(f"  Clasa 0 (Non-Response): {counts[0]} probe")
    print(f"  Clasa 1 (Response):     {counts[1]} probe ({counts[1]/len(y_train)*100:.2f}%)")
    
    # 3. Evaluate different synthetic generation methods
    methods = ['smote', 'gmm', 'jitter']
    results = []
    
    # Baseline performance (Original Train only)
    print("\n[Step 3] Evaluare performanță model Baseline (fără augmentare)...")
    baseline_model = clone(best_model)
    baseline_model.fit(X_train_trans, y_train)
    y_pred_base = baseline_model.predict(X_test_trans)
    
    if hasattr(baseline_model, "predict_proba"):
        y_prob_base = baseline_model.predict_proba(X_test_trans)[:, 1]
    else:
        y_prob_base = y_pred_base.astype(float)
        
    metrics_base = calculate_metrics(y_test, y_pred_base, y_prob_base)
    metrics_base['Metodă Augmentare'] = 'Baseline (Original)'
    metrics_base['Dimensiune Train'] = len(X_train_trans)
    results.append(metrics_base)
    
    # Dictionary to keep the best generated preprocessed data
    best_syn_X_trans = None
    best_syn_y = None
    best_method = None
    best_f1 = metrics_base['F1-Score']
    
    # Test each method
    for method in methods:
        print(f"\nGenerare date cu metoda: {method.upper()}...")
        # Target: minority is 50% of majority class
        X_aug, y_aug, X_only_syn, y_only_syn = generate_synthetic_data(
            X_train_trans, y_train, best_model, method=method, 
            target_ratio=0.5, filter_threshold=0.55
        )
        
        print(f"  Generat {len(X_only_syn)} probe noi de clasa 1.")
        print(f"  Dimensiune Train Set Nou: {len(X_aug)} probe")
        
        # Retrain model
        aug_model = clone(best_model)
        aug_model.fit(X_aug, y_aug)
        
        # Evaluate
        y_pred_aug = aug_model.predict(X_test_trans)
        if hasattr(aug_model, "predict_proba"):
            y_prob_aug = aug_model.predict_proba(X_test_trans)[:, 1]
        else:
            y_prob_aug = y_pred_aug.astype(float)
            
        metrics_aug = calculate_metrics(y_test, y_pred_aug, y_prob_aug)
        metrics_aug['Metodă Augmentare'] = f'Augmented ({method.upper()})'
        metrics_aug['Dimensiune Train'] = len(X_aug)
        results.append(metrics_aug)
        
        # Track best method by F1-Score
        if metrics_aug['F1-Score'] > best_f1:
            best_f1 = metrics_aug['F1-Score']
            best_syn_X_trans = X_only_syn
            best_syn_y = y_only_syn
            best_method = method
            
    # Print results
    df_res = pd.DataFrame(results)
    cols = ['Metodă Augmentare', 'Dimensiune Train', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    df_res = df_res[cols]
    print("\n=== REZULTATE COMPARATIVE PE SETUL DE TEST ===")
    print(df_res.to_string(index=False))
    
    # 4. Generate and save the best synthetic dataset
    if best_method is not None:
        print(f"\n[Step 4] Metoda câștigătoare: {best_method.upper()} (F1-Score maxim pe test set)")
        print("Reconstrucție date artificiale în spațiul original de trăsături...")
        
        # We need to inverse transform the best synthetic features
        n_num_cols = len([c for c in train_cols if c not in ['Living_Status', 'Education_Level']])
        X_num_syn = best_syn_X_trans[:, :n_num_cols]
        X_cat_syn = best_syn_X_trans[:, n_num_cols:]
        
        # Inverse transform
        X_num_orig = preprocessor.named_transformers_['num'].inverse_transform(X_num_syn)
        X_cat_orig = preprocessor.named_transformers_['cat'].inverse_transform(X_cat_syn)
        
        # Build DataFrames
        num_cols = [c for c in train_cols if c not in ['Living_Status', 'Education_Level']]
        cat_cols = ['Living_Status', 'Education_Level']
        
        df_num = pd.DataFrame(X_num_orig, columns=num_cols)
        df_cat = pd.DataFrame(X_cat_orig, columns=cat_cols)
        
        # Combine
        df_syn = pd.concat([df_num, df_cat], axis=1)
        
        # Apply logic constraints and roundings
        df_syn_clean = postprocess_synthetic_data(df_syn, train_cols)
        df_syn_clean['Response'] = best_syn_y.astype(int)
        
        # Save to CSV
        output_csv = os.path.join(output_dir, "synthetic_marketing_data.csv")
        df_syn_clean.to_csv(output_csv, index=False)
        print(f"Datele artificiale au fost salvate cu succes în '{output_csv}' ({len(df_syn_clean)} instanțe).")
        
        # Check first 5 rows
        print("\nExemplu de date artificiale generate (primele 5 rânduri):")
        print(df_syn_clean.head(5).to_string())
    else:
        print("\n[Step 4] Nicio metodă de augmentare nu a îmbunătățit scorul F1 comparat cu Baseline.")
        print("Se salvează totuși datele generate prin SMOTE ca fallback standard...")
        
        # Run SMOTE again to get features to save
        X_aug, y_aug, X_only_syn, y_only_syn = generate_synthetic_data(
            X_train_trans, y_train, best_model, method='smote', 
            target_ratio=0.5, filter_threshold=0.55
        )
        
        n_num_cols = len([c for c in train_cols if c not in ['Living_Status', 'Education_Level']])
        X_num_syn = X_only_syn[:, :n_num_cols]
        X_cat_syn = X_only_syn[:, n_num_cols:]
        
        X_num_orig = preprocessor.named_transformers_['num'].inverse_transform(X_num_syn)
        X_cat_orig = preprocessor.named_transformers_['cat'].inverse_transform(X_cat_syn)
        
        num_cols = [c for c in train_cols if c not in ['Living_Status', 'Education_Level']]
        cat_cols = ['Living_Status', 'Education_Level']
        
        df_num = pd.DataFrame(X_num_orig, columns=num_cols)
        df_cat = pd.DataFrame(X_cat_orig, columns=cat_cols)
        df_syn = pd.concat([df_num, df_cat], axis=1)
        df_syn_clean = postprocess_synthetic_data(df_syn, train_cols)
        df_syn_clean['Response'] = y_only_syn.astype(int)
        
        output_csv = os.path.join(output_dir, "synthetic_marketing_data.csv")
        df_syn_clean.to_csv(output_csv, index=False)
        print(f"Datele artificiale fallback (SMOTE) salvate în '{output_csv}'.")

if __name__ == "__main__":
    run_synthetic_pipeline()
