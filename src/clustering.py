import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def prepare_data_for_clustering(df):
    """
    Encodes categorical features and scales numerical ones.
    Returns the preprocessed numpy array and the transformer object.
    """
    # Identify numerical and categorical features
    # Living_Status and Education_Level are categorical
    cat_features = ['Living_Status', 'Education_Level']
    num_features = [col for col in df.columns if col not in cat_features]
    
    # Define column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ]
    )
    
    # Fit and transform
    scaled_data = preprocessor.fit_transform(df)
    
    # Get feature names after encoding
    ohe_categories = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = num_features + list(ohe_categories)
    
    print(f"Data scaled and encoded. Shape: {scaled_data.shape}")
    return scaled_data, preprocessor, feature_names

def apply_pca(data, n_components=3):
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca_data = pca.fit_transform(data)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA applied. Explained variance ratio for {n_components} components: {explained_var:.4f}")
    return pca_data, pca

def find_optimal_k(data, max_k=10):
    """
    Computes K-Means inertia for a range of k values to help determine optimal k (Elbow Method).
    """
    inertias = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return list(k_range), inertias

def fit_kmeans(data, n_clusters=4):
    """
    Fits K-Means clustering algorithm on the data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    print(f"K-Means fitted with k={n_clusters}.")
    return cluster_labels, kmeans

def run_clustering_pipeline(df, n_components=3, n_clusters=4):
    """
    Executes the entire clustering pipeline.
    Returns the dataframe with cluster labels, PCA data, and fitted models.
    """
    scaled_data, preprocessor, feature_names = prepare_data_for_clustering(df)
    pca_data, pca = apply_pca(scaled_data, n_components=n_components)
    cluster_labels, kmeans = fit_kmeans(pca_data, n_clusters=n_clusters)
    
    # Add labels back to the original df
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Also add PCA coordinates for visualization
    for i in range(n_components):
        df_clustered[f'PCA_Component_{i+1}'] = pca_data[:, i]
        
    return df_clustered, pca_data, preprocessor, pca, kmeans

if __name__ == "__main__":
    import os
    from data_processing import preprocess_pipeline
    
    # Direct test script
    data_path = os.path.join("data", "marketing_campaign.csv")
    if os.path.exists(data_path):
        print("Running full preprocessing and clustering pipeline...")
        df_processed = preprocess_pipeline(data_path)
        df_clustered, pca_data, preprocessor, pca, kmeans = run_clustering_pipeline(df_processed)
        print("Clustering sample results (first 5 rows):")
        print(df_clustered[['Age', 'Income', 'Total_Spend', 'Cluster']].head())
    else:
        print(f"Test run skipped: '{data_path}' not found. Place the dataset there to run this file directly.")
