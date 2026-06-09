import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.custom_nb import NonParametricNaiveBayesClassifier

def get_models():
    """
    Returns a dictionary of the 11 classification models to be evaluated.
    """
    return {
        'Naive Bayes (Gaussian)': GaussianNB(),
        'Naive Bayes (Non-parametric KDE)': NonParametricNaiveBayesClassifier(),
        'Linear Classifier (LDA)': LinearDiscriminantAnalysis(),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Linear SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Gaussian SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculates key classification metrics.
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_prob)
    }

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    """
    Plots and saves the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Response (0)', 'Response (1)'],
                yticklabels=['No Response (0)', 'Response (1)'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def plot_decision_tree(tree_model, feature_names, save_dir):
    """
    Plots and saves the decision tree diagram.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, feature_names=feature_names, 
              class_names=['No Response', 'Response'], filled=True, rounded=True)
    plt.title("Decision Tree Structure", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "decision_tree_structure.png"), dpi=300)
    plt.close()

def calculate_lift_gain(y_true, y_prob):
    """
    Computes Cumulative Gain and Lift curves.
    """
    # Create a dataframe sorted by probabilities descending
    df_temp = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df_temp = df_temp.sort_values(by='y_prob', ascending=False).reset_index(drop=True)
    
    # Calculate cumulative metrics
    df_temp['cumulative_positives'] = df_temp['y_true'].cumsum()
    total_positives = df_temp['y_true'].sum()
    
    # Gain: percentage of true positives caught up to this sample percentage
    df_temp['gain'] = df_temp['cumulative_positives'] / total_positives
    
    # Percentage of samples examined
    df_temp['sample_perc'] = (df_temp.index + 1) / len(df_temp)
    
    # Lift: ratio of cumulative gain to the baseline (sample_perc)
    df_temp['lift'] = df_temp['gain'] / df_temp['sample_perc']
    
    return df_temp

def plot_robustness_curves(y_true, y_probs_dict, save_dir):
    """
    Plots and saves ROC, Cumulative Gain, and Lift curves for all models.
    """
    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    for model_name, y_prob in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "roc_curves_comparison.png"), dpi=300)
    plt.close()

    # 2. Cumulative Gain Curves
    plt.figure(figsize=(10, 8))
    for model_name, y_prob in y_probs_dict.items():
        df_lg = calculate_lift_gain(y_true, y_prob)
        plt.plot(df_lg['sample_perc'], df_lg['gain'], label=model_name)
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline (Random)')
    plt.xlabel('Percentage of Sample')
    plt.ylabel('Cumulative Gain (Fraction of Positives)')
    plt.title('Cumulative Gain Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "cumulative_gain_curves.png"), dpi=300)
    plt.close()

    # 3. Lift Curves
    plt.figure(figsize=(10, 8))
    for model_name, y_prob in y_probs_dict.items():
        df_lg = calculate_lift_gain(y_true, y_prob)
        # Skip index 0 to avoid dividing by zero if sample_perc is 0,
        # but here we started indexing at 1, so it is fine.
        plt.plot(df_lg['sample_perc'], df_lg['lift'], label=model_name)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (Random)')
    plt.xlabel('Percentage of Sample')
    plt.ylabel('Lift')
    plt.title('Lift Curves')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "lift_curves.png"), dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_dir):
    """
    Plots feature importances for tree-based models (DT or RF).
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Take top 15 features
    top_indices = indices[:15]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_names, palette='viridis')
    plt.title(f'Top Feature Importances - {model_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()
    
    df_imp = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return df_imp

def plot_pca_classification_errors(X_test_scaled, y_true, y_pred, model_name, save_dir):
    """
    Projects the test set into 2D using PCA and plots correctly vs. incorrectly classified instances.
    """
    # Apply PCA to reduce features to 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_test_scaled)
    
    # Identify correct vs incorrect
    is_correct = (y_true == y_pred)
    
    df_plot = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Classification': np.where(is_correct, 'Corect Clasificat', 'Gresit Clasificat')
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Classification',
        palette={'Corect Clasificat': '#2ecc71', 'Gresit Clasificat': '#e74c3c'},
        alpha=0.7, style='Classification', markers={'Corect Clasificat': 'o', 'Gresit Clasificat': 'X'},
        data=df_plot
    )
    plt.title(f'PCA Projection of Correct vs. Incorrect Classifications ({model_name})')
    plt.xlabel(f'Principal Component 1 (Var: {pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 (Var: {pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"pca_classification_errors_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def generate_classification_error_table(X_test_df, y_true, y_pred):
    """
    Compiles a table containing the misclassified samples along with their original feature values.
    """
    errors_mask = (y_true != y_pred)
    df_errors = X_test_df[errors_mask].copy()
    df_errors['Actual_Response'] = y_true[errors_mask]
    df_errors['Predicted_Response'] = y_pred[errors_mask]
    
    return df_errors
