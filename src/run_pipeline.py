import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_prep import (
    prepare_classification_data, evaluate_and_rank_predictors,
    build_preprocessing_pipeline, get_feature_names
)
from src.classification_runner import (
    get_models, calculate_metrics, plot_confusion_matrix,
    plot_decision_tree, plot_robustness_curves, plot_feature_importance,
    plot_pca_classification_errors, generate_classification_error_table
)

def run_classification_pipeline(data_path, output_dir="output"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Step 1: Preparing data and splitting into Train/Test/Application sets...")
    X_train, X_test, X_app, y_train, y_test, y_app = prepare_classification_data(data_path)
    
    print("\nStep 2: Evaluating and ranking predictors...")
    rankings = evaluate_and_rank_predictors(X_train, y_train)
    rankings.to_csv(os.path.join(output_dir, "predictor_rankings.csv"), index=False)
    print("Predictor rankings saved to 'output/predictor_rankings.csv'. Top 5 predictors:")
    print(rankings.head(5))
    
    print("\nStep 3: Fitting preprocessing pipeline (scaling & encoding)...")
    preprocessor = build_preprocessing_pipeline(X_train)
    
    # Transform datasets
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    X_app_trans = preprocessor.transform(X_app)
    
    feature_names = get_feature_names(preprocessor, X_train)
    
    print("\nStep 4: Training and evaluating all 11 classification models...")
    models = get_models()
    
    results = []
    y_probs_dict = {}
    fitted_models = {}
    y_preds_dict = {}
    
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        try:
            model.fit(X_train_trans, y_train)
            fitted_models[model_name] = model
            
            # Predict
            y_pred = model.predict(X_test_trans)
            y_preds_dict[model_name] = y_pred
            
            # Get probabilities for positive class (class 1)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_trans)[:, 1]
            else:
                # If model does not support probabilities (e.g. RidgeClassifier),
                # fallback using decision function or class labels
                if hasattr(model, "decision_function"):
                    df_val = model.decision_function(X_test_trans)
                    # Normalize to [0, 1]
                    y_prob = (df_val - df_val.min()) / (df_val.max() - df_val.min() + 1e-9)
                else:
                    y_prob = y_pred.astype(float)
            
            y_probs_dict[model_name] = y_prob
            
            # Compute metrics
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            metrics['Model'] = model_name
            results.append(metrics)
            
            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred, model_name, output_dir)
            
        except Exception as e:
            print(f"  Error training {model_name}: {e}")
            
    # Compile results into DataFrame
    df_results = pd.DataFrame(results)
    # Reorder columns
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    df_results = df_results[cols].sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    df_results.to_csv(os.path.join(output_dir, "model_comparison_results.csv"), index=False)
    
    print("\nModel Comparison Results (sorted by F1-Score):")
    print(df_results)
    
    print("\nStep 5: Generating robustness curves (ROC, Lift, Cumulative Gain)...")
    plot_robustness_curves(y_test, y_probs_dict, output_dir)
    
    # Identify tree-based models and plot decision tree / feature importances
    dt_model = fitted_models.get('Decision Tree')
    if dt_model:
        print("\nStep 6: Generating Decision Tree Plot...")
        plot_decision_tree(dt_model, feature_names, output_dir)
        df_dt_imp = plot_feature_importance(dt_model, feature_names, "Decision Tree", output_dir)
        df_dt_imp.to_csv(os.path.join(output_dir, "decision_tree_importance.csv"), index=False)
        
    rf_model = fitted_models.get('Random Forest')
    if rf_model:
        print("Generating Random Forest Feature Importance...")
        df_rf_imp = plot_feature_importance(rf_model, feature_names, "Random Forest", output_dir)
        df_rf_imp.to_csv(os.path.join(output_dir, "random_forest_importance.csv"), index=False)
        
    # Find the best model by F1-Score
    best_model_name = df_results.iloc[0]['Model']
    best_model = fitted_models[best_model_name]
    best_y_pred = y_preds_dict[best_model_name]
    print(f"\nBest Model identified: {best_model_name} with F1-Score: {df_results.iloc[0]['F1-Score']:.4f}")
    
    # Save the best model, preprocessor and column names
    print("Saving best model, preprocessor, and column names...")
    joblib.dump(best_model, os.path.join(output_dir, "best_model.joblib"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(X_train.columns.tolist(), os.path.join(output_dir, "train_cols.joblib"))
    
    print(f"\nStep 7: Plotting PCA 2D Classification Errors for best model ({best_model_name})...")
    plot_pca_classification_errors(X_test_trans, y_test, best_y_pred, best_model_name, output_dir)
    
    print("\nStep 8: Generating Classification Error Table for best model...")
    df_errors = generate_classification_error_table(X_test, y_test, best_y_pred)
    df_errors.to_csv(os.path.join(output_dir, "classification_errors.csv"), index=False)
    print(f"Classification errors saved to 'output/classification_errors.csv' ({df_errors.shape[0]} misclassifications out of {len(y_test)}).")
    
    print("\nStep 9: Applying the best model on the application set...")
    y_app_pred = best_model.predict(X_app_trans)
    if hasattr(best_model, "predict_proba"):
        y_app_prob = best_model.predict_proba(X_app_trans)[:, 1]
    else:
        y_app_prob = y_app_pred.astype(float)
        
    df_app_preds = X_app.copy()
    df_app_preds['Actual_Response'] = y_app
    df_app_preds['Predicted_Response'] = y_app_pred
    df_app_preds['Predicted_Probability'] = y_app_prob
    df_app_preds.to_csv(os.path.join(output_dir, "application_predictions.csv"), index=False)
    
    # Calculate performance on the application set
    app_metrics = calculate_metrics(y_app, y_app_pred, y_app_prob)
    print("\nPerformance of the Best Model on the unseen Application Set:")
    for k, v in app_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print(f"\nAll outputs successfully generated and saved to the '{output_dir}/' directory.")
    return df_results, best_model_name

if __name__ == "__main__":
    data_path = os.path.join("data", "marketing_campaign.csv")
    if os.path.exists(data_path):
        run_classification_pipeline(data_path)
    else:
        print(f"Dataset not found at {data_path}. Please download it first.")
