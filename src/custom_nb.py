import numpy as np
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class NonParametricNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    A Naive Bayes classifier that uses:
    - Kernel Density Estimation (KDE) for continuous features.
    - Laplace-smoothed histograms (categorical relative frequencies) for discrete features.
    
    Automatically determines feature type based on the number of unique values.
    """
    def __init__(self, eps=1e-9, max_unique_discrete=2):
        self.eps = eps
        self.max_unique_discrete = max_unique_discrete

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Calculate class priors
        self.class_priors_ = np.zeros(self.n_classes_)
        for idx, c in enumerate(self.classes_):
            self.class_priors_[idx] = np.mean(y == c)
            
        # Detect feature types (continuous vs discrete)
        self.continuous_features_ = []
        self.discrete_features_ = []
        for col in range(self.n_features_):
            unique_vals = np.unique(X[:, col])
            if len(unique_vals) <= self.max_unique_discrete:
                self.discrete_features_.append(col)
            else:
                self.continuous_features_.append(col)
                
        # Train estimators per class and per feature
        # self.kdes_[class_idx][feature_idx]
        self.kdes_ = {c: {} for c in self.classes_}
        # self.discrete_probs_[class_idx][feature_idx] = {value: probability}
        self.discrete_probs_ = {c: {} for c in self.classes_}
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            
            # Continuous features: KDE
            for col in self.continuous_features_:
                vals = X_c[:, col]
                # If all values are identical, KDE will fail due to singular covariance matrix.
                # In that case, add tiny noise (jitter) to make variance positive.
                if np.var(vals) < 1e-8:
                    vals = vals + np.random.normal(0, 1e-4, size=len(vals))
                
                try:
                    kde = gaussian_kde(vals)
                except Exception:
                    # Fallback if KDE still fails: use a simple normal distribution approximation
                    mean_val = np.mean(vals)
                    std_val = np.std(vals) if np.std(vals) > 1e-4 else 1e-4
                    
                    class FallbackKDE:
                        def __init__(self, m, s):
                            self.m = m
                            self.s = s
                        def evaluate(self, x):
                            # Gaussian PDF formula
                            return (1.0 / (self.s * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - self.m) / self.s) ** 2)
                    kde = FallbackKDE(mean_val, std_val)
                    
                self.kdes_[c][col] = kde
                
            # Discrete features: frequency table with Laplace smoothing
            for col in self.discrete_features_:
                vals = X_c[:, col]
                unique_vals, counts = np.unique(vals, return_counts=True)
                counts_dict = dict(zip(unique_vals, counts))
                
                # We assume features are binary/discrete. We find all possible values of this feature in X
                all_possible_vals = np.unique(X[:, col])
                total_counts = len(vals) + len(all_possible_vals) # + V for Laplace smoothing
                
                probs = {}
                for val in all_possible_vals:
                    val_count = counts_dict.get(val, 0) + 1 # Laplace smoothing
                    probs[val] = val_count / total_counts
                self.discrete_probs_[c][col] = probs
                
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        log_posteriors = np.zeros((n_samples, self.n_classes_))
        
        for idx, c in enumerate(self.classes_):
            # Start with log prior
            log_post = np.log(self.class_priors_[idx] + self.eps)
            
            # Continuous features: add log KDE density
            for col in self.continuous_features_:
                kde = self.kdes_[c][col]
                # Evaluate KDE for all samples
                vals = X[:, col]
                # gaussian_kde.evaluate or __call__ evaluates density. We handle both custom and standard
                if hasattr(kde, 'evaluate'):
                    densities = kde.evaluate(vals)
                else:
                    densities = kde(vals)
                log_post += np.log(np.maximum(densities, self.eps))
                
            # Discrete features: add log probabilities
            for col in self.discrete_features_:
                probs = self.discrete_probs_[c][col]
                vals = X[:, col]
                # For each sample, fetch its probability. If value is unseen, use Laplace smoothed baseline.
                all_vals = list(probs.keys())
                min_prob = 1.0 / (len(X) + len(all_vals))
                
                # Map values to their probability
                sample_probs = np.array([probs.get(v, min_prob) for v in vals])
                log_post += np.log(sample_probs + self.eps)
                
            log_posteriors[:, idx] = log_post
            
        # Log-Sum-Exp trick to get stable probabilities
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_post = np.exp(log_posteriors - max_log)
        probs = exp_post / np.sum(exp_post, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[idx] for idx in class_indices])
