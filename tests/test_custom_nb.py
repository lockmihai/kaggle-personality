import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator
from src.custom_nb import NonParametricNaiveBayesClassifier

class TestNonParametricNaiveBayesClassifier(unittest.TestCase):
    def test_basic_fit_predict(self):
        # Create synthetic classification data: 100 samples, 4 features
        X, y = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=0, random_state=42)
        
        # Add a binary discrete feature
        binary_feat = np.random.choice([0, 1], size=(100, 1))
        X = np.hstack([X, binary_feat])
        
        clf = NonParametricNaiveBayesClassifier()
        clf.fit(X, y)
        
        # Check detected features
        self.assertEqual(len(clf.continuous_features_), 4)
        self.assertEqual(clf.discrete_features_, [4])
        
        # Check predict and predict_proba shape
        probs = clf.predict_proba(X)
        self.assertEqual(probs.shape, (100, 2))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)
        
        preds = clf.predict(X)
        self.assertEqual(preds.shape, (100,))
        self.assertTrue(np.all(np.isin(preds, [0, 1])))
        
    def test_constant_feature_handling(self):
        # Create data where one continuous feature has zero variance
        X = np.random.normal(0, 1, size=(50, 2))
        X[:, 1] = 5.0 # Constant feature
        y = np.random.choice([0, 1], size=50)
        
        clf = NonParametricNaiveBayesClassifier()
        # Should not fail due to singular covariance in KDE
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        self.assertEqual(probs.shape, (50, 2))

if __name__ == '__main__':
    unittest.main()
