"""
Train a Noisy Logistic Regression Classifier from Training Data
"""
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


class NoisyLR(LogisticRegression):
    def set_noise_ratio(self, noise_ratio=None):
        self.noise_ratio = noise_ratio

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        if self.noise_ratio is not None:
            for i in range(proba.shape[0]):
                # print(X[i, -1])
                if int(X[i, -1]) == 1:
                    noise_or_not = np.random.binomial(1, self.noise_ratio["maj"])
                else:
                    noise_or_not = np.random.binomial(1, self.noise_ratio["min"])
                if noise_or_not:
                    noise = np.random.beta(1, 4)
                    proba[i, 0] = 1. - noise
                    proba[i, 1] = noise
        return proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, help="the input training data path")
    parser.add_argument("--lbd", type=float, help="L2 regularization parameter")
    # parser.add_argument("--C", type=float, help="L2 regularization parameter")
    parser.add_argument("--classifier_path", type=str, help="the output classifier path")
    parser.add_argument('--noise_ratio_maj', type=float, default=0., help="noise ratio of majority group")
    parser.add_argument('--noise_ratio_min', type=float, default=-1., help="noise ratio of minority group")

    args = parser.parse_args()

    with open(args.train_data_path, "rb") as f:
        X, y = pickle.load(f)
        n = y.shape[0]
        C = 1 / (args.lbd * n)
        # C = args.C

    if args.noise_ratio_min < 0.:
        classifier = LogisticRegression(C=C).fit(X, y)
    else:
        classifier = NoisyLR(C=C).fit(X, y)
        noise_ratio = {}
        noise_ratio["maj"] = args.noise_ratio_maj
        noise_ratio["min"] = args.noise_ratio_min
        classifier.set_noise_ratio(noise_ratio)
    with open(args.classifier_path, "wb") as f:
        pickle.dump(classifier, f)
