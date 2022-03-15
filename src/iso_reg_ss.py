"""
Select a Shortlist of Applicants Based on Isotonic Regression Calibration.
Reference: Accurate Uncertainties for Deep Learning Using Calibrated Regression Volodymyr Kuleshov et al.
"""
import argparse
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression
from utils import calculate_expected_qualified, calculate_expected_selected, transform_except_last_dim
from train_LR import NoisyLR


class IsotonicRegressionSelect(object):
    def __init__(self):
        # Hyper-parameters
        self.delta = 1e-10

        # Parameters to be learned
        self.iso_reg_model = None
        self.t = None
        # Internal variables
        self.fitted = False

    def _nudge(self, matrix):
        return ((matrix + np.random.uniform(low=0,
                                            high=self.delta,
                                            size=matrix.shape)) / (1 + self.delta))

    def fit(self, y_score, y, k, test_size):
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert (y_score.size == y.size), "Check dimensions of input matrices"

        # All required (hyper-)parameters have been passed correctly
        # Isotonic Regression Starts

        # delta-randomization
        y_score = self._nudge(y_score)
        # select items with larger scores first
        y_score = -y_score

        # build dataset for isotonic regression
        cal_size = y.size
        y_score, y = zip(*sorted(zip(y_score, y), key=lambda pair: pair[0]))
        hat_p = []
        for label in y:
            if len(hat_p) == 0:
                hat_p.append(label / cal_size)
            else:
                hat_p.append(label / cal_size + hat_p[-1])
        self.iso_reg_model = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip').fit(y_score, hat_p)
        predicted_p = self.iso_reg_model.predict(y_score)
        self.t = -np.inf
        for i in range(cal_size):
            if predicted_p[i] >= k / test_size:
                self.t = -y_score[i]
                break
        # isotonic regression selection model fitted
        self.fitted = True

    def select(self, scores):
        scores = scores.squeeze()
        size = scores.size
        # delta-randomization
        scores = self._nudge(scores)

        # make decisions
        s = np.zeros(size, dtype=bool)
        for i in range(size):
            if scores[i] >= self.t:
                s[i] = True
            else:
                s[i] = False
        return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--result_path", type=str, help="the output selection result path")
    parser.add_argument("--k", type=float, help="the target expected number of qualified candidates")
    parser.add_argument("--m", type=float, help="the expected number of incoming candidates")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    k = args.k
    m = args.m

    # calibration
    with open(args.cal_data_path, 'rb') as f:
        X_cal, y_cal = pickle.load(f)
    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)
    n = y_cal.size
    scores_cal = classifier.predict_proba(X_cal)[:, 1]
    iso_reg_select = IsotonicRegressionSelect()
    iso_reg_select.fit(scores_cal, y_cal, k, m)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_raw, y_test_raw = pickle.load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_test_raw = transform_except_last_dim(X_test_raw, scaler)
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]
    s_test_raw = iso_reg_select.select(scores_test_raw)
    performance_metrics = {}
    performance_metrics["num_qualified"] = calculate_expected_qualified(s_test_raw, y_test_raw, m)
    performance_metrics["num_selected"] = calculate_expected_selected(s_test_raw, y_test_raw, m)
    performance_metrics["constraint_satisfied"] = True if performance_metrics["num_qualified"] >= k else False
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)
