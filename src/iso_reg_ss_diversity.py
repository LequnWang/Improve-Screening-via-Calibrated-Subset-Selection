"""
Select a Shortlist of Applicants Based on Isotonic Regression Calibration for Each Group.
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
    parser.add_argument("--cal_data_maj_path", type=str, help="the input calibration data from the majority group")
    parser.add_argument("--cal_data_min_path", type=str, help="the input calibration data from the minority group")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--result_path", type=str, help="the output evaluation metrics path")
    parser.add_argument("--k_maj", type=float, help="the target expected number of qualified candidates for "
                                                    "the majority group")
    parser.add_argument("--k_min", type=float, help="the target expected number of qualified candidates for "
                                                    "the minority group")
    parser.add_argument("--m_maj", type=float, help="the expected number of incoming candidates from "
                                                    "the majority group")
    parser.add_argument("--m_min", type=float, help="the expected number of incoming candidates from "
                                                    "the minority group")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    k_maj = args.k_maj
    k_min = args.k_min
    m_maj = args.m_maj
    m_min = args.m_min

    # calibration
    with open(args.cal_data_maj_path, 'rb') as f:
        X_cal_maj, y_cal_maj = pickle.load(f)
    with open(args.cal_data_min_path, 'rb') as f:
        X_cal_min, y_cal_min = pickle.load(f)
    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)
    n_maj = y_cal_maj.size
    n_min = y_cal_min.size
    scores_cal_maj = classifier.predict_proba(X_cal_maj)[:, 1]
    scores_cal_min = classifier.predict_proba(X_cal_min)[:, 1]
    iso_reg_select_maj = IsotonicRegressionSelect()
    iso_reg_select_min = IsotonicRegressionSelect()
    iso_reg_select_maj.fit(scores_cal_maj, y_cal_maj, k_maj, m_maj)
    iso_reg_select_min.fit(scores_cal_min, y_cal_min, k_min, m_min)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_raw, y_test_raw, group_test_raw = pickle.load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_test_raw = transform_except_last_dim(X_test_raw, scaler)
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]
    index_maj = []
    index_min = []
    for (i, label) in enumerate(group_test_raw):
        if label == 1:
            index_maj.append(i)
        else:
            index_min.append(i)
    X_test_raw_maj, y_test_raw_maj, scores_test_raw_maj = X_test_raw[index_maj, :], y_test_raw[index_maj],\
                                                          scores_test_raw[index_maj]
    X_test_raw_min, y_test_raw_min, scores_test_raw_min = X_test_raw[index_min, :], y_test_raw[index_min], \
                                                          scores_test_raw[index_min]
    s_test_raw_maj = iso_reg_select_maj.select(scores_test_raw_maj)
    s_test_raw_min = iso_reg_select_min.select(scores_test_raw_min)
    performance_metrics = {}
    performance_metrics["num_qualified_maj"] = calculate_expected_qualified(s_test_raw_maj, y_test_raw_maj, m_maj)
    performance_metrics["num_selected_maj"] = calculate_expected_selected(s_test_raw_maj, y_test_raw_maj, m_maj)
    performance_metrics["constraint_satisfied_maj"] = True if performance_metrics["num_qualified_maj"] >= k_maj \
        else False
    performance_metrics["num_qualified_min"] = calculate_expected_qualified(s_test_raw_min, y_test_raw_min, m_min)
    performance_metrics["num_selected_min"] = calculate_expected_selected(s_test_raw_min, y_test_raw_min, m_min)
    performance_metrics["constraint_satisfied_min"] = True if performance_metrics["num_qualified_min"] >= k_min \
        else False
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)
