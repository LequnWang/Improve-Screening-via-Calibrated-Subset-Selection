"""
Select a Shortlist of Applicants Based on Calibrated Subset Selection Algorithm
"""
import argparse
import pickle
import numpy as np

from utils import calculate_expected_qualified, calculate_expected_selected, transform_except_last_dim
from train_LR import NoisyLR


def find_t_hat(k, m, n, alpha, scores_cal, y_cal):
    """Find the desirable threshold
        ----------
        k : float
            target expected number of qualified candidates

        m : float
            the expected number of test candidates

        n : int
            the number of calibration data

        alpha : float
            the failure probability

        scores_cal : ndarray of shape (n,) and dtype float
            the predicted scores of the candidates in the calibration data

        y_cal : ndarray of shape (n,) and dtype bool
            the ground truth qualifications of the candidates in the calibration data

        Returns
        -------
        t_hat : float
                the predicted threshold, in case the algorithm cannot guarantee k qualified, return -np.inf
    """
    scores_cal_sorted, y_cal_sorted = zip(*sorted(zip(scores_cal, y_cal), key=lambda pair: pair[0], reverse=True))
    sum_scores = 0
    target_sum_scores = n * (k / m + np.sqrt(np.log(2 / alpha) / (2 * n)))
    for i in range(n):
        sum_scores += y_cal_sorted[i]
        if sum_scores >= target_sum_scores:
            return scores_cal_sorted[i]
    return -np.inf


def shortlist(t_hat, scores):
    """Make shortlist decision
        ----------
        t_hat : float
            desirable threshold

        scores_test : ndarray
            the predicted scores of the candidates in the test set

        Returns
        -------
        s : ndarray of shape (m,) and dtype bool
                the shortlist decision
    """
    s = np.zeros(scores.size, dtype=bool)
    for i in range(scores.size):
        s[i] = scores[i] >= t_hat
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--result_path", type=str, help="the output evaluation metrics path")
    parser.add_argument("--k", type=float, help="the target expected number of qualified candidates")
    parser.add_argument("--m", type=float, help="the expected number of incoming candidates")
    parser.add_argument("--alpha", type=float, help="the failure probability")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    k = args.k
    m = args.m
    alpha = args.alpha

    # calibration
    with open(args.cal_data_path, 'rb') as f:
        X_cal, y_cal = pickle.load(f)
    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)

    n = y_cal.size
    scores_cal = classifier.predict_proba(X_cal)[:, 1]
    t_hat = find_t_hat(k=k, m=m, n=n, alpha=alpha, scores_cal=scores_cal, y_cal=y_cal)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_raw, y_test_raw = pickle.load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_test_raw = transform_except_last_dim(X_test_raw, scaler)
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]
    s_test_raw = shortlist(t_hat, scores_test_raw)
    performance_metrics = {}
    performance_metrics["num_qualified"] = calculate_expected_qualified(s_test_raw, y_test_raw, m)
    performance_metrics["num_selected"] = calculate_expected_selected(s_test_raw, y_test_raw, m)
    performance_metrics["constraint_satisfied"] = True if performance_metrics["num_qualified"] >= k else False
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)
