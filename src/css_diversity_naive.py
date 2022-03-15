"""
Select a Shortlist of Applicants Based on Calibrated Subset Selection Algorithm as One Group
"""
import argparse
import pickle
import numpy as np
from train_LR import NoisyLR
from utils import calculate_expected_qualified, calculate_expected_selected, transform_except_last_dim


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
    parser.add_argument("--alpha", type=float, help="the failure probability")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    k_maj = args.k_maj
    k_min = args.k_min
    m_maj = args.m_maj
    m_min = args.m_min
    alpha = args.alpha

    # calibration
    with open(args.cal_data_maj_path, 'rb') as f:
        X_cal_maj, y_cal_maj = pickle.load(f)
    with open(args.cal_data_min_path, 'rb') as f:
        X_cal_min, y_cal_min = pickle.load(f)
    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)
    X_cal = np.concatenate((X_cal_maj, X_cal_min), axis=0)
    y_cal = np.concatenate((y_cal_maj, y_cal_min))
    n = y_cal.size
    scores_cal = classifier.predict_proba(X_cal)[:, 1]
    k = k_maj + k_min
    m = m_maj + m_min
    t_hat = find_t_hat(k=k, m=m, n=n, alpha=alpha, scores_cal=scores_cal, y_cal=y_cal)

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
    s_test_raw_maj = shortlist(t_hat, scores_test_raw_maj)
    s_test_raw_min = shortlist(t_hat, scores_test_raw_min)
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
