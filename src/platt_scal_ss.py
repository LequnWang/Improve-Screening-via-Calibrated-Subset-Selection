"""
Select a Shortlist of Applicants Based on the Platt Scaling Scores
"""
import argparse
import pickle
import numpy as np
from CalibrationPredictProba import CalibratedClassifierCV
from utils import calculate_expected_qualified, calculate_expected_selected, transform_except_last_dim
from train_LR import NoisyLR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--result_path", type=str, help="the output selection result path")
    parser.add_argument("--k", type=float, help="the target expected number of qualified candidates")
    parser.add_argument("--m", type=int, help="the expected number of incoming candidates")
    parser.add_argument("--n_runs_test", type=int, help="the number of tests for estimating the expectation")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    k = args.k
    m = args.m

    # calibration
    with open(args.cal_data_path, 'rb') as f:
        X_cal, y_cal = pickle.load(f)
    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)
    calibrated_classifier = CalibratedClassifierCV(base_estimator=classifier, cv='prefit')
    calibrated_classifier.fit(X_cal, y_cal)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_raw, y_test_raw = pickle.load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
        X_test_raw = transform_except_last_dim(X_test_raw, scaler)
    scores_test_raw = calibrated_classifier.predict_proba(X_test_raw)[:, 1]
    num_selected = []
    num_qualified = []
    for i in range(args.n_runs_test):
        indexes = np.random.choice(list(range(y_test_raw.size)), m)
        X_test = X_test_raw[indexes, :]
        y_test = y_test_raw[indexes]
        scores_test = scores_test_raw[indexes]
        scores_test_sorted, permutation = zip(*sorted(zip(scores_test, list(range(m))),
                                                      key=lambda pair: pair[0], reverse=True))
        s_test = np.zeros(m, dtype=bool)
        sum_scores = 0
        for j in range(m):
            sum_scores += scores_test_sorted[j]
            s_test[permutation[j]] = True
            if sum_scores >= k:
                break
        num_selected.append(calculate_expected_selected(s_test, y_test, m))
        num_qualified.append(calculate_expected_qualified(s_test, y_test, m))

    performance_metrics = {}
    performance_metrics["num_qualified"] = np.mean(num_qualified)
    performance_metrics["num_selected"] = np.mean(num_selected)
    performance_metrics["constraint_satisfied"] = True if performance_metrics["num_qualified"] >= k else False
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)
