"""
Select a Shortlist of Applicants Based on the Platt Scaling Scores for Each Group
"""
import argparse
import pickle
import numpy as np
from CalibrationPredictProba import CalibratedClassifierCV
from utils import calculate_expected_qualified, calculate_expected_selected, transform_except_last_dim
from train_LR import NoisyLR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_data_maj_path", type=str, help="the input calibration data from the majority group")
    parser.add_argument("--cal_data_min_path", type=str, help="the input calibration data from the minority group")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--result_path", type=str, help="the output selection result path")
    parser.add_argument("--k_maj", type=float, help="the target expected number of qualified candidates for "
                                                    "the majority group")
    parser.add_argument("--k_min", type=float, help="the target expected number of qualified candidates for "
                                                    "the minority group")
    parser.add_argument("--m", type=int, help="the expected number of incoming candidates")
    parser.add_argument("--n_runs_test", type=int, help="the number of tests for estimating the expectation")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    k_maj = args.k_maj
    k_min = args.k_min
    m = args.m

    # calibration
    with open(args.cal_data_maj_path, 'rb') as f:
        X_cal_maj, y_cal_maj = pickle.load(f)
    with open(args.cal_data_min_path, 'rb') as f:
        X_cal_min, y_cal_min = pickle.load(f)
    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)
    calibrated_classifier_maj = CalibratedClassifierCV(base_estimator=classifier, cv='prefit')
    calibrated_classifier_maj.fit(X_cal_maj, y_cal_maj)
    calibrated_classifier_min = CalibratedClassifierCV(base_estimator=classifier, cv='prefit')
    calibrated_classifier_min.fit(X_cal_min, y_cal_min)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_raw, y_test_raw, group_test_raw = pickle.load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_test_raw = transform_except_last_dim(X_test_raw, scaler)
    index_maj = []
    index_min = []
    for (i, label) in enumerate(group_test_raw):
        if label == 1:
            index_maj.append(i)
        else:
            index_min.append(i)
    X_test_raw_maj, y_test_raw_maj = X_test_raw[index_maj], y_test_raw[index_maj]
    X_test_raw_min, y_test_raw_min = X_test_raw[index_min], y_test_raw[index_min]
    scores_test_raw_maj = calibrated_classifier_maj.predict_proba(X_test_raw_maj)[:, 1]
    scores_test_raw_min = calibrated_classifier_min.predict_proba(X_test_raw_min)[:, 1]
    group_test_raw_maj = np.array([1] * y_test_raw_maj.size)
    group_test_raw_min = np.array([2] * y_test_raw_min.size)
    X_test_raw = np.concatenate((X_test_raw_maj, X_test_raw_min), axis=0)
    y_test_raw = np.concatenate((y_test_raw_maj, y_test_raw_min))
    group_test_raw = np.concatenate((group_test_raw_maj, group_test_raw_min))
    scores_test_raw = np.concatenate((scores_test_raw_maj, scores_test_raw_min))
    num_selected_maj = []
    num_qualified_maj = []
    num_selected_min = []
    num_qualified_min = []
    for _ in range(args.n_runs_test):
        indexes = np.random.choice(list(range(y_test_raw.size)), m)
        X_test = X_test_raw[indexes, :]
        y_test = y_test_raw[indexes]
        group_test = group_test_raw[indexes]
        scores_test = scores_test_raw[indexes]
        index_maj = []
        index_min = []
        for (i, label) in enumerate(group_test):
            if label == 1:
                index_maj.append(i)
            else:
                index_min.append(i)

        X_test_maj, y_test_maj, scores_test_maj = X_test[index_maj, :], y_test[index_maj], \
                                                              scores_test[index_maj]
        X_test_min, y_test_min, scores_test_min = X_test[index_min, :], y_test[index_min], \
                                                              scores_test[index_min]

        scores_test_maj_sorted, permutation_maj = zip(*sorted(zip(scores_test_maj, list(range(scores_test_maj.size))),
                                                      key=lambda pair: pair[0], reverse=True))
        s_test_maj = np.zeros(y_test_maj.size, dtype=bool)
        sum_scores_maj = 0
        for j in range(y_test_maj.size):
            sum_scores_maj += scores_test_maj_sorted[j]
            s_test_maj[permutation_maj[j]] = True
            if sum_scores_maj >= k_maj:
                break
        num_selected_maj.append(calculate_expected_selected(s_test_maj, y_test_maj, y_test_maj.size))
        num_qualified_maj.append(calculate_expected_qualified(s_test_maj, y_test_maj, y_test_maj.size))

        scores_test_min_sorted, permutation_min = zip(*sorted(zip(scores_test_min, list(range(scores_test_min.size))),
                                                      key=lambda pair: pair[0], reverse=True))
        s_test_min = np.zeros(y_test_min.size, dtype=bool)
        sum_scores_min = 0
        for j in range(y_test_min.size):
            sum_scores_min += scores_test_min_sorted[j]
            s_test_min[permutation_min[j]] = True
            if sum_scores_min >= k_min:
                break
        num_selected_min.append(calculate_expected_selected(s_test_min, y_test_min, y_test_min.size))
        num_qualified_min.append(calculate_expected_qualified(s_test_min, y_test_min, y_test_min.size))

    performance_metrics = {}
    performance_metrics["num_qualified_maj"] = np.mean(num_qualified_maj)
    performance_metrics["num_selected_maj"] = np.mean(num_selected_maj)
    performance_metrics["constraint_satisfied_maj"] = True if performance_metrics["num_qualified_maj"] >= k_maj \
        else False
    performance_metrics["num_qualified_min"] = np.mean(num_qualified_min)
    performance_metrics["num_selected_min"] = np.mean(num_selected_min)
    performance_metrics["constraint_satisfied_min"] = True if performance_metrics["num_qualified_min"] >= k_min \
        else False
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)
