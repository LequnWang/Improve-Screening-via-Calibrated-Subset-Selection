"""
Prepare data to satisfy the ratio between qualified and unqualified across two groups
"""
import argparse
import pickle

import numpy as np
from sklearn.utils import shuffle
from exp_utils import satisfy_ratio

from folktables import ACSDataSource, ACSEmployment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cal_maj_raw_path", type=str, help="the path for saving the raw data for sampling "
                                                                   "train and calibration data from the majority group")
    parser.add_argument("--train_cal_min_raw_path", type=str, help="the path for saving the raw data for sampling "
                                                                   "train and calibration data from the minority group")
    parser.add_argument("--test_raw_path", type=str, help="the path for saving the raw data for sampling the test data")
    parser.add_argument("--q_ratio", type=float, default=0.2, help="percentage of qualified applicants")
    parser.add_argument("--test_ratio", type=float, default=0.5, help="percentage data as test data")
    args = parser.parse_args()
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(download=True)
    X, y, group = ACSEmployment.df_to_numpy(acs_data)
    X, y, group = shuffle(X, y, group)
    # y = 1 - y
    index_maj = []
    index_min = []
    for (i, label) in enumerate(group):
        if label == 1:
            index_maj.append(i)
        else:
            index_min.append(i)

    # satisfy ratio for each group
    X_maj, y_maj = X[index_maj], y[index_maj]
    X_min, y_min = X[index_min], y[index_min]
    print("Before down-sample majority group size {}".format(y_maj.size))
    print("Before down-sample majority group positive size {}".format(np.sum(y_maj.size)))
    print("Before down-sample minority group size {}".format(y_min.size))
    print("Before down-sample minority group positive size {}".format(np.sum(y_min.size)))
    X_maj, y_maj = satisfy_ratio(X_maj, y_maj, args.q_ratio)
    X_min, y_min = satisfy_ratio(X_min, y_min, args.q_ratio)
    print("After down-sample majority group size {}".format(y_maj.size))
    print("After down-sample majority group positive size {}".format(np.sum(y_maj)))
    print("After down-sample minority group size {}".format(y_min.size))
    print("After down-sample minority group positive size {}".format(np.sum(y_min)))

    maj_test_size = int(y_maj.size * args.test_ratio)
    min_test_size = int(y_min.size * args.test_ratio)
    X_maj_test = X_maj[:maj_test_size]
    X_maj_train_cal = X_maj[maj_test_size:]
    y_maj_test = y_maj[:maj_test_size]
    y_maj_train_cal = y_maj[maj_test_size:]
    X_min_test = X_min[:min_test_size]
    X_min_train_cal = X_min[min_test_size:]
    y_min_test = y_min[:min_test_size]
    y_min_train_cal = y_min[min_test_size:]
    print("After down-sample majority group test size {}".format(y_maj_test.size))
    print("After down-sample majority group test positive size {}".format(np.sum(y_maj_test)))
    print("After down-sample minority group test size {}".format(y_min_test.size))
    print("After down-sample minority group test positive size {}".format(np.sum(y_min_test)))
    print("After down-sample majority ratio {}".format(1. * y_maj_test.size / (y_maj_test.size + y_min_test.size)))
    group_maj_test = np.array([1] * y_maj_test.size)
    group_min_test = np.array([2] * y_min_test.size)
    X_test = np.concatenate((X_maj_test, X_min_test), axis=0)
    y_test = np.concatenate((y_maj_test, y_min_test))
    group_test = np.concatenate((group_maj_test, group_min_test))
    X_test, y_test, group_test = shuffle(X_test, y_test, group_test)

    with open(args.test_raw_path, "wb") as f:
        pickle.dump([X_test, y_test, group_test], f)
    with open(args.train_cal_maj_raw_path, "wb") as f:
        pickle.dump([X_maj_train_cal, y_maj_train_cal], f)
    with open(args.train_cal_min_raw_path, "wb") as f:
        pickle.dump([X_min_train_cal, y_min_train_cal], f)
