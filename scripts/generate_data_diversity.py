"""
Generate Data For Each Run in the diversity experiments
"""
import argparse
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from exp_utils import transform_except_last_dim
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train_maj", type=int, help="number of training in the majority group")
    parser.add_argument("--n_train_min", type=int, help="number of training in the minority group")
    parser.add_argument("--n_cal_maj", type=int, help="number of calibration examples in the majority group")
    parser.add_argument("--n_cal_min", type=int, help="number of calibration examples in the minority group")
    parser.add_argument("--train_cal_maj_raw_path", type=str, help="raw majority group data path")
    parser.add_argument("--train_cal_min_raw_path", type=str, help="raw minority group data path")
    parser.add_argument("--train_data_path", type=str, help="the path for saving the training data")
    parser.add_argument("--cal_data_maj_path", type=str, help="the path for saving the calibration data for the "
                                                              "majority group")
    parser.add_argument("--cal_data_min_path", type=str, help="the path for saving the calibration data for the "
                                                              "minority group")
    parser.add_argument("--scaler_path", type=str, help="the path for saving the scaler")

    args = parser.parse_args()
    n_train_maj = args.n_train_maj
    n_train_min = args.n_train_min
    n_cal_maj = args.n_cal_maj
    n_cal_min = args.n_cal_min
    n_maj = n_train_maj + n_cal_maj
    n_min = n_train_min + n_cal_min
    with open(args.train_cal_maj_raw_path, 'rb') as f:
        X_train_cal_maj_raw, y_train_cal_maj_raw = pickle.load(f)
        X_train_cal_maj_raw, y_train_cal_maj_raw = shuffle(X_train_cal_maj_raw, y_train_cal_maj_raw)
        X_train_cal_maj, y_train_cal_maj = X_train_cal_maj_raw[:n_maj], y_train_cal_maj_raw[:n_maj]
    with open(args.train_cal_min_raw_path, 'rb') as f:
        X_train_cal_min_raw, y_train_cal_min_raw = pickle.load(f)
        X_train_cal_min_raw, y_train_cal_min_raw = shuffle(X_train_cal_min_raw, y_train_cal_min_raw)
        X_train_cal_min, y_train_cal_min = X_train_cal_min_raw[:n_min], y_train_cal_min_raw[:n_min]

    X_train_maj = X_train_cal_maj[:n_train_maj]
    y_train_maj = y_train_cal_maj[:n_train_maj]
    X_cal_maj = X_train_cal_maj[n_train_maj:]
    y_cal_maj = y_train_cal_maj[n_train_maj:]
    X_train_min = X_train_cal_min[:n_train_min]
    y_train_min = y_train_cal_min[:n_train_min]
    X_cal_min = X_train_cal_min[n_train_min:]
    y_cal_min = y_train_cal_min[n_train_min:]
    X_train = np.concatenate((X_train_maj, X_train_min), axis=0)
    y_train = np.concatenate((y_train_maj, y_train_min))
    scaler = StandardScaler()
    X_train = np.concatenate((scaler.fit_transform(X_train[:, :-1]), X_train[:, -1:]), axis=1)
    X_cal_maj = transform_except_last_dim(X_cal_maj, scaler)
    X_cal_min = transform_except_last_dim(X_cal_min, scaler)
    X_train, y_train = shuffle(X_train, y_train)

    with open(args.train_data_path, "wb") as f:
        pickle.dump([X_train, y_train], f)
    with open(args.cal_data_maj_path, "wb") as f:
        pickle.dump([X_cal_maj, y_cal_maj], f)
    with open(args.cal_data_min_path, "wb") as f:
        pickle.dump([X_cal_min, y_cal_min], f)
    with open(args.scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
