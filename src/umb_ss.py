"""
Select a Shortlist of Applicants Based on Uniform Mass Binning (details in the paper).
"""
import argparse
import pickle
import numpy as np
from train_LR import NoisyLR
from utils import calculate_expected_selected, calculate_expected_qualified, transform_except_last_dim


class UMBSelect(object):
    def __init__(self, n_bins, alpha):
        # Hyper-parameters
        self.n_bins = n_bins
        self.delta = 1e-10
        self.alpha = alpha

        # Parameters to be learned
        self.bin_upper_edges = None
        self.num_positives_in_bin = None
        self.num_examples = None
        self.epsilon = None
        self.b = None
        self.theta = None

        # Internal variables
        self.fitted = False

    def _get_uniform_mass_bins(self, scores):
        assert (scores.size >= 2 * self.n_bins), "Fewer points than 2 * number of bins"

        scores_sorted = np.sort(scores)

        # split scores into groups of approx equal size
        groups = np.array_split(scores_sorted, self.n_bins)
        bin_upper_edges = list()

        for cur_group in range(self.n_bins - 1):
            bin_upper_edges += [max(groups[cur_group])]
        bin_upper_edges.append(np.inf)

        return np.array(bin_upper_edges)

    def _bin_points(self, scores):
        assert (self.bin_upper_edges is not None), "Bins have not been defined"
        scores = scores.squeeze()
        assert (np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
        scores = np.reshape(scores, (scores.size, 1))
        bin_edges = np.reshape(self.bin_upper_edges, (1, self.bin_upper_edges.size))
        return np.sum(scores > bin_edges, axis=1)

    def _nudge(self, matrix):
        return ((matrix + np.random.uniform(low=0,
                                            high=self.delta,
                                            size=matrix.shape)) / (1 + self.delta))

    def fit(self, y_score, y, m, k):
        assert (self.n_bins is not None), "Number of bins has to be specified"
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert (y_score.size == y.size), "Check dimensions of input matrices"
        assert (y.size >= 2 * self.n_bins), "Number of bins should be less than two " \
                                            "times the number of calibration points"

        # All required (hyper-)parameters have been passed correctly
        # Uniform-mass binning/histogram binning code starts below
        self.num_examples = y_score.size
        self.epsilon = np.sqrt(2 * np.log(2 / alpha) / n)

        # delta-randomization
        y_score = self._nudge(y_score)

        # compute uniform-mass-bins using calibration data
        self.bin_upper_edges = self._get_uniform_mass_bins(y_score)

        # assign calibration data to bins
        bin_assignment = self._bin_points(y_score)

        # compute statistics of each bin
        self.num_positives_in_bin = np.empty(self.n_bins)
        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            self.num_positives_in_bin[i] = np.sum(np.logical_and(bin_idx, y))

        # find threshold bin and theta
        sum_scores = 0
        b = 0  # bin on the threshold
        theta = 1.
        for i in reversed(range(self.n_bins)):
            sum_scores += m * (self.num_positives_in_bin[i] / self.num_examples - self.epsilon)
            if sum_scores >= k:
                sum_scores -= m * (self.num_positives_in_bin[i] / self.num_examples - self.epsilon)
                b = i
                theta = (k - sum_scores) / (m * (self.num_positives_in_bin[i] / self.num_examples
                                                 - self.epsilon))
                break
        self.b = b
        self.theta = theta

        # histogram binning done
        self.fitted = True

    def select(self, scores):
        scores = scores.squeeze()
        size = scores.size

        # delta-randomization
        scores = self._nudge(scores)

        # assign test data to bins
        test_bins = self._bin_points(scores)

        # make decisions
        s = np.zeros(size, dtype=bool)
        for i in range(size):
            if test_bins[i] > self.b:
                s[i] = True
            elif test_bins[i] == self.b:
                s[i] = bool(np.random.binomial(1, self.theta))
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
    parser.add_argument("--alpha", type=float, default=0.1, help="the failure probability")
    parser.add_argument("--B", type=int, help="the number of bins")
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
    umb_select = UMBSelect(args.B, alpha)
    umb_select.fit(scores_cal, y_cal, m, k)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_raw, y_test_raw = pickle.load(f)
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test_raw = transform_except_last_dim(X_test_raw, scaler)
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]
    s_test_raw = umb_select.select(scores_test_raw)
    performance_metrics = {}
    performance_metrics["num_qualified"] = calculate_expected_qualified(s_test_raw, y_test_raw, m)
    performance_metrics["num_selected"] = calculate_expected_selected(s_test_raw, y_test_raw, m)
    performance_metrics["constraint_satisfied"] = True if performance_metrics["num_qualified"] >= k else False
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)
