"""
Run the experiments where we vary the quality of the predictor by adding noise to the minority group
"""
import os
from exp_utils import generate_commands_diversity, submit_commands
from params_exp_diversity_noise import *
if __name__ == "__main__":
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    train_cal_maj_raw_path = "./data/data_diversity_train_cal_maj_raw_q_ratio_{}_test_ratio_{}.pkl".format(q_ratio,
                                                                                                           test_ratio)
    train_cal_min_raw_path = "./data/data_diversity_train_cal_min_raw_q_ratio_{}_test_ratio_{}.pkl".format(q_ratio,
                                                                                                           test_ratio)
    test_raw_path = "./data/data_diversity_test_raw_q_ratio_{}_test_ratio_{}.pkl".format(q_ratio, test_ratio)
    if prepare_data:
        prepare_data_command = "python ./scripts/prepare_data_diversity.py --train_cal_maj_raw_path {} " \
                               "--train_cal_min_raw_path {} --test_raw_path {} --q_ratio {} " \
                               "--test_ratio {}".format(train_cal_maj_raw_path, train_cal_min_raw_path, test_raw_path,
                                                        q_ratio, test_ratio)
        os.system(prepare_data_command)
    commands = generate_commands_diversity(exp_dir, n_train, n_trains_min, n_cal_maj, n_cals_min, n_test,
                                           n_test_maj, n_test_min, lbds, runs, n_runs_test, k_maj, k_min, alpha,
                                           classifier_type, umb_num_bins, train_cal_maj_raw_path,
                                           train_cal_min_raw_path, test_raw_path, noise_ratio_maj, noise_ratios_min)
    print(len(commands))
    if submit:
        submit_commands(exp_token, exp_dir, split_size, commands, submit)
