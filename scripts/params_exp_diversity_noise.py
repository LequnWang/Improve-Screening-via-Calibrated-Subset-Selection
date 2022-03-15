exp_token = "dns"
exp_dir = "./exp_diversity_noise"
q_ratio = "0.2"
test_ratio = "0.5"
maj_ratio = 0.755558
prepare_data = False
submit = True
split_size = 1000
n_test = 100
n_test_maj = 100 * maj_ratio
n_test_min = 100 - n_test_maj
k_maj = 5 * maj_ratio
k_min = 5. - k_maj
alpha = "0.1"
n_runs = 100
n_runs_test = 1000
n_train = 10000
n_train_min = int(n_train * (1 - maj_ratio))
n_trains_min = [n_train_min]
noise_ratio_maj = "0"
noise_ratios_min = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]
noise_ratios_min_label = noise_ratios_min
n_cal = 10000
n_cal_maj = int(n_cal * maj_ratio)
n_cal_min = n_cal - n_cal_maj
n_cals_min = [n_cal_min]
runs = list(range(n_runs))
classifier_type = "LR"
lbd = "1e-6"
lbds = ["1e-6"]
umb_num_bins = [1, 2, 3, 4, 5]
umb_colors = {1: "tab:orange", 2: "tab:brown", 3: "tab:pink", 4: "tab:gray", 5: "tab:olive"}
umb_markers = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}
