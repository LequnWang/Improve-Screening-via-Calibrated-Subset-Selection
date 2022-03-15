import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from plot_constants import *
plt.rcParams.update(params)
plt.rc('font', family='serif')


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(28, 6)
    from params_exp_noise import *
    algorithms = ["ucss", "iso_reg_ss", "platt_scal_ss"]
    algorithm_df_guarantee = {
        "css": True,
        "ucss": False,
        "iso_reg_ss": False,
        "platt_scal_ss": False
    }
    algorithm_labels = {
        "css": "CSS",
        "ucss": "Uncalibrated",
        "iso_reg_ss": "Isotonic",
        "platt_scal_ss": "Platt"
    }
    algorithm_colors = {
        "css": "tab:blue",
        "ucss": "tab:red",
        "iso_reg_ss": "tab:purple",
        "platt_scal_ss": "tab:cyan"
    }
    algorithm_markers = {
        "css": "s",
        "ucss": 9,
        "iso_reg_ss": 10,
        "platt_scal_ss": 11
    }
    for umb_num_bin in umb_num_bins:
        algorithms.append("umb_" + str(umb_num_bin))
        algorithm_labels["umb_" + str(umb_num_bin)] = "UMB {} Bins".format(umb_num_bin)
        algorithm_colors["umb_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        algorithm_df_guarantee["umb_" + str(umb_num_bin)] = True
        algorithm_markers["umb_" + str(umb_num_bin)] = umb_markers[umb_num_bin]
    algorithms.append("css")
    metrics = ["num_selected", "num_qualified", "num_unqualified", "constraint_satisfied"]
    results = {}
    for noise_ratio in noise_ratios:
        results[noise_ratio] = {}
        for algorithm in algorithms:
            results[noise_ratio][algorithm] = {}
            for metric in metrics:
                results[noise_ratio][algorithm][metric] = {}
                results[noise_ratio][algorithm][metric]["values"] = []

    for noise_ratio in noise_ratios:
        for run in runs:
            exp_identity_string = "_".join([str(n_train), str(noise_ratio), str(n_cal), lbd, str(run)])
            for algorithm in algorithms:
                result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                collect_results_normal_exp(result_path, noise_ratio, algorithm, results)

    for noise_ratio in noise_ratios:
        for algorithm in algorithms:
            for metric in metrics:
                results[noise_ratio][algorithm][metric]["mean"] = np.mean(results[noise_ratio][algorithm][metric]["values"])
                results[noise_ratio][algorithm][metric]["std"] = np.std(results[noise_ratio][algorithm][metric]["values"],
                                                            ddof=1)
    # plotting whether constraint is satisfied
    handles = []
    for algorithm in algorithms:
        mean_algorithm = np.array([results[noise_ratio][algorithm]["constraint_satisfied"]["mean"]
                                                        for noise_ratio in noise_ratios])
        std_err_algorithm = np.array(
            [results[noise_ratio][algorithm]["constraint_satisfied"]["std"] / np.sqrt(n_runs) for noise_ratio in noise_ratios])
        line = axs[0].plot(noise_ratios_label, mean_algorithm, color=algorithm_colors[algorithm],
                           marker=algorithm_markers[algorithm], linewidth=line_width,
                           label=algorithm_labels[algorithm])
        if algorithm == "css":
            handles = [line[0]] + handles
        else:
            handles.append(line[0])
        axs[0].errorbar(noise_ratios_label, mean_algorithm, std_err_algorithm, color=algorithm_colors[algorithm],
                           marker=algorithm_markers[algorithm], linewidth=line_width,
                           label=algorithm_labels[algorithm], capthick=capthick)
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[0].set_xlabel("$r_{\mathrm{noise}}$", fontsize=font_size)
    axs[0].set_ylabel("EQ", fontsize=font_size)

    # plotting the number of selected applicants
    for algorithm in algorithms:
        if not algorithm_df_guarantee[algorithm]:
            continue
        mean_algorithm = np.array([results[noise_ratio][algorithm]["num_selected"]["mean"] for noise_ratio
                                                in noise_ratios])
        std_algorithm = np.array([results[noise_ratio][algorithm]["num_selected"]["std"] for noise_ratio
                                               in noise_ratios])

        axs[1].plot(noise_ratios_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm]
                 , label=algorithm_labels[algorithm])
        axs[1].fill_between(noise_ratios_label, mean_algorithm - std_algorithm,
                         mean_algorithm + std_algorithm, alpha=transparency,
                         color=algorithm_colors[algorithm])
    axs[1].set_xlabel("$r_{\mathrm{noise}}$", fontsize=font_size)
    axs[1].set_ylabel("SS", fontsize=font_size)
    axs[1].set_ylim(top=35)
    axs[1].set_ylim(bottom=5)

    from params_exp_cal_size import *

    results = {}
    for n_cal in n_cals:
        results[n_cal] = {}
        for algorithm in algorithms:
            results[n_cal][algorithm] = {}
            for metric in metrics:
                results[n_cal][algorithm][metric] = {}
                results[n_cal][algorithm][metric]["values"] = []

    for n_cal in n_cals:
        for run in runs:
            exp_identity_string = "_".join([str(n_train), str(noise_ratio), str(n_cal), lbd, str(run)])
            for algorithm in algorithms:
                result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                collect_results_normal_exp(result_path, n_cal, algorithm, results)

    for n_cal in n_cals:
        for algorithm in algorithms:
            for metric in metrics:
                results[n_cal][algorithm][metric]["mean"] = np.mean(results[n_cal][algorithm][metric]["values"])
                results[n_cal][algorithm][metric]["std"] = np.std(results[n_cal][algorithm][metric]["values"],
                                                                  ddof=1)
    # plotting whether constraint is satisfied
    for algorithm in algorithms:
        # if algorithm_df_guarantee[algorithm] and algorithm != "css":
        #     continue
        mean_algorithm = np.array([results[n_cal][algorithm]["constraint_satisfied"]["mean"]
                                                        for n_cal in n_cals])
        std_err_algorithm = np.array(
            [results[n_cal][algorithm]["constraint_satisfied"]["std"] / np.sqrt(n_runs) for n_cal in n_cals])
        axs[2].errorbar(n_cals_label, mean_algorithm, std_err_algorithm, color=algorithm_colors[algorithm],
                        linewidth=line_width, label=algorithm_labels[algorithm], marker=algorithm_markers[algorithm],
                        capthick=capthick)
    axs[2].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[2].set_xlabel("$n$", fontsize=font_size)
    axs[2].set_ylabel("EQ", fontsize=font_size)

    # plotting the number of selected applicants
    for algorithm in algorithms:
        if not algorithm_df_guarantee[algorithm]:
            continue
        mean_algorithm = np.array([results[n_cal][algorithm]["num_selected"]["mean"] for n_cal
                                                in n_cals])
        std_algorithm = np.array([results[n_cal][algorithm]["num_selected"]["std"] for n_cal
                                               in n_cals])
        axs[3].plot(n_cals_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm]
                 , label=algorithm_labels[algorithm])
        axs[3].fill_between(n_cals_label, mean_algorithm - std_algorithm,
                         mean_algorithm + std_algorithm, alpha=transparency,
                         color=algorithm_colors[algorithm])
    axs[3].set_xlabel("$n$", fontsize=font_size)
    axs[3].set_ylabel("SS", fontsize=font_size)
    axs[3].set_ylim(top=35)
    axs[3].set_ylim(bottom=5)

    fig.legend(handles=handles, bbox_to_anchor=(0.5, 1.02), loc="upper center", ncol=5)
    plt.tight_layout(rect=[0, 0, 1, 0.78])
    fig.savefig("./plots/exp_normal.pdf", format="pdf")
