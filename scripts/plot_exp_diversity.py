import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from plot_constants import *
plt.rcParams.update(params)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "dejavuserif"


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(28, 5.4)
    from params_exp_diversity_noise import *
    algorithms = ["css_naive", "ucss", "iso_reg_ss", "platt_scal_ss", "css"]
    algorithm_labels = {
        "css": "CSS (Diversity)",
        "css_naive": "CSS (No Diversity)",
        "ucss": "Uncalibrated (Diversity)",
        "iso_reg_ss": "Isotonic (Diversity)",
        "platt_scal_ss": "Platt (Diversity)"
    }
    algorithm_colors = {
        "css": "tab:blue",
        "css_naive": "tab:green",
        "ucss": "tab:red",
        "iso_reg_ss": "tab:purple",
        "platt_scal_ss": "tab:cyan"
    }
    algorithm_markers = {
        "css": "s",
        "css_naive": "D",
        "ucss": 9,
        "iso_reg_ss": 10,
        "platt_scal_ss": 11
    }
    metrics = ["num_selected_maj", "num_qualified_maj", "num_unqualified_maj", "constraint_satisfied_maj",
               "num_selected_min", "num_qualified_min", "num_unqualified_min", "constraint_satisfied_min"]
    results = {}
    for noise_ratio_min in noise_ratios_min:
        results[noise_ratio_min] = {}
        for algorithm in algorithms:
            results[noise_ratio_min][algorithm] = {}
            for metric in metrics:
                results[noise_ratio_min][algorithm][metric] = {}
                results[noise_ratio_min][algorithm][metric]["values"] = []

    for noise_ratio_min in noise_ratios_min:
        for run in runs:
            exp_identity_string = "_".join([str(n_train_min), str(noise_ratio_min), str(n_cal_min), lbd, str(run)])
            for algorithm in algorithms:
                result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                collect_results_diversity_exp(result_path, noise_ratio_min, algorithm, results)

    for noise_ratio_min in noise_ratios_min:
        for algorithm in algorithms:
            for metric in metrics:
                results[noise_ratio_min][algorithm][metric]["mean"] = np.mean(
                    results[noise_ratio_min][algorithm][metric]["values"])
                results[noise_ratio_min][algorithm][metric]["std"] = np.std(
                    results[noise_ratio_min][algorithm][metric]["values"], ddof=1)

    # plotting whether the constraint is satisfied in the majority group
    handles = []
    for algorithm in algorithms:
        mean_algorithm = np.array([results[noise_ratio_min][algorithm]["constraint_satisfied_maj"]["mean"]
                                                for noise_ratio_min in noise_ratios_min])
        std_err_algorithm = np.array([results[noise_ratio_min][algorithm]["constraint_satisfied_maj"]["std"] / np.sqrt(n_runs)
                                               for noise_ratio_min in noise_ratios_min])
        line = axs[0].plot(noise_ratios_min_label, mean_algorithm, linewidth=line_width,
                     color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        if algorithm == "css":
            handles = [line[0]] + handles
        else:
            handles.append(line[0])
        axs[0].errorbar(noise_ratios_min_label, mean_algorithm, std_err_algorithm, linewidth=line_width,
                     color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm], capthick=capthick, label=algorithm_labels[algorithm])
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[0].set_xlabel("$r_{\mathrm{noise}}$ Minority", fontsize=font_size)
    axs[0].set_ylabel("EQ Majority", fontsize=font_size)

    # plotting whether the constraint is satisfied in the minority group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[noise_ratio_min][algorithm]["constraint_satisfied_min"]["mean"]
                                                for noise_ratio_min in noise_ratios_min])
        std_err_algorithm = np.array([results[noise_ratio_min][algorithm]["constraint_satisfied_min"]["std"] / np.sqrt(n_runs)
                                               for noise_ratio_min in noise_ratios_min])
        axs[1].errorbar(noise_ratios_min_label, mean_algorithm, std_err_algorithm, linewidth=line_width,
                     color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm], capthick=capthick, label=algorithm_labels[algorithm])
    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[1].set_xlabel("$r_{\mathrm{noise}}$ Minority", fontsize=font_size)
    axs[1].set_ylabel("EQ Minority", fontsize=font_size)

    # plotting the number of selected applicants in the majority group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[noise_ratio_min][algorithm]["num_selected_maj"]["mean"]
                                   for noise_ratio_min in noise_ratios_min])
        std_algorithm = np.array([results[noise_ratio_min][algorithm]["num_selected_maj"]["std"]
                                  for noise_ratio_min in noise_ratios_min])
        axs[2].plot(noise_ratios_min_label, mean_algorithm, linewidth=line_width,
                 color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        axs[2].fill_between(noise_ratios_min_label, mean_algorithm - std_algorithm,
                         mean_algorithm + std_algorithm, alpha=transparency,
                         color=algorithm_colors[algorithm])
    axs[2].yaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[2].set_xlabel("$r_{\mathrm{noise}}$ Minority", fontsize=font_size)
    axs[2].set_ylabel("SS Majority", fontsize=font_size)

    # plotting the number of selected applicants in the minority group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[noise_ratio_min][algorithm]["num_selected_min"]["mean"]
                                   for noise_ratio_min in noise_ratios_min])
        std_algorithm = np.array([results[noise_ratio_min][algorithm]["num_selected_min"]["std"]
                                  for noise_ratio_min in noise_ratios_min])
        axs[3].plot(noise_ratios_min_label, mean_algorithm, linewidth=line_width,
                 color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        axs[3].fill_between(noise_ratios_min_label, mean_algorithm - std_algorithm, mean_algorithm + std_algorithm,
                         alpha=transparency, color=algorithm_colors[algorithm])
    axs[3].yaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[3].set_xlabel("$r_{\mathrm{noise}}$ Minority", fontsize=font_size)
    axs[3].set_ylabel("SS Minority", fontsize=font_size)

    fig.legend(handles=handles, bbox_to_anchor=(0.5, 1.02), loc="upper center", ncol=5)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig("./plots/exp_diversity.pdf", format="pdf")
