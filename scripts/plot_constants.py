import pickle
params = {'legend.fontsize': 28,
          'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'lines.markersize': 15,
          'errorbar.capsize': 8.0,
          }
line_width = 3.0
transparency = 0.2
font_size = 28
capthick = 3.0
dpi = 100


def collect_results_normal_exp(result_path, exp_parameter, algorithm, results):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    results[exp_parameter][algorithm]["num_selected"]["values"].append(result["num_selected"])
    results[exp_parameter][algorithm]["num_qualified"]["values"].append(result["num_qualified"])
    results[exp_parameter][algorithm]["num_unqualified"]["values"].append(result["num_selected"] -
                                                                          result["num_qualified"])
    results[exp_parameter][algorithm]["constraint_satisfied"]["values"].append(result["constraint_satisfied"])


def collect_results_diversity_exp(result_path, exp_parameter, algorithm, results):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    results[exp_parameter][algorithm]["num_selected_maj"]["values"].append(result["num_selected_maj"])
    results[exp_parameter][algorithm]["num_qualified_maj"]["values"].append(result["num_qualified_maj"])
    results[exp_parameter][algorithm]["num_unqualified_maj"]["values"].append(result["num_selected_maj"] -
                                                                          result["num_qualified_maj"])
    results[exp_parameter][algorithm]["constraint_satisfied_maj"]["values"].append(result["constraint_satisfied_maj"])
    results[exp_parameter][algorithm]["num_selected_min"]["values"].append(result["num_selected_min"])
    results[exp_parameter][algorithm]["num_qualified_min"]["values"].append(result["num_qualified_min"])
    results[exp_parameter][algorithm]["num_unqualified_min"]["values"].append(result["num_selected_min"] -
                                                                              result["num_qualified_min"])
    results[exp_parameter][algorithm]["constraint_satisfied_min"]["values"].append(result["constraint_satisfied_min"])
