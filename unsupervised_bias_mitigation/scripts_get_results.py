import os
import pandas as pd

import tarfile
import yaml
from yaml.loader import SafeLoader
import torch

def get_model_scores(file_name, GAP_metric, Performance_metric, keep_original_metrics = False):
    """given the log path for a exp, read log and return the dev&test performacne, fairness, and DTO
    Args:
        exp (str): get_dir output, includeing the options and path to checkpoints
        GAP_metric (str): the target GAP metric name
        Performance_metric (str): the target performance metric name, e.g., F1, Acc.
    Returns:
        pd.DataFrame: a pandas df including dev and test scores for each epoch
    """

    epoch_id = []
    epoch_scores_dev = {"performance":[],"fairness":[]}
    epoch_scores_test = {"performance":[],"fairness":[]}

    for epoch_result_dir in [file_name]:
        epoch_result = torch.load(epoch_result_dir)

        # Track the epoch id
        epoch_id.append(epoch_result["epoch"])

        # Get fairness evaluation scores, 1-GAP, the larger the better
        epoch_scores_dev["fairness"].append(1-epoch_result["dev_evaluations"][GAP_metric])
        epoch_scores_test["fairness"].append(1-epoch_result["test_evaluations"][GAP_metric])

        epoch_scores_dev["performance"].append(epoch_result["dev_evaluations"][Performance_metric])
        epoch_scores_test["performance"].append(epoch_result["test_evaluations"][Performance_metric])

        if keep_original_metrics:
            for _dev_keys in epoch_result["dev_evaluations"].keys():
                epoch_scores_dev[_dev_keys] = (epoch_scores_dev.get(_dev_keys,[]) + [epoch_result["dev_evaluations"][_dev_keys]])
            
            for _test_keys in epoch_result["test_evaluations"].keys():
                epoch_scores_test[_test_keys] = (epoch_scores_test.get(_test_keys,[]) + [epoch_result["test_evaluations"][_test_keys]])

    epoch_results_dict = {
            "epoch":epoch_id,
        }

    for _dev_metric_keys in epoch_scores_dev.keys():
        epoch_results_dict["dev_{}".format(_dev_metric_keys)] = epoch_scores_dev[_dev_metric_keys]
    for _test_metric_keys in epoch_scores_test.keys():
        epoch_results_dict["test_{}".format(_test_metric_keys)] = epoch_scores_test[_test_metric_keys]

    epoch_scores = pd.DataFrame(epoch_results_dict)

    return epoch_scores

def load_results(
    tar_file_name,
    index_column_names = [],
    GAP_metric = "TPR_GAP",
    Performance_metric="accuracy",
    keep_original_metrics = True,
):

    with open(tar_file_name, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:gz') # Unpack tar

        best_checkpoint = []
        opt_yaml = []
        for item in tar:
            if item.name.split("/")[-1] == "BEST_checkpoint.pth.tar":
                best_checkpoint.append(item)
            elif item.name.split("/")[-1] == "opt.yaml":
                opt_yaml.append(item)
        
        with tar.extractfile(opt_yaml[0]) as f:
            _opt = yaml.load(f, Loader=SafeLoader)
        with tar.extractfile(best_checkpoint[0]) as f:
            epoch_result = get_model_scores(f, GAP_metric = GAP_metric, Performance_metric=Performance_metric, keep_original_metrics = keep_original_metrics)

        # Get hyperparameters for this epoch
        for hyperparam_key in index_column_names:
            epoch_result[hyperparam_key] = [_opt[hyperparam_key]]*len(epoch_result)
        
        epoch_result["opt_dir"] = [tar_file_name]*len(epoch_result)
    
    return epoch_result


def get_dataset_method_results(
    dataset_name,
    method_name,
    index_column_names = [],
    GAP_metric = "TPR_GAP",
    Performance_metric="accuracy",
    keep_original_metrics = True,
):

    dataset_method_exp_df = exp_df[(exp_df["dataset"] == dataset_name) & ([i.startswith(method_name) for i in exp_df["method"]])]

    result_df = []
    for tar_file_path in dataset_method_exp_df["file_path"]:
        _result_df = load_results(
            tar_file_name = tar_file_path,
            index_column_names = index_column_names,
            GAP_metric = GAP_metric,
            Performance_metric=Performance_metric,
            keep_original_metrics = keep_original_metrics,
        )
        result_df.append(_result_df)

    result_df = pd.concat(result_df)
    result_df = result_df.set_index(index_column_names)

    return result_df

exp_file_dir = r"replace_this_with_your_dir"

file_names = [f for f in os.listdir(exp_file_dir)]

exp_df = pd.DataFrame(
    {
        "file_name":file_names,
        "file_path":[os.path.join(exp_file_dir, i) for i in file_names]
    }
)

exp_df["file_size"] = [os.path.getsize(f) for f in exp_df["file_path"]]

exp_df = exp_df[exp_df["file_size"] > 10000]

exp_df["dataset"] = [i.split("@")[0] for i in list(exp_df["file_name"])]

exp_df["method"] = [i.split("@")[1].split(j)[0][:-1] for i,j in zip(list(exp_df["file_name"]), list(exp_df["dataset"]))]

exp_df.to_pickle("results/exp_df.pkl")

dataset_list = list(set(list(exp_df["dataset"])))
method_list = list(set([i.split("_")[0] for i in set(list(exp_df["method"]))]))

method2index_column_names = {
    'DecoupledAdv' : ["adv_lambda"],
    'Adv' : ["adv_lambda"],
    'UAdv' : ["knn_labels_k", "knn_labels_p", "adv_lambda"],
    'UEOCla' : ["knn_labels_k", "knn_labels_p", "DyBTalpha"],
    'EOCla' : ["DyBTalpha"],
    'Vanilla' : ["lr"],
    'EOGlb' : ["DyBTalpha"],
    'UEOGlb' : ["knn_labels_k", "knn_labels_p", "DyBTalpha"],
    'FairBatch' : ["DyBTalpha"],
    "ARL": ["adv_lr", "adv_level"],
    "UKNN": ["knn_labels_k", "knn_labels_p", "UKNN_lambda"],
}

for _dataset in dataset_list:
    for _method in method2index_column_names.keys():
        try:
            _res_df = get_dataset_method_results(
                dataset_name=_dataset,
                method_name=_method,
                index_column_names=method2index_column_names[_method],
                )
            _res_df.to_pickle("bs1024/{}_{}.pkl".format(_dataset, _method))
        except:
            print("{}_{}.pkl".format(_dataset, _method))