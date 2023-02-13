import argparse
from pathlib import Path
import pandas as pd

from fairlib.src import analysis
from fairlib.src.base_options import BaseOptions
from fairlib.src import networks
from fairlib.src.dataloaders import get_dataloaders, default_dataset_roots
from fairlib.src.evaluators.evaluator import gap_eval_scores

import tarfile
import yaml
from yaml.loader import SafeLoader
import torch
import statistics
import numpy as np

def load_inDomain_model(tar_file_name, device):
    # unpack tar.gz file and load checkpoints
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
            _model = torch.load(f,map_location=device)["model"]

    options = BaseOptions()
    _opt["device_id"] = -1
    _state = options.get_state(
            args=_opt,
            silence=True)
    _state.device = device

    # Init the main task model and load the trained parameters
    model = networks.get_main_model(_state)
    model.load_state_dict(_model)
    model.to(device)
    model.eval()

    return model

def get_evaluation_scores(model, iterator, args):
    (_, preds, labels, private_labels) = networks.utils.eval_epoch(
            model = model, 
            iterator = iterator, 
            args = args)

    scores, _ = gap_eval_scores(
                        y_pred=preds,
                        y_true=labels, 
                        protected_attribute=private_labels,
                        args = args,
                        )
    return scores

class target_dataset_args:
    batch_size = 1024
    num_workers = 0
    encoder_architecture = "Fixed"
    full_label = True
    regression = False
    GBT = False
    BT = None
    DyBT = None
    adv_BT = None
    adv_decoupling = False
    test_batch_size = 1024
    gated = False
    adv_debiasing = False
    FCL = False

if __name__ == '__main__':
    """Evaluation cross domain bias mitigation
    
    E.g., python .\cross_domain_analysis.py --source_dataset Bios_gender --method EOCla --target_dataset Bios_economy
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--target_dataset', type=str, required=True)
    parser.add_argument('--fils_path', type=str, default="bs1024")
    parser.add_argument('--performance_metric', type=str, default="accuracy")
    parser.add_argument('--fairness_metric', type=str, default="TPR_GAP")
    parser.add_argument('--device_id', type=int, default=0, help='device id, -1 is cpu')
    parser.add_argument('--save_dir', type=str, default="cross_domain")

    args = parser.parse_args()

    if args.device_id < 0:
        args.device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.device_id)
        args.device = torch.device("cuda:{}".format(args.device_id))

    # print(args.source_dataset, args.method, args.target_dataset)

    # Get in-domain pareto frontiers
    PREPROCESSED_FILES_DIR = Path(args.fils_path)
    df = pd.read_pickle(PREPROCESSED_FILES_DIR / "{}_{}.pkl".format(args.source_dataset, args.method))
    # print(df)
    splits = ["dev", "test"]
    InDomain_Method_dfs = {}
    for _split in splits:
        InDomain_Method_dfs[_split] = analysis.final_results_df(
            results_dict = {args.method:df},
            pareto = True, pareto_selection = _split,
            selection_criterion = None, return_dev = True,
            )
        # print(InDomain_Method_dfs[_split])


        dev_target_domain_performances_mean = []
        dev_target_domain_fairness_mean = []
        test_target_domain_performances_mean = []
        test_target_domain_fairness_mean = []

        dev_target_domain_performances_std = []
        dev_target_domain_fairness_std = []
        test_target_domain_performances_std = []
        test_target_domain_fairness_std = []

        # Iterative over each Pareto model and evaluate its cross-domain results
        for _opt_dir_list in list(InDomain_Method_dfs[_split]["opt_dir list"]):
            # Each hyperparameter combination is associated with multiple runs
            # Here we calculate its mean and variance 
            _dev_target_domain_performances = []
            _dev_target_domain_fairness = []
            _test_target_domain_performances = []
            _test_target_domain_fairness = []

            target_args = target_dataset_args()
            target_args.dataset = args.target_dataset
            target_args.data_dir = default_dataset_roots[target_args.dataset]
            target_args.device = args.device

            _, target_dev_iterator, target_test_iterator = get_dataloaders(target_args)

            for _exp_file_path in _opt_dir_list:
                _exp_file_path = Path(_exp_file_path)
                print(_exp_file_path)

                # Load model
                _inDomain_model = load_inDomain_model(_exp_file_path, args.device)

                target_test_scores = get_evaluation_scores(
                    model = _inDomain_model, 
                    iterator = target_test_iterator, 
                    args = target_args)

                target_dev_scores = get_evaluation_scores(
                    model = _inDomain_model, 
                    iterator = target_dev_iterator, 
                    args = target_args)
                
                _dev_target_domain_performances.append(target_dev_scores[args.performance_metric])
                _dev_target_domain_fairness.append(1-target_dev_scores[args.fairness_metric])

                _test_target_domain_performances.append(target_test_scores[args.performance_metric])
                _test_target_domain_fairness.append(1-target_test_scores[args.fairness_metric])

            # Calculate statistics
            dev_target_domain_performances_mean.append(
                statistics.mean(_dev_target_domain_performances))
            dev_target_domain_fairness_mean.append(
                statistics.mean(_dev_target_domain_fairness))
            test_target_domain_performances_mean.append(
                statistics.mean(_test_target_domain_performances))
            test_target_domain_fairness_mean.append(
                statistics.mean(_test_target_domain_fairness))

            try:
                dev_target_domain_performances_std.append(
                    statistics.stdev(_dev_target_domain_performances))
                dev_target_domain_fairness_std.append(
                    statistics.stdev(_dev_target_domain_fairness))
                test_target_domain_performances_std.append(
                    statistics.stdev(_test_target_domain_performances))
                test_target_domain_fairness_std.append(
                    statistics.stdev(_test_target_domain_fairness))
            except:
                dev_target_domain_performances_std.append(np.nan)
                dev_target_domain_fairness_std.append(np.nan)
                test_target_domain_performances_std.append(np.nan)
                test_target_domain_fairness_std.append(np.nan)

        InDomain_Method_dfs[_split]["target test_performance mean"] = test_target_domain_performances_mean
        InDomain_Method_dfs[_split]["target test_performance std"] = test_target_domain_performances_std
        InDomain_Method_dfs[_split]["target test_fairness mean"] = test_target_domain_fairness_mean
        InDomain_Method_dfs[_split]["target test_fairness std"] = test_target_domain_fairness_std
        InDomain_Method_dfs[_split]["target dev_performance mean"] = dev_target_domain_performances_mean
        InDomain_Method_dfs[_split]["target dev_performance std"] = dev_target_domain_performances_std
        InDomain_Method_dfs[_split]["target dev_fairness mean"] = dev_target_domain_fairness_mean
        InDomain_Method_dfs[_split]["target dev_fairness std"] = dev_target_domain_fairness_std

        InDomain_Method_dfs[_split].to_pickle(
            Path(args.save_dir) / "{}_{}_{}_{}.pkl".format(args.source_dataset, _split, args.target_dataset, args.method)
            )