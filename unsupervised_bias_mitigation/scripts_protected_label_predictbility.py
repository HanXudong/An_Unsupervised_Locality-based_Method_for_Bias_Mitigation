import os
import argparse
from pathlib import Path
import pandas as pd

from fairlib.src import analysis
from fairlib.src.base_options import BaseOptions
from fairlib.src import networks
from fairlib.src.dataloaders import get_dataloaders, default_dataset_roots
from fairlib.src.evaluators.evaluator import gap_eval_scores
from fairlib.src.networks.knn_labels import KNN

import tarfile
import yaml
from yaml.loader import SafeLoader
import torch
import statistics
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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

# evaluate KNN's accuracy
def eval_KNN_epoch(model, iterator, device, p, k, include_self = False, class_wise = True):    
    model.eval()

    preds = []
    labels = []
    private_labels = []

    for batch in iterator:

        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        y_item = list(set(tags.tolist()))
        y_mask = {}
        for tmp_y in y_item:
            y_mask[tmp_y] = (tags == tmp_y)

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).float()

        hs = model.hidden(text)

        knn_labels = torch.zeros_like(tags)


        if class_wise:
            for temp_y in y_item:
                temp_y_masks = y_mask[temp_y]
                class_hs = hs[temp_y_masks]

                k_indices = KNN(class_hs, class_hs, p, k, include_self = include_self)

                temp_y_knn_labels = p_tags[temp_y_masks][k_indices].mode(1).values.long()

                knn_labels[temp_y_masks.nonzero().squeeze()] = temp_y_knn_labels
        else:
            k_indices = KNN(hs, hs, p, k, include_self = include_self)
            knn_labels = p_tags[k_indices].mode(1).values.long()

        labels += list(tags)
        private_labels += list(batch[2].cpu().numpy())
        preds += list(knn_labels.detach().cpu().numpy())
    
    return (preds, labels, private_labels)

if __name__ == '__main__':
    """Evaluation cross domain bias mitigation
    
    E.g., python .\cross_domain_analysis.py --source_dataset Bios_gender --method EOCla --target_dataset Bios_economy
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--file_path', type=str, default="bs1024")
    parser.add_argument('--device_id', type=int, default=0, help='device id, -1 is cpu')
    parser.add_argument('--save_dir', type=str, default="knn_leakage")

    args = parser.parse_args()

    if args.device_id < 0:
        args.device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.device_id)
        args.device = torch.device("cuda:{}".format(args.device_id))

    # Get in-domain pareto frontiers
    PREPROCESSED_FILES_DIR = Path(args.file_path)
    df = pd.read_pickle(PREPROCESSED_FILES_DIR / "{}_{}.pkl".format(args.source_dataset, args.method))
    splits = ["dev", "test"]
    InDomain_Method_dfs = {}
    for _split in splits:
        InDomain_Method_dfs[_split] = analysis.final_results_df(
            results_dict = {args.method:df},
            pareto = True, pareto_selection = _split,
            selection_criterion = None, return_dev = True,
            )

        knn_results_dir_list = []
        # Iterative over each Pareto model and evaluate its cross-domain results
        for _model_id, _opt_dir_list in enumerate(list(InDomain_Method_dfs[_split]["opt_dir list"])):
            
            _model_id_results = []
            
            target_args = target_dataset_args()
            target_args.dataset = args.source_dataset
            target_args.data_dir = default_dataset_roots[target_args.dataset]
            target_args.device = args.device

            for _repeat_id, _exp_file_path in enumerate(_opt_dir_list):
                _exp_file_path = Path(_exp_file_path)

                # Load model
                _inDomain_model = load_inDomain_model(_exp_file_path, args.device)

                for _batch_size in [128, 256, 512, 1024]:
                    # Vary batch size
                    target_args.batch_size = _batch_size
                    _target_train_iterator, _, _ = get_dataloaders(target_args)

                    for _include_self in [True, False]:
                        for _class_wise in [True, False]:
                            for _p in [2,4,6,8]:
                                for _k in range(1,16,2):
                                    try:
                                        _knn_preds, _labels, _private_labels = eval_KNN_epoch(
                                            model = _inDomain_model, 
                                            iterator = _target_train_iterator,
                                            device = args.device, 
                                            p = _p, 
                                            k = _k,
                                            include_self = _include_self,
                                            class_wise=_class_wise)
                                        _acc = accuracy_score(_private_labels, _knn_preds)
                                        _fscore = f1_score(_private_labels, _knn_preds)
                                    except:
                                        _acc = np.nan
                                        _fscore = np.nan
                                    
                                    _model_id_results.append(
                                        {
                                            "repeat_id":_repeat_id,
                                            "include_self":_include_self,
                                            "class_wise":_class_wise,
                                            "p":_p,
                                            "k":_k,
                                            "acc":_acc,
                                            "fscore":_fscore,
                                        }
                                    )
            _model_id_results_df = pd.DataFrame(_model_id_results)
            _fname_model_id_results_df = "KNNScores_{}_{}_{}_{}.pkl".format(args.source_dataset, _split, args.method, _model_id)

            _model_id_results_df.to_pickle(
                Path(args.save_dir) / _fname_model_id_results_df
            )
            knn_results_dir_list.append(_fname_model_id_results_df)

        InDomain_Method_dfs[_split]["knn_results_dir_list"] = knn_results_dir_list

        InDomain_Method_dfs[_split].to_pickle(
            Path(args.save_dir) / "{}_{}_{}.pkl".format(args.source_dataset, _split, args.method)
            )