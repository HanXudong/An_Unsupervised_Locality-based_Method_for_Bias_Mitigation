# An_Unsupervised_Locality-based_Method_for_Bias_Mitigation
Source codes for ICLR 2023 paper "Everybody Needs Good Neighbours: An Unsupervised Locality-based Method for Bias Mitigation"

If you use the code, please cite the following paper:

```
@inproceedings{han2023everybody,
title={Everybody Needs Good Neighbours: An Unsupervised Locality-based Method for Bias Mitigation},
author={Xudong Han and Timothy Baldwin and Trevor Cohn},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=pOnhudsvzR}
}
```

All experiments are conducted with the open source fairlib library, which includes all baseline methods in this paper. 
Please check the homepage of fairlib for more details.


Disaggregated plots are shown in `disaggregated_results.html`, and full size figures are stored in `full_results_figures/`.
The code to do the analysis can be found form `unsupervised_bias_mitigation\NB_Appendix_indomain_tradeoffs_dispaly.ipynb`.

Here we introduce the key modifications to fairlib and the steps for reproducing reported results in this paper.

## Modifications

### 1. Proxy label assignment and correction


**file path**: `fairlib\src\networks\knn_labels.py`  

```python
KNN_labels(criterion, tags, text, model, predictions, loss, p = 2, k = 5, average_first = False, include_self = True)
    """Derive proxy labels with NN correction

    Args:
        criterion (function): loss function
        tags (torch.tensor): target labels
        text (inputs): inputs
        model (torch.module): target model
        predictions (torch.tensor): model predictions
        loss (torch.tensor): average loss
        p (int, optional): norm. Defaults to 2.
        k (int, optional): number of NN. Defaults to 5.
        average_first (bool, optional): voting after average aggregation. Defaults to False.
        include_self (bool, optional): if the query instance itself is considered as a NN. Defaults to True.

    Returns:
        proxy label (torch.tensor): proxy label assignment with correction
    """
```

### 2. RL implementation

**file path**: `fairlib\src\networks\knn_labels.py`  

```
KNN_Loss(torch.nn.Module)
```

This module returns reweighted loss as discussed in the paper.

### 3. Replacing true protected labels with proxy labels

**file path:** `fairlib\src\networks\utils.py`, line 61 ~ 69.

```python
# Simulating fairness without demographics in using KNN based labels
if args.knn_labels:
    # Derive proxy labels with correction
    p_tags = KNN_labels(
        criterion = criterion, 
        tags = tags if not args.regression else regression_tags, 
        predictions = predictions, 
        text = text, 
        model = model, 
        loss = loss, 
        p = args.knn_labels_p, 
        k = args.knn_labels_k)
    
    # replace protected labels with proxy labels
    batch = batch.copy()
    batch[2] = p_tags
```

## Reproducing numbers

### Step 1: Install packages

```bash
# Start a new virtual environment:
conda create -n debiasing_py38 python=3.8
conda activate debiasing_py38

python setup.py develop
```

### Step 2: Prepare datasets

fairlib provides simple APIs to access fairness benchmark datasets that are publicly available and under strict ethical guidelines. 

For example, 
```python
from fairlib import datasets

datasets.prepare_dataset("moji", "data/deepmoji")
```

Five benchmark datasets are used in this paper, please download four of them as follows:

```python
from fairlib import datasets

datasets.prepare_dataset("moji", "data/moji")
datasets.prepare_dataset("bios", "data/bios")
datasets.prepare_dataset("adult", "data/adult")
datasets.prepare_dataset("compas", "data/compas")
```

In terms of the TrustPilot dataset, please follow the instruction from https://github.com/lrank/Robust_and_Privacy_preserving_Text_Representations

### Step 3: Run experiments

Taking proxy label based EO_CLA as an example (`--DyBT GroupDifference --DyBTObj EO`) over the moji dataset (`--dataset Moji`), the following code train a model using proxy labels (`--knn_labels`) with correction based on 5 nearest neighbors(`--knn_labels_k 5`), where distances are measured with Euclidean distance (--knn_labels_p 2). The strength of group difference loss is `--DyBTalpha 0.00630957344480193`. 

```bash
python fairlib --exp_id UEOCla_0_Moji_1_2_0_0 --epochs_since_improvement 10 --epochs 100 --results_dir /results --knn_labels --knn_labels_k 5 --knn_labels_p 2 --DyBT GroupDifference --DyBTObj EO --DyBTalpha 0.00630957344480193 --log_interval 5 --save_batch_results --dataset Moji --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --base_seed 6844597 --project_dir Vanilla --emb_size 2304 --num_classes 2 --num_groups 2
```

check the `fairlib\src\base_options.py` for more details of each hyperparameter.

### Step 4: Main results


**file path:** `unsupervised_bias_mitigation\scripts_get_results.py`

Section 4 EXPERIMENTAL RESULTS, and 5.3 DEBIASING FOR INTERSECTIONAL GROUPS, and 5.4 OTHER FAIRNESS METRICS, DP

After running experiments, scripts_get_results.py is able to collect results and save them into pickle files.

Based on the collected results, please follow the instruction of fairlib to derive tables.

### Step 5: Analysis
- Section 5.1 PROXY LABEL ASSIGNMENT  
  NB_Analysis_KNN_labels.ipynb: 
- Section 5.2 EFFECTIVENESS OF THE KNN CORRECTION  
  scripts_protected_label_predictbility.py: 
- Appendix C.3 COMPUTATIONAL BUDGET  
  scripts_computational_budgets.py
- Appendix E.5 HOW DOES BIAS MITIGATION AFFECT FAIRNESS FOR UNOBSERVED GROUPS?  
    scripts_cross_domain_analysis.py