import sys
import yaml

from.loaders import TestDataset, SampleDataset
from .Moji import DeepMojiDataset
from .Bios import BiosDataset
from .Valence import ValenceDataset
from .FCL_BiosDataset import FCL_BiosDataset
from .Trustpilot import TrustpilotDataset
from .Adult import AdultDataset
from .COMPAS import COMPASDataset
from .imSitu import imSituDataset
from .ColoredMNIST import MNISTDataset



data_dir = r'data'
default_dataset_roots = dict(
    Moji= data_dir + '/moji',
    Bios_gender= data_dir + '/bios',
    Bios_economy= data_dir + '/bios',
    Bios_both= data_dir + '/bios',
    Bios_intersection= data_dir + '/bios',
    FCL_Bios= data_dir + '/bios',
    Trustpilot_gender= data_dir + '/trustpilot',
    Trustpilot_age= data_dir + '/trustpilot',
    Trustpilot_country= data_dir + '/trustpilot',
    Trustpilot_intersection= data_dir + '/trustpilot',
    Adult_gender= data_dir + '/adult',
    Adult_race= data_dir + '/adult',
    Adult_intersection= data_dir + '/adult',
    COMPAS_gender= data_dir + '/compas',
    COMPAS_race= data_dir + '/compas',
    COMPAS_intersection= data_dir + '/compas',
)


loader_map = {
    "moji":DeepMojiDataset,
    "bios":BiosDataset,
    "test":TestDataset,
    "sample":SampleDataset,
    "valence":ValenceDataset,
    "fclbios":FCL_BiosDataset,
    "trustpilot":TrustpilotDataset,
    "adult":AdultDataset,
    "compas":COMPASDataset,
    "imsitu":imSituDataset,
    "mnist":MNISTDataset,
}

def name2loader(args):
    dataset_name = args.dataset.split("_")[0].lower()

    if  len(args.dataset.split("_")) > 1:
        args.protected_task = args.dataset.split("_")[1]

    # Load default hyperparameters
    try:
        with open("dataset_specific_hyperparameters.yaml", 'r') as f:
            dataset_specific_hyperparameters = yaml.full_load(f)
    except:
        dataset_specific_hyperparameters = {}
    print(dataset_specific_hyperparameters)

    if args.dataset in dataset_specific_hyperparameters.keys():
        dataset_specific_hyperparameters = dataset_specific_hyperparameters[args.dataset]
        if args.emb_size is None:
            args.emb_size = dataset_specific_hyperparameters["emb_size"]
        if args.num_classes is None:
            args.num_classes = dataset_specific_hyperparameters["num_classes"]
        if args.num_groups is None:
            args.num_groups = dataset_specific_hyperparameters["num_groups"]

    if dataset_name in loader_map.keys():
        return loader_map[dataset_name]
    else:
        raise NotImplementedError