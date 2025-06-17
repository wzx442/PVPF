import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, OrderedDict, Tuple, Optional, Any

# Custom Libraries
from utils.options import lth_args_parser
from utils.train_utils import prepare_dataloaders, get_data
from utils.train_functions import evaluate, train_personalized
from pflopt.optimizers import MaskLocalAltSGD, local_alt
from lottery_ticket import init_mask_zeros, delta_update
from broadcast import (
    broadcast_server_to_client_initialization,
    div_server_weights,
    add_masks,
    add_server_weights,
)
import random
from torchvision.models import resnet18
from torchvision import models
from models import load_model as load_model_from_models
from fedselect import fedselect_algorithm, cross_client_eval
from utils.log_utils import setup_logger, restore_stdout

# My Library
from Enc_and_Dec.init import init_seed_pair


def get_cross_correlation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Get cross correlation between two tensors using F.conv2d.
    Args:
        A: First tensor
        B: Second tensor

    Returns:
        torch.Tensor: Cross correlation result
    """
    # Normalize A
    A = A.cuda() if torch.cuda.is_available() else A 
    B = B.cuda() if torch.cuda.is_available() else B 
    A = A.unsqueeze(0).unsqueeze(0) 
    B = B.unsqueeze(0).unsqueeze(0) 
    A = A / (A.max() - A.min()) if A.max() - A.min() != 0 else A 
    B = B / (B.max() - B.min()) if B.max() - B.min() != 0 else B 
    return F.conv2d(A, B) 


def run_base_experiment(model: nn.Module, args: Any, seed_pairs: List[Tuple[int, int]]) -> None:
    """Run base federated learning experiment.
    Args:
        model: Neural network model 
        args: Experiment arguments 
    """
    dataset_train, dataset_test, dict_users_train, dict_users_test, labels = get_data(args) 
    idxs_users = np.arange(args.num_users * args.frac) 
    m = max(int(args.frac * args.num_users), 1) 
    idxs_users = np.random.choice(range(args.num_users), m, replace=False) 
    idxs_users = [int(i) for i in idxs_users] 
    fedselect_algorithm(
        model, 
        args, 
        dataset_train, 
        dataset_test, 
        dict_users_train, 
        dict_users_test, 
        labels, 
        idxs_users, 
        seed_pairs, 
    )


def load_model(args: Any) -> nn.Module:
    """Load and initialize model.
    Args:
        args: Model arguments 

    Returns:
        nn.Module: Initialized model 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = load_model_from_models(
        model_name=args.model if hasattr(args, 'model') else 'resnet18',
        num_classes=args.num_classes,
        dataset=args.dataset if hasattr(args, 'dataset') else 'cifar10',
        device=device
    )
    
    return model


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    Args:
        seed: Random seed value 
    """
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 



if __name__ == "__main__":
    args = lth_args_parser()
    seed_pairs = init_seed_pair(args.num_users)
    print(f"seed_pairs: {seed_pairs}")

    setup_seed(args.seed)
    model = load_model(args)

    run_base_experiment(model, args, seed_pairs)
