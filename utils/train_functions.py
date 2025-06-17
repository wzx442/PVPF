import torch
import torch.nn as nn
from typing import OrderedDict, Tuple, Optional, Any
from tqdm import tqdm
from pflopt.optimizers import MaskLocalAltSGD, local_alt

def evaluate(
    model: nn.Module, ldr_test: torch.utils.data.DataLoader, args: Any
) -> float:
    """Evaluate model accuracy on test data loader.

    Args:
        model: Neural network model to evaluate
        ldr_test: Test data loader
        args: Arguments containing device info


    Returns:
        float: Average accuracy on test set 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_accuracy = 0
    # set model to evaluation mode
    # 1. disable dropout layer
    # 2. use running statistics of batch normalization layer instead of calculating new statistics
    # 3. do not calculate gradients, save memory
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ldr_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
            average_accuracy += acc
        average_accuracy /= len(ldr_test)
    return average_accuracy


def train_personalized(
    model: nn.Module,   
    ldr_train: torch.utils.data.DataLoader, 
    mask: OrderedDict, 
    args: Any, 
    initialization: Optional[OrderedDict] = None, 
    verbose: bool = False, 
    eval: bool = True, 
) -> Tuple[nn.Module, float]:
    """Train model with personalized local alternating optimization. 

    Args:
        model: Neural network model to train               
        ldr_train: Training data loader                    
        mask: Binary mask for parameters                   
        args: Training arguments                           
        initialization: Optional initial model state       
        verbose: Whether to print training progress        
        eval: Whether to evaluate during training          

    Returns:
        Tuple containing:           
            - Trained model         
            - Final training loss   
    """
    if initialization is not None:
        model.load_state_dict(initialization)
    optimizer = MaskLocalAltSGD(model.parameters(), mask, lr=args.lr)
    epochs = args.la_epochs # rounds of training for local alt optimization 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss() 
    train_loss = 0  
    with tqdm(total=epochs) as pbar: 
        for i in range(epochs): 
            train_loss = local_alt(
                model,
                criterion,
                optimizer,
                ldr_train,
                device,
                clip_grad_norm=args.clipgradnorm, 
            )
            if verbose: 
                print(f"Epoch: {i} \tLoss: {train_loss}")
            pbar.update(1) 
            pbar.set_postfix({"Loss": train_loss}) 
    return model, train_loss 