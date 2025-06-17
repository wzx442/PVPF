import torch
import time
from typing import OrderedDict, Dict, List
import numpy as np
from .large import evaluate_polynomial

def Dec(
    agg_B: List[float], # agg_B[k] represents the aggregated coefficient of the k-th group
    client_mask: OrderedDict[str, torch.Tensor], # client_mask[i] represents the mask of the i-th client
    A_dict: Dict[int, np.ndarray], # A_dict[i][k] represents the random integer sequence of the i-th client for the k-th group
    R: List[np.ndarray], # R[k] represents the random integer sequence of the k-th group
    M: int, # M represents the number of elements in each group
    num_group: int, # num_group represents the number of groups
    num_user: int, # num_user represents the number of users
    client_initialization: OrderedDict[str, torch.Tensor], # client_initialization[i] represents the initialization of the i-th client
) -> OrderedDict[str, torch.Tensor]:
    """Decrypts aggregated ciphertext using Lagrange interpolation. 

    Args:
        agg_B: List of aggregated coefficients 
        client_mask: Client mask indicating which parameters are local (1) vs global (0) 
        A_dict: Dictionary of random integer sequences for each client 
        R: Random integer sequence shared by all clients 
        M: Number of elements in each group 
        num_group: Number of groups 
        num_user: Number of users 
        client_initialization: OrderedDict[str, torch.Tensor] 
    Returns:
        Decrypted server weights 
    """
    
    poly_value = []
    for group_idx in range(num_group):
        group_coeffs = agg_B[group_idx] # agg_B[k] represents the aggregated coefficient of the k-th group
        
       
        for j in range(M):
           
            x = R[group_idx][j]
            value = evaluate_polynomial(group_coeffs, x)
            poly_value.append(value)
            
       
        if judge(group_coeffs, R[group_idx][M], num_user, A_dict, group_idx):
            pass
        else:
            print(f"group {group_idx} verification failed")


    client_initialization = DEC_broadcast_server_to_client_initialization(
        poly_value, 
        client_mask, 
        client_initialization, 
        num_user,
        )
            
    return client_initialization


def judge(
    coeffs: list[float], # coeffs[k] represents the aggregated coefficient of the k-th group    
    x: int, # x
    num_user: int, # num_user
    A_dict: Dict[int, np.ndarray], # A_dict[i][k] represents the random integer sequence of the i-th client for the k-th group
    k: int, # k
): 
    """
    """
    sum_A = 0
    for i in range(num_user): 
        sum_A += A_dict[i][k]
    # print(f"sum_A: {sum_A}")

    # calculate f_{[group_idx]}(x)
    f_x = evaluate_polynomial(coeffs, x)

    # verify, allow ± 1e-3 error range
    if abs(f_x - sum_A) <= 1e-3:
        return True
    else:
        print(f"{f_x} != {sum_A}")
        return False
    

def DEC_broadcast_server_to_client_initialization(
    poly_value: list, # poly_value[k] represents the aggregated coefficient of the k-th group
    mask: OrderedDict[str, torch.Tensor], # client_mask[i] represents the mask of the i-th client
    client_initialization: OrderedDict[str, torch.Tensor], # client_initialization[i] represents the initialization of the i-th client
    num_user,
) -> OrderedDict[str, torch.Tensor]:
    """broadcast server weights to client initialization, only update the part of the mask is 0"""
    
    # convert poly_value to numpy array
    flat_array = np.array(poly_value, dtype=np.float32)
     
    # track the current index position
    index = 0
    
    # assign value to each parameter
    for key, param in client_initialization.items():
        if "weight" in key or "bias" in key:
            # calculate the number of elements of the current parameter
            num_elements = param.numel()
            
            # extract the corresponding value from flat_array
            param_values = flat_array[index:index + num_elements]
        
            # reshape to the shape of the parameter
            param_values = param_values.reshape(param.shape)
            
            # convert to torch tensor and only update the part of the mask is 0
            param_tensor = torch.tensor(param_values, dtype=param.dtype, device=param.device)
            client_initialization[key][mask[key] == 0] = param_tensor[mask[key] == 0]  
            # divide
            index += num_elements
    
    
    return client_initialization







