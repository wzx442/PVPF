
import torch
from typing import OrderedDict, Dict, List



def agg_server_weights(
    B: Dict[int, List[float]], # B[i][k] represents the coefficient list of the k-th group for the i-th client
    server_accumulate_mask: Dict[str, torch.Tensor], # server_accumulate_mask[k] represents the mask of the k-th group
    num_users: int,
    num_groups: int,
) -> List[float]:
    """Aggregates server weights by group. 

    Args:
        B: Dictionary of client coefficients, where B[i] is the coefficient list for client i
        num_users: Number of users

    Returns:
        List of aggregated coefficients 
    """
    
    if not B:
        print("B is empty")
        return []
    
   
    agg_coeffs = {}
    for k in range(num_groups):
        temp_coeffs = [0.0] * (len(B[0][k]))  # Initialize with zeros, length M+1
        for i in range(num_users):
            for j in range(len(B[i][k])):
                temp_coeffs[j] += B[i][k][j]
        agg_coeffs[k] = temp_coeffs

    return agg_coeffs # agg_coeffs[k] represents the aggregated coefficient of the k-th group   


