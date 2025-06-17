import torch
from typing import OrderedDict, List, Tuple, Dict
import numpy as np
import time


from scipy.interpolate import lagrange
from Enc_and_Dec.large import lagrange_interpolation

def Enc(
        client_weights: OrderedDict[str, torch.Tensor],
        client_mask: OrderedDict[str, torch.Tensor],   # client_mask  0 represents global parameter, 1 represents local parameter
        seed_pairs: List[Tuple[int, int]],
        A_dict: Dict[int, np.ndarray], 
        R: np.ndarray,
        client_id: int = 0,  # client_id represents the id of the current client
        M: int = 10,  # M represents the number of elements in each group
        num_groups: int = 10,  # num_groups represents the number of groups
        num_users: int=10,   # num_users represents the number of users
        ) -> List[np.ndarray]:
    """encrypt function, encrypt the model parameters of the current client
    
    Args:
        client_weights: client model state dict
        client_mask: client mask, 0 represents global parameter, 1 represents local parameter
        seed_pairs: seed pairs between clients
        A_dict: random sequence dictionary of each client
        R: random sequence shared by all clients
        client_id: id of the current client
        M: number of elements in each group
        num_groups: number of groups
        num_users: number of users

    Returns:
        ciphertext: encrypted coefficient list, length is num_groups, each element is a np.ndarray, length is M+1
        
    Description:
        1. first, extract the parameters to be uploaded according to the mask(global parameter, mask=0)
        2. encrypt the parameters
    """
    
    
    
    # extract global parameter (mask=0 represents global parameter)
    global_param = None
    for key in client_weights:
        if "weight" in key or "bias" in key:
            # extract the parameters and mask of the current layer
            param = client_weights[key]
            mask = client_mask[key]
            
            # only process the global parameter (mask=0)
            layer_global_param = param * (1-mask)
            
            # if global_param is None, then initialize it
            if global_param is None:
                global_param = layer_global_param
            else:
                # otherwise, concatenate the global parameter of the current layer to the existing global parameter
                global_param = torch.cat([global_param.flatten(), layer_global_param.flatten()])
    
    # add debug information
    print("Global param shape:", global_param.shape)
            
    """group, first, divide global_param into num_groups groups, each group contains M elements, if the last group is less than M elements, then use 0 to fill"""
    
    # flatten and convert to list
    global_param_list = global_param.flatten().tolist()
    print("global_param_list length:", len(global_param_list))

    """ add actual encryption logic, use seed_pairs as input of pseudo-random function: encrypted_param = global_param_list + \\sum_{client_id < j} PRG(seed_pairs[client_id,j]) - \\sum_{client_id > j} PRG(seed_pairs[client_id,j]) , PRG is pseudo-random number generator, seed_pairs[client_id,j] is seed pair, PRG output is a random number sequence of length global_param_list.length
    """

    # initialize encrypted parameter as a copy of the original parameter
    encrypted_param = global_param_list.copy()
    param_length = len(global_param_list)
    
    # process each user j
    ################################################################
    for j in range(num_users):
        if j == client_id:  # skip itself
            continue
        if client_id < j:
            # use seed pair as seed of random number generator
            seed = hash((seed_pairs[client_id, j]))  # use tuple as hash input
            np.random.seed(seed)
            # generate random number sequence of the same length as the parameter list
            random_values = np.random.rand(param_length)
            # add random number to encrypted parameter
            for idx in range(param_length):
                encrypted_param[idx] += random_values[idx]
        else:  # client_id > j
            # use seed pair as seed of random number generator
            seed = hash((seed_pairs[j, client_id]))  # use tuple as hash input
            np.random.seed(seed)
            # generate random number sequence of the same length as the parameter list
            random_values = np.random.rand(param_length)
            # subtract random number from encrypted parameter
            for idx in range(param_length):
                encrypted_param[idx] -= random_values[idx]
    #############################################################################
    # then divide encrypted_param into num_groups groups, each group contains M elements, if the last group is less than M elements, then use 0 to fill
    # num_groups is given in the parameter
    # create a new list to store the grouped parameters
    encrypted_param_grouped = []
    
    # calculate the total number of elements needed
    total_elements = num_groups * M
    
    # if encrypted_param is less than the total number of elements, use 0 to fill
    if len(encrypted_param) < total_elements:
        encrypted_param.extend([0] * (total_elements - len(encrypted_param)))
    print("encrypted_param length:", len(encrypted_param))
    
    # group
    for i in range(num_groups):
        start_idx = i * M
        end_idx = (i + 1) * M
        group = encrypted_param[start_idx:end_idx]
        encrypted_param_grouped.append(group)
    """encrypt, encrypt each group, encrypt method is Lagrange interpolation polynomial fitting"""        
    ciphertext = np.zeros((num_groups, M+1))
    for k in range(num_groups):
        # get x and y points
        x_points = R[k]  # directly
        y_points = np.array(encrypted_param_grouped[k])  # convert list to numpy array
        
        # use np.append to add extra elements
        y_points = np.append(y_points, A_dict[client_id][k])  # add A_dict[client_id][k] element
        ciphertext[k] = lagrange_interpolation(x_points, y_points) # k represents the coefficient of the k-th group
    """each time the loop adds an coefficients array to ciphertext
    coefficients is the result of np.polyfit, is a one-dimensional numpy array, length is M+1(from highest power to constant term)
    therefore, the final structure is:
    ciphertext is a Python list, length is num_groups
    each element of ciphertext is a numpy array, length is M+1
    that is, ciphertext is a two-dimensional structure of shape [num_groups, M+1].

    from specific representation:
    # assume num_groups = 3, M = 2
    ciphertext = [
        array([a1, b1, c1]),  # the polynomial coefficient of the 1st group
        array([a2, b2, c2]),  # the polynomial coefficient of the 2nd group
        array([a3, b3, c3])   # the polynomial coefficient of the 3rd group
    ]
    """
    print("encryption completed")
    return ciphertext # ciphertext[i] represents the coefficient of the i-th group

