import numpy as np
from typing import Dict, Tuple
import secrets
import sys
# import cupy as cp

# init function


# generate a random integer sequence A_i for each client, A_i's length is num_group



# define a init_seed_pair function, input is num_users, generate seed pair s_{i,j} between client i and client j
# seed pair s_{i,j} is the input of pseudo-random function PRG, where s_{i,j} is the seed pair between client i and client j, and s_{i,j} = s_{j,i}
def init_seed_pair(num_users: int) -> np.ndarray:
    """generate shared seed for each pair of clients. use two-dimensional array to store, seed_pairs[i][j] represents the shared seed between client i and client j.
    
    Args:
        num_users (int): number of clients
        
    Returns:
        np.ndarray: two-dimensional array, store the shared seed between each pair of clients
    """
    # set seed length
    seed_length = 16
    
    # initialize two-dimensional array
    seed_pairs = np.zeros((num_users, num_users), dtype=np.int64)
    
    # generate shared seed for each pair of clients
    for i in range(num_users):
        for j in range(i + 1, num_users):
            # use cryptographically secure random number generator
            seed = secrets.randbits(seed_length)
            # store seed pair, keep symmetry
            seed_pairs[i][j] = seed
            seed_pairs[j][i] = seed
    
    return seed_pairs

# generate a random integer sequence A_i for each client, A_i's length is num_group
def init_A(num_users: int, num_groups: int) -> Dict[int, np.ndarray]:
    """generate a random integer sequence A_i for each client, A_i's length is num_group. range is (0,1], and can be repeated
    
    Args:
        num_users (int): number of clients
        num_groups (int): number of groups
        
    Returns:
        Dict[int, np.ndarray]: dictionary of random integer sequence for each client, key is client index, value is random integer sequence
    """
    A_dict: Dict[int, np.ndarray] = {}
    
    for i in range(num_users):
        # generate num_groups random numbers between (0,1], allow repetition
        A_dict[i] = np.random.uniform(0.0001, 5, size=num_groups)  # use 0.0001 as minimum value
    
    return A_dict

def init_R(num_group: int, num_param_in_group: int):
    """initialize R, all clients share one R
    Args:
        num_group: number of groups
        num_param_in_group: number of elements in each group, M+1
    Returns:
        R: random integer sequence, shape is (num_group, num_param_in_group)
    """
    # directly generate two-dimensional array
    R = np.zeros((num_group, num_param_in_group))
    for i in range(num_group):
        # generate num_param_in_group non-repeating values (positive numbers) for each group
        R[i] = np.random.choice(np.arange(1, 2*num_param_in_group + 1), size=num_param_in_group, replace=False)
    return R
