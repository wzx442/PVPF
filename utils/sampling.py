import random
import numpy as np
import torch
from typing import Dict, List, Set, Union, Tuple
from torch.utils.data import Dataset


def iid(dataset: Dataset, num_users: int) -> Dict[int, Set[int]]:

    num_items = len(dataset)
    dict_users = {i: set() for i in range(num_users)} 
    all_idxs = np.arange(num_items) 

    # shuffle all indices
    np.random.shuffle(all_idxs)

    # use numpy.array_split to split the shuffled indices as evenly as possible into num_users parts
    # this function will automatically handle the case where it cannot be divided evenly
    idx_chunks = np.array_split(all_idxs, num_users)

    # assign the split indices to each user
    for i in range(num_users):
        dict_users[i] = set(idx_chunks[i].tolist()) # convert numpy array to list, then to set

    # verify that all indices are assigned and no duplicates
    all_assigned_idxs = set().union(*dict_users.values())
    assert len(all_assigned_idxs) == num_items, "Not all indices were assigned!"
    assert len(set(all_idxs)) == num_items, "Original indices count mismatch!" # Ensure original calculation is correct

    # verify that each client received data (unless the number of users is greater than the number of data points)
    if num_users <= num_items:
        for i in range(num_users):
            assert len(dict_users[i]) > 0, f"Client {i} received no data!"
    else:
         print(f"Warning: Number of users ({num_users}) > number of data points ({num_items}). Some clients will get no data.")


    return dict_users
    """
    Sample I.I.D. client data from dataset, ensuring ALL data points are distributed. 

    Args:
        dataset: The full dataset to sample from. 
        num_users: Number of clients to divide data between. 

    Returns:
        Dict mapping client IDs to sets of data indices assigned to that client. 
        Client data sizes may
    """
    # num_items_total = len(dataset)
    # all_idxs = np.arange(num_items_total) # Get all indices as a numpy array 
    # np.random.shuffle(all_idxs) # Shuffle all indices randomly in place 

    # # Use np.array_split to divide the indices into num_users chunks. 
    # # This function handles non-even splits automatically and distributes 
    # # the remainder items across the first few chunks.
    # user_idx_chunks = np.array_split(all_idxs, num_users) 

    # dict_users = {}
    # for i in range(num_users):
    #     # np.array_split returns numpy arrays, convert them to sets of integers
    #     dict_users[i] = set(user_idx_chunks[i].tolist())

    # # Optional: Verification step to ensure all indices are assigned exactly once
    # total_assigned = sum(len(s) for s in dict_users.values())
    # assert total_assigned == num_items_total, f"Error: Assigned {total_assigned} indices, but dataset has {num_items_total}"
    # all_assigned_indices = set().union(*dict_users.values())
    # assert len(all_assigned_indices) == num_items_total, "Error: Not all indices were assigned uniquely or some were missed."

    # return dict_users


def noniid(
    dataset: Dataset,
    num_users: int,
    shard_per_user: int,
    server_data_ratio: float = 0.0,
    size: Union[int, None] = None,
    rand_set_all: List = [],
) -> Tuple[Dict[Union[int, str], Union[np.ndarray, Set[int]]], np.ndarray]:
    """Sample non-I.I.D client data from dataset by dividing data by class labels.   

    Args:
        dataset: The full dataset to sample from 
        num_users: Number of clients to divide data between 
        shard_per_user: Number of class shards to assign to each user 
        server_data_ratio: Fraction of data to reserve for server (default: 0.0) 
        size: Optional size to limit each user's data to 
        rand_set_all: Optional pre-defined random class assignments 

    Returns:
        Tuple containing:
            - Dict mapping client IDs to arrays of assigned data indices 
    """
    if shard_per_user == 0:
        return iid(dataset, num_users), np.array([])

    dict_users, all_idxs = {i: np.array([], dtype="int64") for i in range(num_users)}, [
        i for i in range(len(dataset))
    ]

    targets = None
    targets = []
    for elem in dataset:
        if isinstance(elem[1], torch.Tensor):
            targets.append(elem[1].item())
        else:
            targets.append(elem[1])

    # dictionary of indices in the dataset for each label
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(targets)[value])
        assert (len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(dataset)
    assert len(set(list(test))) == len(dataset)

    if server_data_ratio > 0.0:
        dict_users["server"] = set(
            np.random.choice(
                all_idxs, int(len(dataset) * server_data_ratio), replace=False
            )
        )

    for i in range(num_users):
        num_elem = len(dict_users[i])
        dict_users[i] = np.concatenate(
            [
                dict_users[i][k : k + size]
                for k in range(0, num_elem, num_elem // shard_per_user + 1)
            ]
        )

    return dict_users, rand_set_all
