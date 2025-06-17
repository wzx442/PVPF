import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, OrderedDict, Tuple, Optional, Any
import time  # Import the time module for timing
# Custom Libraries
from utils.train_utils import prepare_dataloaders
from utils.train_functions import evaluate, train_personalized
from pflopt.optimizers import MaskLocalAltSGD, local_alt
from lottery_ticket import init_mask_zeros, delta_update
from broadcast import (
    broadcast_server_to_client_initialization,
    div_server_weights,
    add_masks,
    add_server_weights,
)

# My Library
from Enc_and_Dec.init import init_A, init_R
from Enc_and_Dec.Enc import Enc
from Enc_and_Dec.agg import agg_server_weights
from Enc_and_Dec.Dec import Dec




def cross_client_eval(
    model: nn.Module, 
    client_state_dicts: Dict[int, OrderedDict], 
    dataset_train: torch.utils.data.Dataset, 
    dataset_test: torch.utils.data.Dataset, 
    dict_users_train: Dict[int, np.ndarray], 
    dict_users_test: Dict[int, np.ndarray], 
    args: Any, 
    no_cross: bool = True, 
) -> torch.Tensor:
    """Evaluate models across clients. 

    Args:
        model: Neural network model 
        client_state_dicts: Client model states 
        dataset_train: Training dataset 
        dataset_test: Test dataset 
        dict_users_train: Mapping of users to training data indices 
        dict_users_test: Mapping of users to test data indices 
        args: Evaluation arguments 
        no_cross: Whether to only evaluate on own data 

    Returns:
        torch.Tensor: Matrix of cross-client accuracies 
    """
    cross_client_acc_matrix = torch.zeros(
        (len(client_state_dicts), len(client_state_dicts))
    )
    idx_users = list(client_state_dicts.keys())
    for _i, i in enumerate(idx_users):
        model.load_state_dict(client_state_dicts[i])
        for _j, j in enumerate(idx_users):
            if no_cross:
                if i != j:
                    continue
            # eval model i on data from client j
            _, ldr_test = prepare_dataloaders(
                dataset_train,
                dict_users_train[j],
                dataset_test,
                dict_users_test[j],
                args,
            )
            acc = evaluate(model, ldr_test, args)
            cross_client_acc_matrix[_i, _j] = acc
    return cross_client_acc_matrix


def fedselect_algorithm(
    model: nn.Module, 
    args: Any, 
    dataset_train: torch.utils.data.Dataset, 
    dataset_test: torch.utils.data.Dataset, 
    dict_users_train: Dict[int, np.ndarray], 
    dict_users_test: Dict[int, np.ndarray], 
    labels: np.ndarray, 
    idxs_users: List[int], 
    seed_pairs: List[Tuple[int, int]], 
) -> Dict[str, Any]:
    """Main FedSelect federated learning algorithm. 

    Args:
        model: Neural network model 
        args: Training arguments 
        dataset_train: Training dataset 
        dataset_test: Test dataset 
        dict_users_train: Mapping of users to training data indices 
        dict_users_test: Mapping of users to test data indices 
        labels: Data labels 
        idxs_users: List of user indices 
        seed_pairs: List of seed pairs 
    Returns:
        Dict containing:
            - client_accuracies: Accuracy history for each client 
            - labels: Data labels 
            - client_masks: Final client masks 
            - args: Training arguments 
            - cross_client_acc: Cross-client accuracy matrix 
            - lth_convergence: Lottery ticket convergence history 
    """

    # initialize model
    initial_state_dict = copy.deepcopy(model.state_dict()) 
    com_rounds = args.com_rounds 
    # initialize server
    client_accuracies = [{i: 0 for i in idxs_users} for _ in range(com_rounds)] 
    client_state_dicts = {i: copy.deepcopy(initial_state_dict) for i in idxs_users} 
    client_state_dict_prev = {i: copy.deepcopy(initial_state_dict) for i in idxs_users} 
    client_masks = {i: None for i in idxs_users} 
    client_masks_prev = {i: init_mask_zeros(model) for i in idxs_users} 
    server_accumulate_mask = OrderedDict()  
    lth_iters = args.lth_epoch_iters 
    prune_rate = args.prune_percent / 100 
    prune_target = args.prune_target / 100 
    lottery_ticket_convergence = [] 

    B = {} 

    agg_B = {}

    # Begin FL
    #####################################################################################################################
    model_params = 0 # model parameters
    M = args.M # parameters per group
    num_group = 0 # number of groups
    A_dict = {} # initialize A, to save all client's random integer sequence
    flag_A = False # whether to initialize A
    for round_num in range(com_rounds): # traverse communication rounds
        # create file to save encryption time for each round
        with open(f"enc_dec_time/ClientNum({args.num_users})_({args.model})_({args.dataset})_iid({args.iid})_com_rounds({args.com_rounds})__M{args.M}_BS({args.batch_size})_frac({args.frac})_lr({args.lr})_lth_epoch_iters({args.lth_epoch_iters})_la-epochs({args.la_epochs})_prune_target({args.prune_target}).txt", "a") as f:
            f.write('round_num\t\tinit_time\t\tAvg_Enc_time\t\tagg_time\t\tavg_dec_time\t\tagg_B_size(MB)\n')


        round_loss = 0 # current round loss
        Enc_time_list = {}
        Avg_Enc_time = 0 # average encryption time for current round
        agg_B_size = 0.0 # size of aggregated B for current round
        init_time = 0.0 # initialization time
        for i in idxs_users: # traverse each client
            # initialize model
            model.load_state_dict(client_state_dicts[i]) # load client model state
            model_params = sum(p.numel() for p in model.parameters()) # calculate model parameters

            # calculate number of groups. group strategy: group model parameters, each group has M parameters, if the last group has less than M parameters, use 0 to fill.
            num_group = model_params // M
            if model_params % M != 0:
                num_group += 1
            print(f"In round {round_num}, client {i} model params number: {model_params}, num_group: {num_group}")

            #######################Initialize A and R##################################
            if round_num == 0 and not flag_A:  # only initialize once
                # initialize A, to save all client's random integer sequence
                init_start = time.time()
                A_dict = init_A(len(idxs_users), num_group) # initialize A, to save all client's random integer sequence
                # initialize R, all clients share one R
                # each group needs M+1 R values, so the total size should be num_group * (M+1)
                R = init_R(num_group, M+1)
                init_time = time.time() - init_start

                flag_A = True # mark A as initialized
                print(f"A_dict: {A_dict}")
                print(f"R: {R}")

            # get data
            # ldr_train: training data loader
            ldr_train, _ = prepare_dataloaders(
                dataset_train,
                dict_users_train[i],
                dataset_test,
                dict_users_test[i],
                args,
            )
            # Update LTN_i on local data
            client_mask = client_masks_prev.get(i) 
            # Update u_i parameters on local data
            # 0s are global parameters, 1s are local parameters 0 means global parameters, 1 means local parameters
            client_model, loss = train_personalized(model, ldr_train, client_mask, args)
            round_loss += loss
            # Send u_i update to server send u_i update to server here
            if round_num < com_rounds - 1:
                # region: old code (origin code)
                server_accumulate_mask = add_masks(server_accumulate_mask, client_mask)
                # add client model parameters and masks to server weights
                server_weights = add_server_weights(
                    server_weights, 
                    client_model.state_dict(), 
                    client_mask
                )
                # endregion
                       
                # encryption time
                start_time = time.time()  # Start timing the encryption process
                # encryption
                ciphertext = Enc(
                    client_model.state_dict(), 
                    client_mask, 
                    seed_pairs, 
                    A_dict, 
                    R, 
                    i, 
                    M, 
                    num_group, 
                    len(idxs_users)
                    )
                encryption_time = time.time() - start_time  # calculate encryption time for client i
                Enc_time_list[i] = encryption_time
                Avg_Enc_time += encryption_time
                print(f"Client {i} in round {round_num} encryption time: {encryption_time:.4f} seconds")

                # add ciphertext to coefficient set B
                B[i] = ciphertext  # use client ID as key to store coefficient set B[i][k] represents the coefficient of the k-th group of the i-th client
                

            client_state_dicts[i] = copy.deepcopy(client_model.state_dict())
            client_masks[i] = copy.deepcopy(client_mask)

            if round_num % lth_iters == 0 and round_num != 0:
                client_mask = delta_update(
                    prune_rate, # pruning rate
                    client_state_dicts[i], # current model state
                    client_state_dict_prev[i], # previous model state
                    client_masks_prev[i], # previous mask
                    bound=prune_target, # pruning bound
                    invert=True, # invert
                )
                client_state_dict_prev[i] = copy.deepcopy(client_state_dicts[i])
                client_masks_prev[i] = copy.deepcopy(client_mask) 
        Avg_Enc_time /= len(idxs_users)
        for client_id, enc_time in Enc_time_list.items():
            print(f"Client {client_id}: {enc_time:.4f} seconds")
        print(f"Average encryption time for round {round_num}: {Avg_Enc_time:.4f} seconds")
        


        round_loss /= len(idxs_users) # calculate current round loss
        cross_client_acc = cross_client_eval(
            model, # model
            client_state_dicts, # current model state
            dataset_train, # training dataset
            dataset_test, # test dataset
            dict_users_train, # training data indices
            dict_users_test, # test data indices
            args, # training parameters
        )

        accs = torch.diag(cross_client_acc) # diagonal elements
        for i in range(len(accs)):
            client_accuracies[round_num][i] = accs[i] # update client accuracy
        print("###############################################################")
        print("Client Accs: ", accs, " | Mean: ", accs.mean()) # print client accuracy
        print("###############################################################")

        # write average accuracy and average loss to log file
        with open(f"logs/ClientNum({args.num_users})_({args.model})_iid({args.iid})_com_rounds({args.com_rounds})_BS({args.batch_size})_frac({args.frac})_lr({args.lr})_params({model_params})_lth_epoch_iters({args.lth_epoch_iters})_la-epochs({args.la_epochs})_prune_target({args.prune_target}).txt", "a") as f:
            f.write(f"{round_num}\t\t{accs.mean():.4f}\t\t{round_loss:.4f}\n")

        agg_time = 0.0
        avg_dec_time = 0.0
        if round_num < com_rounds - 1:
            """aggregate"""
            # Server averages u_i server averages u_i
            server_weights = div_server_weights(server_weights, server_accumulate_mask, len(idxs_users))
            agg_start_time = time.time()
            agg_B = agg_server_weights(B, server_accumulate_mask, len(idxs_users), num_group) # directly aggregate # agg_B[k] represents the aggregation coefficient of the k-th group
            agg_B_size = com_agg_B_size(agg_B)
            agg_time = time.time() - agg_start_time  # calculate aggregation time for round t
            print(f"Aggregation time for round {round_num}: {agg_time:.4f} seconds")
            """broadcast"""
            # Server broadcasts non lottery ticket parameters u_i to every device server broadcasts non lottery ticket parameters u_i to every device
            # for i in idxs_users:
            #     client_state_dicts[i] = broadcast_server_to_client_initialization(
            #         server_weights, # server weights
            #         client_masks[i], # client masks
            #         client_state_dicts[i] # client model state
            #     )

        
            # client first restores server weights
            for i in idxs_users:
                # decryption time
                dec_start_time = time.time()
                client_state_dicts[i] = Dec(
                  agg_B,  # aggregated coefficient set
                  client_masks[i],  # client masks
                  A_dict,  # sequence A, A[i][k] represents the verification point of the k-th group of client i
                  R,  # sequence R, R[k][j] represents the j-th interpolation point of the k-th group. that is x_points
                  M,  # number of elements in each group
                  num_group, # number of groups
                  len(idxs_users), # number of clients  
                  client_state_dicts[i], # client model state
               )
                dec_time = time.time() - dec_start_time  # calculate decryption time for client i
                avg_dec_time += dec_time
                print(f"Client {i} in round {round_num} decryption time: {dec_time:.4f} seconds")
        avg_dec_time /= len(idxs_users)
        # write average privacy protection time and data size to log file
        with open(f"enc_dec_time/ClientNum({args.num_users})_({args.model})_({args.dataset})_iid({args.iid})_com_rounds({args.com_rounds})__M{args.M}_BS({args.batch_size})_frac({args.frac})_lr({args.lr})_lth_epoch_iters({args.lth_epoch_iters})_la-epochs({args.la_epochs})_prune_target({args.prune_target}).txt", "a") as f:
            f.write(f"{round_num}\t\t{init_time:.4f}\t\t{Avg_Enc_time:.4f}\t\t{agg_time:.4f}\t\t{avg_dec_time:.4f}\t\t{agg_B_size:.4f}\n")


            # noinspection PyTypeHints 
            server_accumulate_mask = OrderedDict() # server accumulate mask
            # noinspection PyTypeHints 
            server_weights = OrderedDict() # server weights



    # calculate cross-client accuracy
    cross_client_acc = cross_client_eval(
        model,              # model
        client_state_dicts, # client model state
        dataset_train,      # training dataset
        dataset_test,       # test dataset
        dict_users_train,   # training data indices
        dict_users_test,    # test data indices
        args,               # training parameters
        no_cross=False,     # whether to only evaluate own data
    )

    # output dictionary
    out_dict = {
        "client_accuracies": client_accuracies, # client accuracy history
        "labels": labels,                       # data labels
        "client_masks": client_masks,           # client masks
        "args": args,                           # training parameters
        "cross_client_acc": cross_client_acc,   # cross-client accuracy matrix
        "lth_convergence": lottery_ticket_convergence, # lottery ticket convergence history
    }

    return out_dict 

def com_agg_B_size(agg_B: List[float]) -> float:
    """
    calculate the data size of agg_B list, return unit is MB
    
    Args:
    agg_B: list containing floats
    
    Returns:
    float: data size, unit is MB
    """
    import sys
    import numpy as np
    
    # method 1: accurate calculation (suitable for pure Python list)
    # size_in_bytes = sys.getsizeof(agg_B) + sum(sys.getsizeof(x) for x in agg_B)
    
    # method 2: efficient calculation (
    if len(agg_B) == 0:
        return 0.0
    
    # assume all elements are float (8 bytes)
    size_in_bytes = sys.getsizeof(agg_B) + len(agg_B) * 8
    
    # method 3: convert to numpy array calculation (if allowed to use numpy)
    # arr = np.array(agg_B)
    # size_in_bytes = arr.nbytes
    
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb