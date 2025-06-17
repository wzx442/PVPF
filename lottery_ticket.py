# Importing Libraries
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import types
from collections import OrderedDict
from typing import List, Tuple, Dict, OrderedDict, Optional, Union


def eval_per_layer_sparsity(mask: OrderedDict) -> List[Tuple[str, str, str, float]]:
    """Calculate sparsity statistics for each weight layer in the mask.

    Args:
        mask: OrderedDict containing binary masks for model parameters

    Returns:
        List of tuples containing (num ones, num zeros, layer name, sparsity) for each weight layer
    """
    return [
        (
            f"1: {torch.count_nonzero(mask[name])}",
            f"0: {torch.count_nonzero(1-mask[name])}",
            name,
            (
                torch.count_nonzero(1 - mask[name])
                / (
                    torch.count_nonzero(mask[name])
                    + torch.count_nonzero(1 - mask[name])
                )
            ).item(),
        )
        for name in mask.keys()
        if "weight" in name
    ]


def eval_layer_sparsity(
    mask: OrderedDict, layer_name: str
) -> Tuple[str, str, str, float]:
    """Calculate sparsity statistics for a specific layer in the mask. 计算指定层的掩码稀疏度统计。

    Args:
        mask: OrderedDict containing binary masks for model parameters 包含模型参数的二进制掩码的有序字典
        layer_name: Name of layer to analyze 要分析的层的名称

    Returns:
        Tuple containing (num ones, num zeros, layer name, sparsity) for specified layer 包含指定层的(num ones, num zeros, layer name, sparsity)的元组
    """
    if layer_name not in mask:
        print(f"Warning: Layer '{layer_name}' not found in mask!")
        return "Layer not found"
    return (
        f"1: {torch.count_nonzero(mask[layer_name])}", # 计算指定层的掩码中1的数量
        f"0: {torch.count_nonzero(1-mask[layer_name])}", # 计算指定层的掩码中0的数量
        layer_name, # 指定层的名称
        (
            torch.count_nonzero(1 - mask[layer_name]) # 计算指定层的掩码中0的数量   
            / (
                torch.count_nonzero(mask[layer_name]) # 计算指定层的掩码中1的数量
                + torch.count_nonzero(1 - mask[layer_name]) # 计算指定层的掩码中0的数量
            )
        ).item(), # 计算指定层的稀疏度
    )


def print_nonzeros(
    model: OrderedDict, verbose: bool = False, invert: bool = False
) -> float:
    """Print statistics about non-zero parameters in model. 打印模型中非零参数的统计信息。

    Args:
        model: OrderedDict containing model parameters 包含模型参数的有序字典
        verbose: Whether to print detailed statistics 是否打印详细统计信息
        invert: Whether to count zeros instead of non-zeros 是否计数0而不是非零

    Returns:
        Percentage of pruned parameters 修剪参数的百分比
    """
    nonzero = total = 0
    for name, p in model.items():
        tensor = p.data.cpu().numpy() # 将模型参数转换为numpy数组
        nz_count = (
            np.count_nonzero(tensor) if not invert else np.count_nonzero(1 - tensor) # 如果未反转，则计算非零参数的数量，否则计算0的数量
        )
        total_params = np.prod(tensor.shape) # 计算模型参数的总数量
        nonzero += nz_count # 累加非零参数的数量
        total += total_params # 累加模型参数的总数量
        if verbose:
            print(
                f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
            )
    if verbose:
        print(
            f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
        )
    return 100 * (total - nonzero) / total


def print_lth_stats(mask: OrderedDict, invert: bool = False) -> None:
    """Print lottery ticket hypothesis statistics about mask sparsity. 打印掩码稀疏度的统计信息。

    Args:
        mask: OrderedDict containing binary masks 包含二进制掩码的有序字典
        invert: Whether to invert the sparsity calculation 是否反转稀疏度计算
    """
    current_prune = print_nonzeros(mask, invert=invert) # 打印非零参数的统计信息
    print(f"Mask Sparsity: {current_prune:.2f}%") # 打印掩码的稀疏度


def _violates_bound(
    mask: torch.Tensor, bound: Optional[float] = None, invert: bool = False
) -> bool:
    """Check if mask sparsity violates specified bound. 检查掩码稀疏度是否违反指定的界限。

    Args:
        mask: Binary mask tensor 二进制掩码张量
        bound: Maximum allowed sparsity 最大允许的稀疏度
        invert: Whether to invert the sparsity calculation 是否反转稀疏度计算

    Returns:
        True if bound is violated, False otherwise 如果违反界限，则返回True，否则返回False  
    """
    if invert:
        return (
            torch.count_nonzero(mask)
            / (torch.count_nonzero(mask) + torch.count_nonzero(1 - mask))
        ).item() > bound
    else:
        return (
            torch.count_nonzero(1 - mask)
            / (torch.count_nonzero(mask) + torch.count_nonzero(1 - mask))
        ).item() > bound


def init_mask(model: nn.Module) -> OrderedDict:
    """Initialize binary mask of ones for model parameters. 初始化模型参数的二进制掩码为1。

    Args:
        model: Neural network model 神经网络模型

    Returns:
        OrderedDict containing binary masks initialized to ones 包含初始化为1的二进制掩码的有序字典 
    """
    mask = OrderedDict()
    for name, param in model.named_parameters():
        # 为BatchNorm层参数特殊处理
        if 'batch_norm' in name or '.bn' in name or 'running_' in name:
            # 批量归一化层参数通常应该是全局的（掩码为0）
            mask[name] = torch.zeros_like(param)
        else:
            mask[name] = torch.ones_like(param)
    return mask


def init_mask_zeros(model: nn.Module) -> OrderedDict:
    """Initialize binary mask of zeros for model parameters. 初始化模型参数的二进制掩码为0。

    Args:
        model: Neural network model 神经网络模型

    Returns:
        OrderedDict containing binary masks initialized to zeros 包含初始化为0的二进制掩码的有序字典
    """
    mask = OrderedDict()
    for name, param in model.named_parameters():
        # 为BatchNorm层参数特殊处理
        if 'batch_norm' in name or '.bn' in name or 'running_' in name:
            # 批量归一化层参数应该保持全局（掩码为0）
            mask[name] = torch.zeros_like(param)
        else:
            mask[name] = torch.zeros_like(param)
    return mask


def get_mask_from_delta(
    prune_percent: float, # 要修剪的参数的百分比    
    current_state_dict: OrderedDict, # 当前模型状态
    prev_state_dict: OrderedDict, # 前一个模型状态
    current_mask: OrderedDict, # 当前的二进制掩码
    bound: float = 0.80, # 最大允许的稀疏度
    invert: bool = True, # 是否反转修剪逻辑
) -> OrderedDict:
    """Generate new mask based on parameter changes between states. 根据状态之间的参数变化生成新的掩码。

    Args:
        prune_percent: Percentage of parameters to prune 要修剪的参数的百分比
        current_state_dict: Current model state 当前模型状态
        prev_state_dict: Previous model state 前一个模型状态
        current_mask: Current binary mask 当前的二进制掩码
        bound: Maximum allowed sparsity 最大允许的稀疏度
        invert: Whether to invert the pruning logic 是否反转修剪逻辑

    Returns:
        Updated binary mask based on parameter changes 基于参数变化更新的二进制掩码
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return_mask = copy.deepcopy(current_mask) # 深拷贝当前掩码
    
    # 存储处理的权重层，用于后续打印统计信息
    processed_weight_layers = []
    
    for name, param in current_state_dict.items():
        if "weight" in name: # 如果参数名包含"weight"
            processed_weight_layers.append(name)  # 记录处理过的权重层
            if _violates_bound(current_mask[name], bound=bound, invert=invert): # 如果当前掩码违反了最大允许的稀疏度
                continue # 跳过当前参数
            tensor = param.data.cpu().numpy() # 将当前参数转换为numpy数组
            compare_tensor = prev_state_dict[name].cpu().numpy() # 将前一个模型状态的参数转换为numpy数组
            delta_tensor = np.abs(tensor - compare_tensor) # 计算当前参数和前一个模型状态的参数之间的差异

            delta_percentile_tensor = (
                delta_tensor[current_mask[name].cpu().numpy() == 1] # 如果掩码为1
                if not invert # 如果未反转
                else delta_tensor[current_mask[name].cpu().numpy() == 0] # 如果掩码为0
            )
            sorted_weights = np.sort(np.abs(delta_percentile_tensor)) # 对差异进行排序  
            if not invert: # 如果未反转
                cutoff_index = np.round(prune_percent * sorted_weights.size).astype(int) # 计算截断索引
                cutoff = sorted_weights[cutoff_index] # 计算截断值

                # Convert Tensors to numpy and calculate 将张量转换为numpy并计算    
                new_mask = np.where(
                    abs(delta_tensor) <= cutoff, 0, return_mask[name].cpu().numpy() # 如果差异小于截断值，则掩码为0，否则为当前掩码 
                )
                return_mask[name] = torch.from_numpy(new_mask).to(device) # 将新的掩码转换为张量并存储回掩码字典中
            else: # 如果反转    
                cutoff_index = np.round(
                    (1 - prune_percent) * sorted_weights.size
                ).astype(int) # 计算截断索引    
                cutoff = sorted_weights[cutoff_index] # 计算截断值  

                # Convert Tensors to numpy and calculate 将张量转换为numpy并计算
                new_mask = np.where( 
                    abs(delta_tensor) >= cutoff, 1, return_mask[name].cpu().numpy() # 如果差异大于截断值，则掩码为1，否则为当前掩码
                )
                return_mask[name] = torch.from_numpy(new_mask).to(device) # 将新的掩码转换为张量并存储回掩码字典中
    
    # 打印第一个权重层的稀疏度统计，而不是硬编码的"fc.weight"
    if processed_weight_layers:
        print(f"模型包含以下权重层: {processed_weight_layers}")
        print(eval_layer_sparsity(return_mask, processed_weight_layers[0]))
    else:
        print("警告: 模型中没有找到权重层")
    
    return return_mask


def delta_update(
    prune_percent: float, # 要修剪的参数的百分比
    current_state_dict: OrderedDict, # 当前模型状态
    prev_state_dict: OrderedDict, # 前一个模型状态
    current_mask: OrderedDict, # 当前的二进制掩码
    bound: float = 0.80, # 最大允许的稀疏度
    invert: bool = False, # 是否反转修剪逻辑
) -> OrderedDict:
    """Update mask based on parameter changes between states.

    Args:
        prune_percent: Percentage of parameters to prune 要修剪的参数的百分比
        current_state_dict: Current model state 当前模型状态
        prev_state_dict: Previous model state 前一个模型状态
        current_mask: Current binary mask 当前的二进制掩码
        bound: Maximum allowed sparsity 最大允许的稀疏度
        invert: Whether to invert the pruning logic 是否反转修剪逻辑

    Returns:
        Updated binary mask 基于参数变化更新的二进制掩码    
    """
    mask = get_mask_from_delta(
        prune_percent, # 要修剪的参数的百分比   
        current_state_dict, # 当前模型状态
        prev_state_dict, # 前一个模型状态
        current_mask, # 当前的二进制掩码
        bound=bound, # 最大允许的稀疏度
        invert=invert, # 是否反转修剪逻辑
    )
    print_lth_stats(mask, invert=invert) # 打印掩码的稀疏度统计
    return mask
