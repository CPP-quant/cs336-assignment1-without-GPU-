import torch
import numpy as np

def data_loading(
    dataset:np.array,
    batch_size:int,
    context_length:int,
    device:str
)->tuple[torch.Tensor, torch.Tensor]:
    """
    从1D token_id序列中随机采样batch_size个(input, target)tuple, torch.LongTensors格式
    """
    # 1.随机采样batch_size个起始位置
    data_len=len(dataset)
    max_start_idx=data_len-context_length-1
    #np.random.randint(start,end+1,size)得到随机数数组
    start_indices=np.random.randint(0,max_start_idx+1,size=batch_size)

    # 2.构建inputs和targets
    #np.int64和torch.long的dtype对应
    inputs=np.zeros((batch_size,context_length),dtype=np.int64)
    targets=np.zeros((batch_size,context_length),dtype=np.int64)
    for i,start in enumerate(start_indices):
        inputs[i]=dataset[start:start+context_length]
        targets[i]=dataset[start+1:start+context_length+1]

    # 3.将numpy转换为torch.LongTensors并放到指定设备
    inputs_tensor=torch.tensor(inputs,dtype=torch.long,device=device)
    targets_tensor=torch.tensor(targets,dtype=torch.long,device=device)

    return inputs_tensor,targets_tensor