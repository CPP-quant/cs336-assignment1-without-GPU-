import os
from typing import BinaryIO,IO
import torch
import torch.nn as nn
"""
恢复中断的训练所需状态有三类：
    1.模型权重. 存放在model.state_dict()	               
    ->防止模型回到随机初始化
    2.优化器状态. 存放在optimizer.state_dict()
    ->防止AdamW的一阶/二阶矩丢失,优化轨迹断裂,loss会突然跳变
    3.训练进度. 一个int, 如iteration
    ->防止不知道该从第几步继续,学习率调度/日志/数据顺序全乱
"""
def save_checkpoint(
        model:nn.Module,
        optimizer:torch.optim.Optimizer,
        iteration:int,
        out:str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    目的: 
        把模型、优化器和迭代次数的所有状态转储到类文件对象out中
    步骤: 
        先用模型和优化器的state_dict方法获取它们相关的状态
        然后用torch.save(obj,out)把obj转储到out, 即输出
    """
    checkpoint={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":iteration,
    }
    torch.save(checkpoint,out)

def load_checkpoint(
    src:str | os.PathLike | BinaryIO | IO[bytes],
    model:nn.Module,
    optimizer:torch.optim.Optimizer,
):
    """
    目的:
        从src(路径或类文件对象)加载一个checkpoint来恢复模型和优化器的状态
    步骤:
        先用torch.load(src)恢复在save_checkpoint实现中保存的内容
        然后用模型和优化器的load_state_dict方法把它们恢复到之前的状态。
    Returns:
        返回保存到checkpoint中的迭代次数
    """
    #model.parameters()函数 返回模型的迭代器
    #next()函数 从迭代器里取下一个元素
    #第一次调用取第一个元素, 第二次调用取第二个, 以此类推
    #.device函数 取Tensor变量的device属性, 即返回张量所在的设备
    device=next(model.parameters()).device
    checkpoint=torch.load(src,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]