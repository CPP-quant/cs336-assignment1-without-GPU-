import torch
import torch.nn as nn

def softmax(in_features:torch.Tensor,dim:int)->torch.Tensor:
    """
    softmax: 把未归一化的分数向量转成归一化分布的操作
    softmax(v)_i=exp(v_i)/Σexp(v)
    防止数值过大exp溢出: 我们会从v的所有元素中减去v的最大项, 使新的最大项变成0
    exp(x-c)=exp(x)/exp(c), 因此该处理后softmax不变

    Args:
        in_features:Float[Tensor, "..."]  任意形状的浮点张量
        dim:int  指定对哪一维做softmax 的整数
    Returns:
        返回形状与in_features相同, dim维度完成softmax归一化
    """
    #沿dim求最大值：其他下标一样、该维度下标不同的元素放一起softmax
    #keepdim=True让结果保留该维
    #.values取最大值，不取最大值位置索引
    max_values=in_features.max(dim=dim,keepdim=True).values
    #广播相减去：广播即将最大值所在维数扩展到和in_features相同
    shifted=in_features-max_values
    #沿dim维度做softmax归一化（所有数字都要exp，只是按照dim分组求和）
    exp_values=torch.exp(shifted)
    return exp_values/exp_values.sum(dim=dim,keepdim=True)
