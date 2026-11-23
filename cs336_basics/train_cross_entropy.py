import torch

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor):
    """
    预测集θ和训练集D的交叉熵: l(θ,D) = -Σ p_D(j) * p_θ(j).log()  其中j为类别

    多分类交叉熵中, 假设正确类别编号是target, 训练目标的p_D是独热分布: 正确为1, 其他都是0
    带入得到l=-Σp_θ(target).log()
    softmax给出正确类别target的概率是p_θ(t)=target_logits.exp()/logits.exp().sum()
    所以l=logits.exp().sum().log()-target_logits

    避免数值溢出 -> 同减max_logits
    所以p_θ(t)=(target_logits-max_logits).exp()/(logits-max_logits).exp().sum()

    所以l=(logits-max_logits).exp().sum().log()-(target_logits-max_logits)
    Args:
        logits (batch_size,seq_len,vocab_size)
        targets (batch_size,seq_len)
    Returns:
        交叉熵总量(标量): 对batch和sequence中所有位置的损失取平均
    """
    # 1.计算max_logits
    #沿dim=-1 即vocab_size维度求最大值
    #keepdim=True 表示让结果保留该维
    #.values 表示取最大值，不取最大值位置索引
    max_logits=logits.max(dim=-1,keepdim=True).values

    # 2.计算(logits-max_logits).exp().sum().log()
    #稳定地计算log(sum(exp(logits-max)))
    shifted_logits=logits-max_logits
    log_sum_exp=torch.exp(shifted_logits).sum(dim=-1).log()

    # 3.计算target_logits
    #unsqueeze(-1)函数把(batch_size,seq_len)扩展成(batch_size,seq_len,1)
    #即把seq_len维度的每个值"加括号", 即正确token_id序号
    index=targets.unsqueeze(-1)
    #torch.gather(input,dim,index)函数表示:
    #沿着dim维, 用index里的每个值当"下标", 去input里把下标对应元素"抓"出来。输出形状和index完全一样
    #作用是找到目标正确token_id对应的logits值
    target_logits=torch.gather(input=logits,dim=-1,index=index)

    # 4.计算交叉熵损失张量
    #.squeeze(-1)函数表示合并最后一维, 成为(batch_size,seq_len)
    losses=log_sum_exp-(target_logits.squeeze(-1)-max_logits.squeeze(-1))

    # 5.总平均交叉熵
    return losses.mean()