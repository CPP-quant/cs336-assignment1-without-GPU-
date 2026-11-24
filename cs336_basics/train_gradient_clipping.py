import torch

def gradient_clipping(param,M,eps=1e-6):
    grads=[p.grad for p in param if p.grad is not None]
    #view()展平: 把任意形状的张量展平成一维向量; -1表示自动推断长度
    #torch.cat()拼接
    #.norm(2)计算L2范数，输出标量
    total_norm=torch.cat([g.view(-1) for g in grads]).norm(2)

    clip_coef=M/(total_norm+eps)
    if M<=total_norm:
        for g in grads:
            g.mul_(clip_coef)