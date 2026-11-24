import torch
import math
from typing import Any

class AdamW(torch.optim.Optimizer):
    """
    假设损失曲面像一条弯曲山谷：
        梯度方向可能一边指向谷底，一边左右摇摆
        一阶矩把方向平滑下来，减少左右抖动
        m_t可理解为近期梯度的加权平均
        二阶矩看每个方向的梯度尺度，限制梯度特别大的参数更新
        v_t可理解为近期梯度平方的加权平均, 限制梯度特别大的参数更新
    
    1  init(θ)                                     ▷ 初始化可学习参数
    2  m ← 0                                       ▷ 一阶矩向量的初始值,与 θ 同形状
    3  v ← 0                                       ▷ 二阶矩向量的初始值,与 θ 同形状
    4  for t = 1, …, T do
    5      Sample batch of data B_t
    6      g ← ∇_θ ℓ(θ; B_t)                      ▷ 计算损失的梯度
    7      α_t ← α √(1-β_2^t)/(1-β_1^t)            ▷ 计算迭代 t 的调整后的α
    8      θ ← θ − αλθ                             ▷ 应用 weight decay
    9      m ← β_1 m + (1 − β_1) g                 ▷ 更新一阶矩估计
    10     v ← β_2 v + (1 − β_2) g²                ▷ 更新二阶矩估计
    11     θ ← θ − α_t m/(√v + ε)                  ▷ 应用矩调整后的权重更新
    12  end for
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        #params: 参数集合
        #lr: 学习率α
        #betas: β_1和β_2
        #eps: 数值稳定项ε
        #weight_decay: 权重衰减率λ

        #把四个超参打包成一个字典
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        #把params和defaults交给torch.optim.Optimizer基类去处理
        #存放在self.param_groups和self.state
        super().__init__(params,defaults)

    @torch.no_grad()    #关掉梯度，节省内存
    def step(self, closure=None):
        #惯例：如果用户传了closure, 就在更新前调用它重新算一次loss
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            #提取超参数
            lr=group["lr"]
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                #拿到p的梯度
                grad=p.grad
                #self.state[p]是该参数专属的状态字典。
                state=self.state[p]
                #第一次遇到参数时初始化step、exp_avg、exp_avg_sq
                if len(state)==0:
                    #迭代计数t、一阶矩m、二阶矩v
                    #m←β_1*m+(1−β_1)g 、v←β_2*v+(1−β_2)g²
                    #所以m v的形状和参数表p一样
                    state["step"]=0
                    state["exp_avg"]=torch.zeros_like(p)
                    state["exp_avg_sq"]=torch.zeros_like(p)
                exp_avg=state["exp_avg"]
                exp_avg_sq=state["exp_avg_sq"]

                #更新step
                state["step"]+=1
                t=state["step"]

                #就地更新计算函数(更省内存)
                #t.mul_(x)表示t=t*x
                #t.add_(x,alpha=a)表示t=t+a*x
                #t.addcmul_(a,b,value=v)表示t=t+v*(a*b)
                #t.addcdiv_(a,b,value=v)表示t=t+v*(a/b)

                #修正学习率α_t←α*√(1-β_2^t)/(1-β_1^t) 
                bias_correction1=1-beta1**t
                bias_correction2=1-beta2**t
                alpha_t=lr*math.sqrt(bias_correction2)/bias_correction1
                #应用decoupled weight decay: θ←θ−αλθ
                p.mul_(1-lr*weight_decay)
                #更新一阶矩exp_avg: m←β_1*m+(1−β_1)g
                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                #更新二阶矩exp_avg_sq: v←β_2*v+(1−β_2)g²
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                #计算α_t并更新p: θ←θ−α_t*m/(√v+ε)
                denom=exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg,denom,value=-alpha_t)

        return loss