import torch

class AdamW(torch.optim.Optimizer):
    """
    假设损失曲面像一条弯曲山谷：
        梯度方向可能一边指向谷底，一边左右摇摆
        一阶矩把方向平滑下来，减少左右抖动
        二阶矩看每个方向的梯度尺度，限制梯度特别大的参数更新
    
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
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("betas must be in [0, 1)")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # self.state[p] 是该参数专属的状态字典。
                state = self.state[p]

                #第一次遇到参数时初始化 step、exp_avg、exp_avg_sq。

                # state["step"] = ...

                # state["exp_avg"] = ...

                # state["exp_avg_sq"] = ...

                #更新 step。

                #更新一阶矩 exp_avg。

                #更新二阶矩 exp_avg_sq。

                #应用 decoupled weight decay。

                #计算题目中的 alpha_t 并更新 p。

        return loss