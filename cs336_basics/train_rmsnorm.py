import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        防止梯度爆炸/消失：残差连接(子层输出加回输入，保证至少残差流通过)、归一化(标准尺度)

        一个Transformer block架构：
        子层1：多头自注意力（让每个token看其他token）
        子层2：前馈网络FFN（两层MLP，逐token独立处理）

        ##残差连接##
        Post-Norm:                            |   Pre-Norm:
        x ──→ Attention ──→ (+) ──→ LN ──→ x  |   x ──→ LN ──→ Attention ──→ (+) ──→ x
             ↑              ↑                 |      ↑                      ↑
             └──────────────┘                 |      └──────────────────────┘
            残差加完再LayerNorm                |    LayerNorm在子层前，残差流不被碰

        Pre-Norm的优势：保留残差流，深层训练更稳定
        Pre-Norm的代价：残差流数值会随层数增长，所以最后要补一个LayerNorm归一化：
        
        for block in blocks:
            x=block(x)    #每个block内部是pre-norm
        x=final_LayerNorm(x)    #最后统一归一化一次

        ##归一化##
        LayerNorm：对每个样本内部的特征做归一化，不依赖batch的批次数
        LayerNorm公式：(a-μ)/(σ*γ)+β
        简单版：RMSNorm(a_i)=g_i*a_i/RMS(a)，g_i是要学习的参数
        RMS：均方根 Root Mean Square -> RMS(a)=√[ε+(∑a_i²/d)]
        去掉:-μ（Transformer激活值均值本身接近零，不引入浮点数误差）
             +β（去掉bias偏差，可以节省参数）

        Args:
            RMS(a)=√[ε+(∑a_i²/d)]
            d_model: int —— 模型的隐藏维度d
            eps: float = 1e-5 —— 用于数值稳定性的epsilon值ε
            device: torch.device | None = None —— 存储参数的设备
            dtype: torch.dtype | None = None —— 参数的数据类型
        """

        super().__init__()
        self.d_model=d_model
        self.eps=eps
        #要学习的参数是gain(g_i)，形状和隐藏维度d_model相同，先初始化为1。
        self.gain_weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm(a_i)=g_i*a_i/RMS(a)，g_i是要学习的参数
        RMS：均方根 Root Mean Square -> RMS(a)=√[ε+(∑a_i²/d)]
        """
        # 1.平方时float32可能溢出，处理dtype数据类型
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 2.计算∑a_i²/d -> (batch,seq,1)
        mean_sq=(x**2).mean(dim=-1,keepdim=True)
        # 3.计算RMS -> (batch,seq,1)
        rms=torch.sqrt(mean_sq+self.eps)
        # 4.计算RMSNorm  (batch,seq,d)/(batch,seq,1)*(d,)=
        result=(x/rms)*self.gain_weight
        # 5.降回dtype
        return result.to(in_dtype)