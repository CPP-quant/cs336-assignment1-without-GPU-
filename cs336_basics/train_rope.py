import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    RoPE (Rotaty Position Embedding)
    目的：把token的序列位置编码进query和key向量
    操作：
    以query向量为例: q_i=W_q*x_i。其中i是token在序列的位置，q_i的维度d_k是单头注意力的维数
    1. 将q_i的d_k个分量两两配对(要求2|d_k)
    2. 对每个二维向量(d_k/2对)逆时针旋转θ_i,k, 旋转矩阵Rotary_i,k
       |cosθ -sinθ|
       |sinθ  cosθ|
    3. θ_i,k=i/Θ^[(2k-2)/d_k]
       i是token的序列位置, k是二维向量对数(1,2...d_k/2), Θ是超参数
    4. R_i=diag(Rotaty_i,k), k=1,2...d_k/2, d_k*d_k的旋转矩阵
       q'_i=R_i*q_i, 即把token的序列位置信息i编码进了query向量
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k=d_k
        self.max_seq_len=max_seq_len

        #torch.arange(d_k)生成从k-1=0到d_k/2-1的一维张量（维度）
        pair_indices=torch.arange(d_k//2,device=device,dtype=torch.float32)
        #计算θ_i,k=i/Θ^[(2k-2)/d_k]公式中的指数: -(2k-2)/d_k
        #k-1=0,1,2,...,d_k/2-1
        #int**一维tensor=一维tensor
        inv_freq=theta**(-2*pair_indices/d_k)
        #torch.arange(max_seq_len)生成从i=0到max_seq_len-1的一维张量（token位置）
        positions=torch.arange(max_seq_len,device=device,dtype=torch.float32)
        #torch.outer(positions,inv_freq)对两个一维张量做外积
        #外积前列后行。形状为(max_seq_len,d_k/2)。i行k列。
        angles=torch.outer(positions,inv_freq)
        
        #对角度逐元素计算正余弦，将θ_i,k注册为固定buffer，不是可训练参数
        #两个buffer的形状都是(max_seq_len,d_k/2)
        self.register_buffer("cos_cached",angles.cos(),persistent=False)
        self.register_buffer("sin_cached",angles.sin(),persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)->torch.Tensor:
        """
        Args:
            x的形状是(...,seq_len,d_k)
            token_positions的形状是 (...,seq_len)
            ...可以是batch批次等前置维度
            其中x和token_positions的前几位参数都一样
            seq_len<=max_seq_len
        """

        # 1.构造三角变换，与数据对齐，便于逐元素相乘
        #cos和sin通过token_positions对着seq_len维度查出来
        #形状为(..., seq_len,d_k/2)
        #.to(dtype=x.dtype)把查到的值转成和输入x相同的数据类型
        cos=self.cos_cached[token_positions].to(dtype=x.dtype)
        sin=self.sin_cached[token_positions].to(dtype=x.dtype)

        # 2.拆分x张量的奇偶项
        #reshape改变张量的形状，不改变其中数据的顺序
        #*x.shape[:-1]是d_k前的维度
        x_pairs=x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        #...表示前面的维度都保留
        x_even=x_pairs[..., 0]
        x_odd=x_pairs[..., 1]

        # 3.旋转变换
        #x、y、cos、sin形状均为(..., seq_len,d_k/2)
        #逐元素乘法: 
        #每个位置、每个坐标对分别使用对应的cos/sin，逐元素实现一个2×2旋转
        y_even=x_even*cos-x_odd*sin
        y_odd=x_even*sin+x_odd*cos

        # 4.合并奇偶项：合并最后一个维度并展平
        #torch.stack在末尾增加一个维度，把对应的y_even和y_odd放入同一对
        #2个(..., seq_len,d_k/2)合并成(..., seq_len,d_k/2,2)
        y=torch.stack((y_even, y_odd), dim=-1)
        #flatten(start_dim=-2)展平最后两个维度
        #(..., seq_len,d_k/2,2)->(..., seq_len,d_k)
        return y.flatten(start_dim=-2)