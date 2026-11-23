import torch
import torch.nn as nn
from cs336_basics.train_multihead_self_attention_with_rope import(
    MultiHeadSelfAttentionWithRoPE,
)
from cs336_basics.train_swiglu import SwiGLU
from cs336_basics.train_rmsnorm import RMSNorm

class Transformer_block(nn.Module):
    """
    pre-norm Transformer block=Attention + SwiGlU(FFN)
    应用pre-Norm残差连接和RMSNorm归一化
    y=MultiHeadSelfAttentionWithRoPE(RMSNorm(x))
    z=SwiGLU(RMSNorm(y))
    Weights:
        Attention: W_Q、W_K、W_V
        LayerNorm (RMSNorm): LN1、LN2
        SwiGLU (FFN): W1、W2、W3
    Args:
        d_model, num_heads, d_ff, max_seq_len, theta
    Returns:
        torch.Tensor: (..., seq_len,d_model)
    """
    def __init__(self,
        d_model,num_heads,d_ff,
        max_seq_len,theta,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.ln1=RMSNorm(d_model,device=device,dtype=dtype)
        self.attention=MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn=SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=x+self.attention(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x