import torch
import torch.nn as nn
from cs336_basics.train_scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.train_rope import RoPE

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    MultiHeadSelfAttention(x)=W_O*MultiHead(W_Q*x,W_K*x,W_V*x)
    head_dim=d_model/num_heads=d_k=d_v
    """
    def __init__(self,
        d_model : int,
        num_heads : int,
        max_seq_len : int,
        theta : float,
        device : torch.device | None = None,
        dtype : torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads

        #定义四个矩阵变换 (未被切片)
        self.q_proj=nn.Linear(d_model,d_model,bias=False,device=device,dtype=dtype)
        self.k_proj=nn.Linear(d_model,d_model,bias=False,device=device,dtype=dtype)
        self.v_proj=nn.Linear(d_model,d_model,bias=False,device=device,dtype=dtype)
        self.o_proj=nn.Linear(d_model,d_model,bias=False,device=device,dtype=dtype)

        #RoPE
        self.rope=RoPE(theta,self.head_dim,max_seq_len,device)

    def forward(self,
        x : torch.Tensor,
        token_positions : torch.Tensor | None = None,
    )->torch.Tensor:
        # 1.预处理
        #x: (..., seq_len,d_model)
        seq_len=x.shape[-2]
        #(..., seq_len,d_model)
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        #如果未提供token序列位置，就用0,1,...,seq_len-1
        if token_positions is None:
            token_positions=torch.arange(seq_len,
                device=x.device,
                dtype=torch.long,
            )

        # 2.因果掩码：
        #True表示允许关注：query位置i可以关注key位置j。j<=i
        #torch.tril保留矩阵的下三角部分
        causal_mask=torch.tril(
            torch.ones(
                seq_len,seq_len,
                dtype=torch.bool,
                device=x.device
            )
        )

        # 3.拆分头：
        #(..., seq_len,d_model)
        #->(..., seq_len,num_heads,head_dim)
        q=q.reshape(*q.shape[:-1],self.num_heads,self.head_dim)
        k=k.reshape(*k.shape[:-1],self.num_heads,self.head_dim)
        v=v.reshape(*v.shape[:-1],self.num_heads,self.head_dim)

        # 4.与scaled_dot_product_attention()对齐：
        #q.k.v: (..., seq_len,num_heads,head_dim)
        #->(..., num_heads,seq_len,head_dim)
        #scaled_dot_product_attention()直接处理最后两维度
        #把num_heads视为batch_size即...
        q=q.transpose(-3,-2)
        k=k.transpose(-3,-2)
        v=v.transpose(-3,-2)

        # 5.用RoPE将位置信息编码进q、k：
        #unsqueeze()函数用于在张量的指定位置添加一个大小为1的新维度
        #这个操作不会改变张量的数据，但会改变其形状
        #token_positions: (..., sequence_length)
        #-> rope_positions: (..., 1,sequence_length)
        rope_positions=token_positions.unsqueeze(-2)
        #q、k: (..., num_heads,seq_len,head_dim)
        #与rope_positions: (..., 1,sequence_length)对齐
        #以便应用RoPE类
        q=self.rope(q,rope_positions)
        k=self.rope(k,rope_positions)

        # 6.对每个头分别作attention：
        #head_outputs: (..., num_heads,seq_len,head_dim)
        #causal_mask的(seq_len,seq_len)会广播扩散到前面的...批次维和num_head维
        head_outputs=scaled_dot_product_attention(q,k,v,causal_mask)

        # 7.合并多头维度：
        #head_outputs: (..., num_heads,seq_len,head_dim)
        #->(..., seq_len,num_heads,head_dim)
        head_outputs=head_outputs.transpose(-3,-2)
        concatenated=head_outputs.reshape(
            *head_outputs.shape[:-2],
            self.d_model,
        )
        return self.o_proj(concatenated)