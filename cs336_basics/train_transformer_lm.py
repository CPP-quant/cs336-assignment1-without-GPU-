import torch
import torch.nn as nn
from cs336_basics.train_embedding import Embedding
from cs336_basics.train_rmsnorm import RMSNorm
from cs336_basics.train_transformer_block import Transformer_block

class Transformer_LM(nn.Module):
    """
    新增Args:
        vocab_size —— 词表大小，用于确定token embedding矩阵的维度。
        context_length —— 最大上下文长度，用于确定RoPE的sin/cos buffer的维度。
        num_layers —— 使用的Transformer block数量。
    """
    def __init__(self,
        d_model,num_heads,d_ff,theta,
        vocab_size,context_length,num_layers,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.token_embeddings=Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        #设置num_layers层block
        #nn.ModuleList表示一个包含子模块的列表
        #保证每个子模块的权重保存在state_dict里
        #[]内是列表推导式
        self.layers=nn.ModuleList([
            Transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        #ln_final做最后一次LayerNorm
        self.ln_final=RMSNorm(d_model,device=device,dtype=dtype)

        #lm_head把每个位置的d_model维隐藏向量转换为vocab_size个分数
        #(vocab_size,d_model)
        self.lm_head=nn.Linear(
            d_model,
            vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self,indices):
        #in_indices: 上下文的token ID序列 (batch_size,seq_len)
        #->x: token ID编码成嵌入向量 (batch_size,seq_len,d_model)
        x=self.token_embeddings(indices)
        #过num_layers层Transformer block
        for layer in self.layers:
            x=layer(x)
        #过最后一层LayerNorm
        x=self.ln_final(x)
        #将d_model隐藏表示维度计算为logits分数
        logits=self.lm_head(x)
        #logits分数: (batch_size,seq_len,vocab_size)
        return logits