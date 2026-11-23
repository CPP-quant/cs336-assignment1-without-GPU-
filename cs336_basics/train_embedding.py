import torch
import torch.nn as nn

class Embedding(nn.Module):
    """把整数token ID映射到维度为d_model的向量空间"""
    """每一行是一个token的词义向量"""
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Args:
            num_embeddings: int —— 词表大小
            embedding_dim: int —— embedding 向量的维度，即d_model
            device: torch.device | None = None —— 存储参数的设备
            dtype: torch.dtype | None = None —— 参数的数据类型
        """
        #super().__init__()调用父类的初始化方法。父类superclass。
        super().__init__()
        #把传入的局部变量绑定到对象自身上->方便外部查看
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim

        #初始化权重矩阵W
        #torch.empty() 表示向内存申请一块矩阵空间（随机数据）
        #torch.nn.Parameter() 会让PyTorch将这个矩阵识别为可训练参数：可以自动求导和注册
        self.weight=nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        #trunc_normal_ 是指截断正态分布
        #它会按照正态分布随机生成数据填入W中，但会“掐掉”两端概率极低的离群值
        #std：Standard Deviation
        #正态分布来设置矩阵参数：便于后续反向传播
        torch.nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        通过索引一个形状为(vocab_size,d_model)的embedding矩阵，
        为每个token ID选出对应的embedding行向量
        Args:
            形状为(batch_size, sequence_length)的token ID的torch.LongTensor
        Returns:
            形状为(batch_size, sequence_length, embedding_dim)的token embedding词义张量
        """
        #输出的最后一维度为embedding_dim，即嵌入向量的维数
        return self.weight[token_ids]