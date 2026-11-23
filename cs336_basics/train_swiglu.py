import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    """
    GLU(x,W1,W2)=σ(W1*x)⊙(W2*x)  #⊙是逐元素乘
    σ(W1*x)：门控，激活函数起到作为“开关”的作用
    (W2*x)：值，不受激活函数影响
    软开关门控的精髓：网络自己学"每个维度该放多少过去
    sigmoid：相比ReLU，不全“关”，类似调光旋钮
    防止梯度消失：求梯度后得到形如σ'+ σ的偏导数
    σ'=σ(1-σ)。如果σ'=0进入饱和区，σ=1还能防止梯度消失
    
    SwiGLU(x,W1,W2,W3)=W2*(SiLU(W1*x)⊙W3*x)
    相比GLU的门控和值投影，还多了输出投影，共3个矩阵
    
    参数计算（省略偏置层bias）：
    经典比例：d_ff=4*d_model
    FFN=2*d_model*d_ff=2*d_model*4*d_model=8d²_model
    SwiGLU=3*d_model*d_ff
    二者参数量对齐：d_ff=(8/3)*d_model
    """
    def __init__(self,d_model:int,d_ff:int | None=None,device=None,dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff=int(8/3*d_model)
            d_ff=64*((d_ff+63)//64)  #向上取整到64的倍数
        self.d_model=d_model
        self.d_ff=d_ff
        #w=nn.Linear(输入维度,输出维度)是一个线性变换
        #w(x)=x@w1.weight.T（转置）
        self.w1=nn.Linear(d_model,d_ff,bias=False,device=device,dtype=dtype)
        self.w2=nn.Linear(d_ff,d_model,bias=False,device=device,dtype=dtype)
        self.w3=nn.Linear(d_model,d_ff,bias=False,device=device,dtype=dtype)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        """SwiGLU(x,W1,W2,W3)=W2*(SiLU(W1*x)⊙W3*x)"""
        gate=nn.functional.silu(self.w1(x))
        value=self.w3(x)
        hidden=gate*value   #逐元素乘
        return self.w2(hidden)