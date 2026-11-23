import torch
import torch.nn as nn   #neural network模块

class Linear(nn.Module):
    """构造一个线性变换模块，接口与PyTorch内置正版nn.Linear模块相同，除了bias参数（仿射变换）"""
    #类继承：class 子类名(父类)
    #一个新类"基于"一个已有的类来定义，自动获得已有类的属性和方法，再按需增加或修改
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Arg:
            in_features: int —— 输入的最后一个维度。
            out_features: int —— 输出的最后一个维度。
            device: torch.device | None = None —— 存储参数的设备。
            dtype: torch.dtype | None = None —— 参数的数据类型。
        """

        #super().__init__()调用父类的初始化方法。父类superclass。
        super().__init__()
        #把传入的局部变量绑定到对象自身上->方便外部查看
        self.in_features=in_features
        self.out_features=out_features

        #初始化权重矩阵W
        #torch.empty() 表示向内存申请一块矩阵空间（随机数据）
        #torch.nn.Parameter() 会让PyTorch将这个矩阵识别为可训练参数：可以自动求导和注册
        self.W=nn.Parameter(torch.empty(in_features,out_features,device=device,dtype=dtype))

        #trunc_normal_ 是指截断正态分布
        #它会按照正态分布随机生成数据填入W中，但会“掐掉”两端概率极低的离群值
        #std：Standard Deviation
        #正态分布来设置矩阵参数：便于后续反向传播
        torch.nn.init.trunc_normal_(self.W,std=0.02)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """矩阵乘法：对输入应用线性变换"""
        return x @ self.W
