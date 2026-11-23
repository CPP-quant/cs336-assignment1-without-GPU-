import torch
import math
from cs336_basics.train_softmax import softmax

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    attention=softmax(Q*K.T/√d_k+mask)*V
    Args:
        Q.shape=float(..., queries,d_k)
        K.shape=float(..., keys,d_k)
        V.shape=float(..., keys,d_v)
        mask.shape=Bool(..., queries,keys)
    Returns:
        output.shape=float(..., queries,d_v)
    """
    d_k=Q.shape[-1]
    scores=Q @ K.transpose(-2,-1)
    scores/=math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(~mask,float("-inf"))
    weights=softmax(scores,dim=-1)
    return weights@V

    

