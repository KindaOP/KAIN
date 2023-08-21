import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple


class ContextVector:
    def __normalize(self):
        n = torch.sqrt(torch.sum(self.tensor**2, dim=-1, keepdim=True))
        self.tensor = torch.where(n!=0, self.tensor/n, 0)
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(torch.tensor(*args, **kwargs))

    def __init__(self, tensor:torch.Tensor):
        self.tensor = tensor.type(torch.float32)
        self.__normalize()

    def __add__(self, tensor:torch.Tensor):
        return self.__class__(self.tensor+tensor.tensor)

    def __iadd__(self, tensor:torch.Tensor):
        self.tensor += tensor.tensor
        self.__normalize()
        return self
    
    def __mul__(self, tensor:torch.Tensor):
        return self.__class__(self.tensor*tensor)
    
    def __imul__(self, tensor:torch.Tensor):
        self.tensor *= tensor
        self.__normalize()
        return self

    def __repr__(self):
        return f"context_{self.tensor}"
    
    def __getitem__(self, x):
        return self.tensor[x]
    
    def sum(self, dim:int=-2) -> torch.Tensor:
        self.tensor = self.tensor.sum(dim=dim, keepdim=True)
        self.__normalize()
        return self
    

class PositionalEncoding(nn.Module):
    def __init__(self, num_vectors:int, num_features:int, n:int=10000):
        assert num_features%2==0
        super().__init__()
        v_ids = torch.arange(0, num_vectors, dtype=torch.float32)
        f_ids = torch.arange(0, num_features//2, dtype=torch.float32)
        v, f = torch.meshgrid(v_ids, f_ids, indexing='ij')
        phases = v / n**(2*f/num_features)
        p_sin = torch.sin(phases)
        p_cos = torch.cos(phases)
        pos = torch.stack([p_sin, p_cos], dim=-1)
        pos = pos.reshape(pos.shape[0], -1)
        self.register_buffer('pos', pos.unsqueeze(dim=0))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pos
        return x
    

class PositionalEmbedding(nn.Module):
    def __init__(self, num_positions:int, num_features:int):
        super().__init__()
        pos = torch.arange(num_positions)
        self.register_buffer("pos", pos.unsqueeze(dim=0))
        self.pos_emb = nn.Embedding(num_positions, num_features, max_norm=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pos_emb(self.pos)
        return x


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, num_features:int, num_heads:int):
        assert num_features%num_heads==0
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.head_size = num_features // num_heads

        self.fc_Q = nn.Linear(num_features, num_features)
        self.fc_K = nn.Linear(num_features, num_features)
        self.fc_V = nn.Linear(num_features, num_features)
        self.fc_out = nn.Linear(num_features, num_features)

    def split_into_heads(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, num_vectors = x.shape[:2]
        x = x.reshape(batch_size, num_vectors, self.num_heads, self.head_size)
        x = x.transpose(1, 2)
        return x
    
    def combine_heads(self, x:torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        batch_size, num_vectors = x.shape[:2]
        x = x.reshape(batch_size, num_vectors, self.num_features)
        return x

    def forward(
        self, 
        Q:torch.Tensor, 
        K:torch.Tensor, 
        V:torch.Tensor,
        mask:Union[torch.Tensor, None]
        ) -> torch.Tensor:
        Q = self.fc_Q(Q)
        Q = self.split_into_heads(Q)
        K = self.fc_K(K)
        K = self.split_into_heads(K)
        V = self.fc_V(V)
        V = self.split_into_heads(V)
        x = Q@K.transpose(-2, -1) / np.sqrt(self.num_heads)
        if mask is not None:
            x = torch.masked_fill(x, mask, np.nan)
        x = torch.softmax(x, dim=-1) @ V
        x = self.combine_heads(x)
        x = self.fc_out(x)
        return x
    

class FeedForwardLayer(nn.Module):
    def __init__(self, num_features:int, num_hidden_features:int):
        super().__init__()
        self.fc_in = nn.Linear(num_features, num_hidden_features)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(num_hidden_features, num_features)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x
    

class AddNormLayer(nn.Module):
    def __init__(self, num_features:int, dropout_rate:float):
        assert 0<=dropout_rate<=1
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(y)
        x = self.layer_norm(x)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(
        self, 
        num_features:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        self.self_attention = MultiheadAttentionLayer(num_features, num_heads)
        self.feed_forward = FeedForwardLayer(num_features, num_ff_dim)
        self.addnorm_sa = AddNormLayer(num_features, dropout_rate)
        self.addnorm_ff = AddNormLayer(num_features, dropout_rate)

    def forward(
        self, x:torch.Tensor, mask:Union[torch.Tensor, None]
        ) -> torch.Tensor:
        y = self.self_attention(x, x, x, mask)
        x = self.addnorm_sa(x, y)
        y = self.feed_forward(x)
        x = self.addnorm_ff(x, y)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, 
        num_features:int,
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        self.self_attention = MultiheadAttentionLayer(num_features, num_heads)
        self.cross_attention = MultiheadAttentionLayer(num_features, num_heads)
        self.feed_forward = FeedForwardLayer(num_features, num_ff_dim)
        self.addnorm_sa = AddNormLayer(num_features, dropout_rate)
        self.addnorm_ca = AddNormLayer(num_features, dropout_rate)
        self.addnorm_ff = AddNormLayer(num_features, dropout_rate)

    def forward(
        self, 
        x_src:torch.Tensor, 
        x_tgt:torch.Tensor, 
        mask_src:Union[torch.Tensor, None],
        mask_tgt:Union[torch.Tensor, None]
        ):
        y = self.self_attention(x_tgt, x_tgt, x_tgt, mask_tgt)
        x = self.addnorm_sa(x_tgt, y)
        y = self.cross_attention(x_tgt, x_src, x_src, mask_src)
        x = self.addnorm_ca(x, y)
        y = self.feed_forward(x)
        x = self.addnorm_ff(x, y)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_blocks:int,
        num_features:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            setattr(
                self,
                f'enc_{i}', 
                EncoderBlock(
                    num_features, num_heads, num_ff_dim, dropout_rate
                )
            )

    def forward(
        self, 
        x:torch.Tensor, 
        mask:Union[torch.Tensor, None]
        ) -> torch.Tensor:
        for i in range(self.num_blocks):
            enc_block = getattr(self, f'enc_{i}')
            x = enc_block(x, mask)
        return x
    

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        num_blocks:int,
        num_features:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            setattr(
                self,
                f'dec_{i}', 
                DecoderBlock(
                    num_features, num_heads, num_ff_dim, dropout_rate
                )
            )

    def forward(
        self, 
        x_src:torch.Tensor, 
        x_tgt:torch.Tensor,
        mask_src:Union[torch.Tensor, None],
        mask_tgt:Union[torch.Tensor, None]
        ) -> torch.Tensor:
        for i in range(self.num_blocks):
            dec_block = getattr(self, f'dec_{i}')
            x_tgt = dec_block(x_src, x_tgt, mask_src, mask_tgt)
        return x_tgt
