import torch
import torch.nn as nn
from typing import Sequence, Any

from .core import PositionalEncoding, TransformerEncoder, TransformerDecoder
from .utils import round_within_range
from train.configs import Text


class SenseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def predict(self, x:Any) -> Any:
        raise NotImplementedError
    

class TextEncoder(SenseNetwork):
    def __init__(
        self, 
        langs:Sequence[str],
        max_length:int,
        num_blocks:int,
        num_latent_dim:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        self.max_length = max_length
        self.enc_dict, _ = Text.get_character_dicts(langs)
        self.pos_enc = PositionalEncoding(max_length, num_latent_dim)
        self.encoder = TransformerEncoder(
            num_blocks, num_latent_dim, num_heads, num_ff_dim, dropout_rate    
        )

    @torch.no_grad()
    def encode(self, sentences:Sequence[str]) -> torch.Tensor:
        dict_length = len(self.enc_dict)
        result_list = []
        for sent in sentences:
            vec = [[self.enc_dict[c]/dict_length] for c in sent]
            pad_length = self.max_length - len(vec)
            vec += pad_length * [[self.enc_dict[Text.PAD_TOKEN]]]
            result_list.append(vec)
        result = torch.tensor(result_list, dtype=torch.float32)
        return result
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        x = self.encoder(x, None)
        return x
    
    @torch.no_grad()
    def predict(self, x:Sequence[str]) -> torch.Tensor:
        x = self.encode(x)
        x = self(x)
        return x
    

class TextDecoder(SenseNetwork):
    def __init__(
        self, 
        langs:Sequence[str],
        max_length:int,
        num_blocks:int,
        num_latent_dim:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        self.max_length = max_length
        _, self.dec_dict = Text.get_character_dicts(langs)
        self.pos_enc = PositionalEncoding(max_length, num_latent_dim)
        self.encoder = TransformerDecoder(
            num_blocks, num_latent_dim, num_heads, num_ff_dim, dropout_rate    
        )

    @torch.no_grad()
    def decode(self, vectors:torch.Tensor) -> Sequence[str]:
        dict_length = len(self.dec_dict)
        result_list = vectors.tolist()
        result = []
        for vec in result_list:
            vec = [self.dec_dict[
                round_within_range(i[0]*dict_length, 0, dict_length-1)
            ] for i in vec]
            sent = ''.join([c for c in vec if not c==Text.PAD_TOKEN])
            result.append(sent)
        return result
    
    def forward(self, x_src:torch.Tensor, x_tgt:torch.Tensor) -> torch.Tensor:
        x_tgt = self.pos_enc(x_tgt)
        x_tgt = self.encoder(x_src, x_tgt, None, None)
        return x_tgt
    
    @torch.no_grad()
    def predict(self, x_src:torch.Tensor, x_tgt:torch.Tensor) -> Sequence[str]:
        x = self(x_src, x_tgt)
        x = self.decode(x)
        return x


class ImageEncoder(nn.Module):
    pass


class ImageDecoder(nn.Module):
    pass


class VoiceEncoder(nn.Module):
    pass


class VoiceDecoder(nn.Module):
    pass