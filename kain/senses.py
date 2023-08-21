import torch
import torch.nn as nn
import numpy as np
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from typing import Sequence, Any, Tuple, Union

from .core import *
from .configs import Text


class SenseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        x:Union[torch.Tensor, ContextVector]
        ) -> Union[torch.Tensor, ContextVector]:
        raise NotImplementedError
    
    @torch.no_grad()
    def predict(self, x:Any) -> Union[torch.Tensor, ContextVector]:
        raise NotImplementedError
    

class TextEncoder(SenseNetwork):
    def __init__(
        self, 
        lang_tensor:torch.Tensor,
        num_vectors:int,
        num_blocks:int,
        num_features:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        self.enc_dict, _ = Text.get_character_dicts(lang_tensor)
        self.pos_enc = PositionalEncoding(num_vectors, num_features)
        self.encoder = TransformerEncoder(
            num_blocks, num_features, num_heads, num_ff_dim, dropout_rate    
        )

    @torch.no_grad()
    def encode(self, sentences:Sequence[str]) -> torch.Tensor:
        dict_length = len(self.enc_dict)
        num_vectors = self.pos_enc.pos.shape[-2]
        result_list = []
        for sent in sentences:
            vecs = [[self.enc_dict[c]/dict_length] for c in sent]
            pad_length = num_vectors - len(vecs)
            vecs += pad_length * [[self.enc_dict[Text.PAD_TOKEN]]]
            result_list.append(vecs)
        result = torch.tensor(result_list, dtype=torch.float32)
        return result
    
    def forward(self, x:torch.Tensor) -> ContextVector:
        x = self.pos_enc(x)
        x = self.encoder(x, None)
        return ContextVector(x).sum(dim=-2)
    
    @torch.no_grad()
    def predict(self, x:Sequence[str]) -> ContextVector:
        x = self.encode(x)
        x = self(x)
        return x
    

class TextDecoder(SenseNetwork):
    def __init__(
        self, 
        lang_tensor:torch.Tensor,
        num_vectors:int,
        num_blocks:int,
        num_features:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        _, self.dec_dict = Text.get_character_dicts(lang_tensor)
        self.pos_enc = PositionalEncoding(num_vectors, num_features)
        self.encoder = TransformerDecoder(
            num_blocks, num_features, num_heads, num_ff_dim, dropout_rate    
        )

    @torch.no_grad()
    def decode(self, vectors:torch.Tensor) -> Sequence[str]:
        dict_length = len(self.dec_dict)
        result_array = vectors.numpy()
        result_list = np.clip(
            np.round(result_array*dict_length), 0, dict_length-1
        ).tolist()
        result = []
        for vecs in result_list:
            vecs = [self.dec_dict[i[0]] for i in vecs]
            sent = ''.join([c for c in vecs if not c==Text.PAD_TOKEN])
            result.append(sent)
        return result
    
    def forward(self, x_src:ContextVector, x_tgt:ContextVector) -> torch.Tensor:
        x_tgt = self.pos_enc(x_tgt.tensor)
        x_tgt = self.encoder(x_src.tensor, x_tgt, None, None)
        return x_tgt
    
    @torch.no_grad()
    def predict(self, x_src:ContextVector, x_tgt:ContextVector) -> Sequence[str]:
        x = self(x_src, x_tgt)
        x = self.decode(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        side_lengths:Tuple[int, int],
        num_patches:Tuple[int, int],
        num_channels:int,
        num_blocks:int,
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        assert side_lengths[0]%num_patches[0]==0
        assert side_lengths[1]%num_patches[1]==0
        super().__init__()
        py, px = tuple(
            l//n for (l, n) in zip(side_lengths, num_patches)
        )
        num_features = num_channels * py * px
        self.patcher = nn.Unfold((py, px), stride=(py, px))
        self.pos_emb = PositionalEmbedding(
            num_patches[0]*num_patches[1], num_features
        )
        self.encoder = TransformerEncoder(
            num_blocks, num_features, num_heads, num_ff_dim, dropout_rate
        )

    def forward(self, x:torch.Tensor) -> ContextVector:
        x = self.patcher(x)
        x = x.transpose(-2, -1)
        x = self.pos_emb(x)
        x = self.encoder(x, None)
        return ContextVector(x).sum(dim=-2)
    
    @torch.no_grad()
    def predict(self, x:np.ndarray) -> ContextVector:
        x = torch.from_numpy(x).type(torch.float32).transpose(-3, -1)
        x = self(x)
        return x
    

class ImageDecoder(nn.Module):
    def __init__(
        self,
        side_lengths:Tuple[int, int],
        num_patches:Tuple[int, int],
        num_channels:int,
        num_blocks:int,
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        assert side_lengths[0]%num_patches[0]==0
        assert side_lengths[1]%num_patches[1]==0
        super().__init__()
        py, px = tuple(
            l//n for (l, n) in zip(side_lengths, num_patches)
        )
        num_features = num_channels*py*px
        self.pos_emb = PositionalEmbedding(
            num_patches[0]*num_patches[1], num_features
        )
        self.decoder = TransformerDecoder(
            num_blocks, num_features, num_heads, num_ff_dim, dropout_rate
        )
        self.depatcher = nn.Fold(side_lengths, (py, px), stride=(py, px))

    def forward(self, x_src:ContextVector, x_tgt:ContextVector) -> torch.Tensor:
        x_tgt = self.pos_emb(x_tgt.tensor)
        x = self.decoder(x_src.tensor, x_tgt, None, None)
        x = x.transpose(-2, -1)
        x = self.depatcher(x)
        return x

    @torch.no_grad()
    def predict(self, x_src:ContextVector, x_tgt:ContextVector) -> np.ndarray:
        x = self(x_src, x_tgt)
        x = x.transpose(-3, -1).numpy()
        x = np.clip(x, 0, 1)
        return x


class VoiceEncoder(SenseNetwork):
    def __init__(
        self, 
        max_length:int,
        num_blocks:int,
        num_features:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        n_fft = 2*(num_features-1)
        win_length = n_fft
        hop_length = n_fft // 2
        num_vectors = max_length / hop_length
        self.spec = Spectrogram(
            n_fft=2*(num_features-1),
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            normalized=True,
            center=True,
            pad_mode='reflect',
            onesided=True
        )
        self.pos_enc = PositionalEncoding(num_vectors, num_features)
        self.encoder = TransformerEncoder(
            num_blocks, num_features, num_heads, num_ff_dim, dropout_rate    
        )

    def forward(self, x:torch.Tensor) -> ContextVector:
        x = self.pos_enc(x)
        x = self.encoder(x, None)
        return ContextVector(x).sum(dim=-2)
    
    @torch.no_grad()
    def predict(self, x:np.ndarray) -> ContextVector:
        x = torch.from_numpy(x).type(torch.float32)
        x = x - x.mean(dim=-1, keepdims=True)
        x = self.spec(x)
        x = x.transpose(-2, -1)
        x = self(x)
        return x
    

class VoiceDecoder(SenseNetwork):
    def __init__(
        self, 
        max_length:int,
        num_blocks:int,
        num_features:int, 
        num_heads:int, 
        num_ff_dim:int,
        dropout_rate:float
        ):
        super().__init__()
        n_fft = 2*(num_features-1)
        win_length = n_fft
        hop_length = n_fft // 2
        num_vectors = max_length / hop_length
        self.pos_enc = PositionalEncoding(num_vectors, num_features)
        self.real_decoder = TransformerDecoder(
            num_blocks, num_features, num_heads, num_ff_dim, dropout_rate    
        )
        self.imag_decoder = TransformerDecoder(
            num_blocks, num_features, num_heads, num_ff_dim, dropout_rate    
        )
        self.inv_spec = InverseSpectrogram(
            n_fft=2*(num_features-1),
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            normalized=True,
            center=True,
            pad_mode='reflect',
            onesided=True
        )
        
    def forward(self, x_src:ContextVector, x_tgt:ContextVector) -> torch.Tensor:
        x_tgt = self.pos_enc(x_tgt.tensor)
        x_real = self.real_decoder(x_src.tensor, x_tgt, None, None)
        x_imag = self.imag_decoder(x_src.tensor, x_tgt, None, None)
        x = torch.stack([x_real, x_imag], dim=-1)
        return x
    
    @torch.no_grad()
    def predict(self, x_src:ContextVector, x_tgt:ContextVector) -> np.ndarray:
        x = self(x_src, x_tgt)
        x = torch.view_as_complex(x)
        x = x.transpose(-2, -1)
        x = self.inv_spec(x)
        x = x - x.mean(dim=-1, keepdims=True)
        x = x.numpy()
        return x
    