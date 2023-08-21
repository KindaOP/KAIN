from typing import Any, Mapping, Union
from .senses import *
from .configs import *
import torch.nn as nn


class Kain(nn.Module):
    def __init__(self, lang_indices:Union[torch.Tensor, None]):
        super().__init__()
        lang_tensor = torch.zeros((len(Text.UNICODES),), dtype=torch.bool)
        if lang_indices is None:
            lang_indices[0] = True
        else:
            lang_tensor[lang_indices] = True
        self.register_buffer('lang_tensor', lang_tensor, persistent=True)
        self.long_term_memory = []
        self.short_term_memory = []
        self.text_encoder = TextEncoder(
            lang_tensor=self.lang_tensor,
            num_vectors=Text.MAX_CHARS_PER_SENT,
            num_blocks=General.NUM_BLOCKS_PER_STACK,
            num_features=General.NUM_FEATURES, 
            num_heads=General.NUM_HEADS_PER_BLOCK, 
            num_ff_dim=General.NUM_FEEDFORWARD_DIM,
            dropout_rate=General.ADDNORM_DROPOUT_RATE
        )
        self.text_decoder = TextDecoder(
            lang_tensor=self.lang_tensor,
            num_vectors=Text.MAX_CHARS_PER_SENT,
            num_blocks=General.NUM_BLOCKS_PER_STACK,
            num_features=General.NUM_FEATURES, 
            num_heads=General.NUM_HEADS_PER_BLOCK, 
            num_ff_dim=General.NUM_FEEDFORWARD_DIM,
            dropout_rate=General.ADDNORM_DROPOUT_RATE
        )
        self.image_encoder = ImageEncoder(
            side_lengths=Image.IMAGE_SHAPE,
            num_patches=Image.NUM_PATCHES,
            num_channels=Image.NUM_CHANNELS,
            num_blocks=General.NUM_BLOCKS_PER_STACK,
            num_heads=General.NUM_HEADS_PER_BLOCK, 
            num_ff_dim=General.NUM_FEEDFORWARD_DIM,
            dropout_rate=General.ADDNORM_DROPOUT_RATE
        )
        self.image_decoder = ImageDecoder(
            side_lengths=Image.IMAGE_SHAPE,
            num_patches=Image.NUM_PATCHES,
            num_channels=Image.NUM_CHANNELS,
            num_blocks=General.NUM_BLOCKS_PER_STACK,
            num_heads=General.NUM_HEADS_PER_BLOCK, 
            num_ff_dim=General.NUM_FEEDFORWARD_DIM,
            dropout_rate=General.ADDNORM_DROPOUT_RATE
        )
        self.voice_encoder = VoiceEncoder(
            max_length=Voice.MAX_SIGNAL_LENGTH,
            num_blocks=General.NUM_BLOCKS_PER_STACK,
            num_features=General.NUM_FEATURES, 
            num_heads=General.NUM_HEADS_PER_BLOCK, 
            num_ff_dim=General.NUM_FEEDFORWARD_DIM,
            dropout_rate=General.ADDNORM_DROPOUT_RATE
        )
        self.voice_decoder = VoiceDecoder(
            max_length=Voice.MAX_SIGNAL_LENGTH,
            num_blocks=General.NUM_BLOCKS_PER_STACK,
            num_features=General.NUM_FEATURES, 
            num_heads=General.NUM_HEADS_PER_BLOCK, 
            num_ff_dim=General.NUM_FEEDFORWARD_DIM,
            dropout_rate=General.ADDNORM_DROPOUT_RATE
        )

    def set_languages(self, lang_indices:torch.Tensor):
        lang_tensor = torch.zeros((len(Text.UNICODES),), dtype=torch.bool)
        lang_tensor[lang_indices] = True
        enc_dict, dec_dict = Text.get_character_dicts(lang_tensor)
        self.text_encoder.enc_dict = enc_dict
        self.text_decoder.dec_dict = dec_dict
        self.lang_tensor = lang_tensor
        return self

    def load_state_dict(self, state_dict:Mapping[str, Any], strict:bool=True):
        model = super().load_state_dict(state_dict, strict)
        enc_dict, dec_dict = Text.get_character_dicts(self.lang_tensor)
        self.text_encoder.enc_dict = enc_dict
        self.text_decoder.dec_dict = dec_dict
        return model