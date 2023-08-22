from typing import Any, Mapping, Union
from .senses import *
from .configs import *
from .utils import join_parameters
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


class Kain(nn.Module):
    def __init__(self, lang_indices:Union[torch.Tensor, None]=None):
        super().__init__()
        lang_tensor = torch.zeros((len(Text.UNICODES),), dtype=torch.bool)
        if lang_indices is None:
            lang_tensor[0] = True
        else:
            lang_tensor[lang_indices] = True
        self.register_buffer('lang_tensor', lang_tensor, persistent=True)
        self.memory = []
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

        self.opt_ti = optim.Adam(
            join_parameters(
                self.text_encoder, 
                self.image_encoder, 
                self.text_decoder, 
                self.image_decoder
            ),
            lr=General.LEARNING_RATE
        )
        self.opt_iv = optim.Adam(
            join_parameters(
                self.image_encoder, 
                self.voice_encoder, 
                self.image_decoder,
                self.voice_decoder
            ),
            lr=General.LEARNING_RATE
        )
        self.opt_vt = optim.Adam(
            join_parameters(
                self.voice_encoder, 
                self.text_encoder, 
                self.voice_decoder,
                self.text_decoder
            ),
            lr=General.LEARNING_RATE
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
    
    def load_memory(self, csv_path:str):
        return self
    
    def save_all(self) -> None:
        return
    
    @staticmethod
    def fit_pairwise(
        loader_12:Dataset,
        enc_1:nn.Module,
        enc_2:nn.Module,
        dec_1:nn.Module,
        dec_2:nn.Module,
        optimizer:optim.Optimizer
        ) -> np.ndarray:
        total_losses = np.zeros((3,), dtype=float)
        for x_1, x_2 in loader_12:
            optimizer.zero_grad()
            c_1 = enc_1(x_1)
            c_2 = enc_2(x_2)
            y_1 = dec_1(c_1)
            y_2 = dec_2(c_2)
            ae_loss_1 = nn.functional.mse_loss(y_1, x_1)
            ae_loss_2 = nn.functional.mse_loss(y_2, x_2)
            ctx_loss = nn.functional.mse_loss(c_1, c_2)
            pairwise_loss = ae_loss_1 + ae_loss_2 + ctx_loss
            pairwise_loss.backward()
            optimizer.step()
            total_losses += np.array(
                [ae_loss_1.item(), ae_loss_2.item(), ctx_loss.item()],
                dtype=float
            )
        return total_losses / len(loader_12)
    
    @staticmethod
    def make_dataloaders(
        dataset:Dataset,
        lengths:Tuple[int, int, int],
        batch_size:int
        ) -> DataLoader:
        assert sum(lengths)==1
        datasets = random_split(dataset, lengths)
        return tuple(DataLoader(d, batch_size=batch_size) for d in datasets)
    
    def fit(
        self,
        dataset_ti:Dataset,
        dataset_iv:Dataset,
        dataset_vt:Dataset,
        lengths:Tuple[int, int, int],
        batch_size:int,
        max_epochs:int
        ):
        loaders_ti = Kain.make_dataloaders(dataset_ti, lengths, batch_size)
        loaders_iv = Kain.make_dataloaders(dataset_iv, lengths, batch_size)
        loaders_vt = Kain.make_dataloaders(dataset_vt, lengths, batch_size)
        ti_losses = np.nan * np.ones((3,), dtype=float)
        iv_losses = np.nan * np.ones((3,), dtype=float)
        vt_losses = np.nan * np.ones((3,), dtype=float)
        set_desc = lambda: pbar.set_description(
            f"""
            | PAIR\t| AE1\t| AE2\t| CTX\t|\n
            | T/I\t| {ti_losses[0]}\t| {ti_losses[1]}\t| {ti_losses[2]}\t|\n
            | I/V\t| {iv_losses[0]}\t| {iv_losses[1]}\t| {iv_losses[2]}\t|\n
            | V/T\t| {vt_losses[0]}\t| {vt_losses[1]}\t| {vt_losses[2]}\t|\n
            """
        )
        pbar = tqdm(range(max_epochs))
        for _ in pbar:
            ti_losses = Kain.fit_pairwise(
                loaders_ti[0], self.text_encoder, self.image_encoder,
                self.text_decoder, self.image_decoder, self.opt_ti
            )
            set_desc()
            iv_losses = Kain.fit_pairwise(
                loaders_iv[0], self.image_encoder, self.voice_encoder,
                self.image_decoder, self.voice_decoder, self.opt_iv
            )
            set_desc()
            vt_losses = Kain.fit_pairwise(
                loaders_vt[0], self.voice_encoder, self.text_encoder,
                self.voice_decoder, self.text_decoder, self.opt_vt
            )
            set_desc()
