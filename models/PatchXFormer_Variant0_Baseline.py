import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np


class Model(nn.Module):
    """
    PatchXFormer Variant 0: Baseline Model
    
    This is the baseline model with NO enhancements:
    - Standard PatchEmbedding (no global token, no enhanced initialization)
    - Standard EncoderLayer (no frequency attention, no adaptive norm)
    - Standard FullAttention only (no frequency enhancement)
    - Standard LayerNorm (no adaptive normalization)
    - Simple Linear Prediction Head (no residual connections)
    - No cross-attention with exogenous features
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_vars = configs.enc_in
        
        # Standard patch embedding (no enhancements)
        padding = stride
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # Exogenous embedding (kept for consistency, but not used in cross-attention)
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Calculate patch number
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        
        # Standard encoder layers (no enhancements)
        self.encoder = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)
        ])
        
        # Final normalization (standard LayerNorm)
        self.final_norm = nn.LayerNorm(configs.d_model)
        
        # Simple linear prediction head (no residual connections)
        self.head_nf = configs.d_model * self.patch_num  # No +1 since no global token
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = nn.Linear(self.head_nf, configs.pred_len)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = nn.Linear(self.head_nf, configs.seq_len)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Standard normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Standard patch embedding
        x_enc = x_enc.permute(0, 2, 1)  # [bs, nvars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [bs*nvars, patch_num, d_model]
        
        # Exogenous embedding (not used in cross-attention for baseline)
        # Kept for consistency but not passed to encoder
        
        # Standard encoder processing (no cross-attention)
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)
        
        # Final normalization
        enc_out = self.final_norm(enc_out)
        
        # Reshape for prediction head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num]
        
        # Simple linear prediction
        enc_out_flat = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1)  # [bs, nvars, d_model*patch_num]
        dec_out = self.head(enc_out_flat)  # [bs, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs, pred_len, nvars]
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Standard normalization for imputation
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        
        # Same processing as forecast
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)
        
        enc_out = self.final_norm(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        enc_out_flat = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1)
        dec_out = self.head(enc_out_flat)
        dec_out = dec_out.permute(0, 2, 1)
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def anomaly_detection(self, x_enc):
        # Same as forecast
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)
        
        enc_out = self.final_norm(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        enc_out_flat = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1)
        dec_out = self.head(enc_out_flat)
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)
        
        enc_out = self.final_norm(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

