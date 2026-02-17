import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np


class EnhancedPatchEmbedding(nn.Module):
    """Enhanced patch embedding with improved features but simpler architecture"""
    def __init__(self, n_vars, d_model, patch_len, stride, padding, dropout):
        super(EnhancedPatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        
        # Enhanced patch embedding with better initialization
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        nn.init.xavier_uniform_(self.value_embedding.weight)
        
        # Enhanced global token
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        nn.init.xavier_uniform_(self.glb_token)
        
        # Enhanced positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Improved normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [bs, nvars, seq_len]
        batch_size, n_vars, seq_len = x.shape
        
        # Apply padding
        if self.padding:
            x = F.pad(x, (0, self.padding), mode='replicate')
        
        # Create patches with unfold
        x_patch = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x_patch: [bs, nvars, patch_num, patch_len]
        
        curr_patch_num = x_patch.shape[2]
        
        # Reshape for processing
        x_patch = x_patch.reshape(-1, curr_patch_num, self.patch_len)
        
        # Apply enhanced patch embedding
        x_embedded = self.value_embedding(x_patch)  # [bs*nvars, patch_num, d_model]
        
        # Add positional encoding
        pos_embed = self.position_embedding(x_embedded)
        x_embedded = x_embedded + pos_embed
        
        # Reshape back
        x_embedded = x_embedded.reshape(batch_size, n_vars, curr_patch_num, -1)
        
        # Add enhanced global token
        glb = self.glb_token.repeat(batch_size, 1, 1, 1)
        x_embedded = torch.cat([x_embedded, glb], dim=2)  # [bs, nvars, patch_num+1, d_model]
        
        # Reshape for encoder
        x_final = x_embedded.reshape(-1, curr_patch_num + 1, x_embedded.shape[-1])
        
        # Apply enhanced normalization and dropout
        x_final = self.layer_norm(x_final)
        x_final = self.dropout(x_final)
        
        return x_final, n_vars


class Model(nn.Module):
    """
    PatchXFormer Variant 1: +Enhanced Patch Embedding
    
    This variant adds Enhanced Patch Embedding:
    - EnhancedPatchEmbedding (with global token, enhanced initialization, positional embedding)
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
        
        # Enhanced patch embedding (with global token)
        padding = stride
        self.patch_embedding = EnhancedPatchEmbedding(
            self.n_vars, configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # Exogenous embedding (kept for consistency, but not used in cross-attention)
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Calculate patch number (base calculation, +1 for global token added in EnhancedPatchEmbedding)
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
        # Note: EnhancedPatchEmbedding adds +1 for global token, so we use patch_num + 1
        self.head_nf = configs.d_model * (self.patch_num + 1)
        
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
        
        # Enhanced patch embedding (with global token)
        x_enc = x_enc.permute(0, 2, 1)  # [bs, nvars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [bs*nvars, patch_num+1, d_model]
        
        # Exogenous embedding (not used in cross-attention for this variant)
        # Kept for consistency but not passed to encoder
        
        # Standard encoder processing (no cross-attention)
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)
        
        # Final normalization
        enc_out = self.final_norm(enc_out)
        
        # Reshape for prediction head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num+1]
        
        # Simple linear prediction
        enc_out_flat = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1)  # [bs, nvars, d_model*(patch_num+1)]
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

