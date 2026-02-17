"""
PatchXFormer Ablation Study Variants
This file contains different variants of PatchXFormer for ablation studies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, DataEmbedding_inverted, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np
import math


class StandardPatchEmbedding(nn.Module):
    """Standard patch embedding without enhancements"""
    def __init__(self, n_vars, d_model, patch_len, stride, padding, dropout):
        super(StandardPatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        
        # Standard patch embedding
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Standard initialization (no Xavier)
        
        # Standard positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Standard normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_vars, seq_len = x.shape
        
        # Apply padding
        if self.padding:
            x = F.pad(x, (0, self.padding), mode='replicate')
        
        # Create patches
        x_patch = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        curr_patch_num = x_patch.shape[2]
        
        # Reshape
        x_patch = x_patch.reshape(-1, curr_patch_num, self.patch_len)
        
        # Embed patches
        x_embedded = self.value_embedding(x_patch)
        
        # Add positional encoding
        pos_embed = self.position_embedding(x_embedded)
        x_embedded = x_embedded + pos_embed
        
        # Reshape back
        x_embedded = x_embedded.reshape(batch_size, n_vars, curr_patch_num, -1)
        
        # NO GLOBAL TOKEN - this is the difference
        
        # Reshape for encoder
        x_final = x_embedded.reshape(-1, curr_patch_num, x_embedded.shape[-1])
        
        # Normalization and dropout
        x_final = self.layer_norm(x_final)
        x_final = self.dropout(x_final)
        
        return x_final, n_vars


class StandardAttention(nn.Module):
    """Standard self-attention without frequency enhancement"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(StandardAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Standard attention only (no frequency)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(attn_output), attn_weights


# Import from main model
from models.PatchXFormer import (
    EnhancedPatchEmbedding,
    FrequencyEnhancedAttention,
    AdaptiveNormalization,
    EnhancedHybridEncoderLayer,
    EnhancedPredictionHead
)


class StandardEncoderLayer(nn.Module):
    """Standard encoder layer without enhancements"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="gelu", use_cross_attn=False):
        super(StandardEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.use_cross_attn = use_cross_attn
        
        # Standard feed-forward
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Standard normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Standard self-attention only
        std_attn_out = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=delta)[0]
        x = x + self.dropout(std_attn_out)
        x = self.norm1(x)
        
        # Cross-attention (if enabled and available)
        if self.use_cross_attn and cross is not None:
            # Standard cross-attention on all patches (not just global token)
            cross_attn_out = self.cross_attention(x, cross, cross, 
                                                attn_mask=cross_mask, tau=tau, delta=delta)[0]
            x = x + self.dropout(cross_attn_out)
            x = self.norm2(x)
        
        # Standard feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)


class StandardPredictionHead(nn.Module):
    """Standard prediction head without enhancements"""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.target_window = target_window
        
        # Simple linear head
        self.linear = nn.Linear(nf, target_window)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear(x)
        return output


class HybridEncoderLayerVariant(nn.Module):
    """Encoder layer with configurable components"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="gelu",
                 use_freq_attention=False, use_adaptive_norm=False, use_cross_attn=False):
        super(HybridEncoderLayerVariant, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # Conditional frequency-enhanced attention
        self.use_freq_attention = use_freq_attention
        if use_freq_attention:
            self.freq_attention = FrequencyEnhancedAttention(d_model, n_heads=8, dropout=dropout)
        
        # Feed-forward
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Conditional normalization
        self.use_adaptive_norm = use_adaptive_norm
        if use_adaptive_norm:
            self.norm1 = AdaptiveNormalization(d_model)
            self.norm2 = AdaptiveNormalization(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.use_cross_attn = use_cross_attn

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Frequency-enhanced attention (if enabled)
        if self.use_freq_attention:
            freq_attn_out, _ = self.freq_attention(x, x_mask)
            x = x + self.dropout(freq_attn_out)
            x = self.norm1(x)
        
        # Standard self-attention
        std_attn_out = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=delta)[0]
        x = x + self.dropout(std_attn_out)
        x = self.norm2(x)
        
        # Cross-attention with exogenous features (if enabled)
        if self.use_cross_attn and cross is not None:
            B = cross.shape[0]
            L, D = x.shape[1], x.shape[2]
            
            # Extract global token (last token) - only if it exists
            if x.shape[1] > 1:  # Check if global token exists
                x_glb_ori = x[:, -1, :].unsqueeze(1)
                x_glb = x_glb_ori.reshape(B, -1, D)
                
                x_glb_attn = self.cross_attention(x_glb, cross, cross, 
                                                attn_mask=cross_mask, tau=tau, delta=delta)[0]
                x_glb_attn = x_glb_attn.reshape(-1, 1, D)
                
                x_glb_final = x_glb_ori + self.dropout(x_glb_attn)
                x = torch.cat([x[:, :-1, :], x_glb_final], dim=1)
        
        # Feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)


class ModelAblation(nn.Module):
    """
    PatchXFormer with configurable components for ablation study
    
    Variants:
    - 'baseline': Standard patch transformer (PatchTST-like)
    - 'enhanced_patch': + Enhanced patch embedding with global token
    - 'freq_attention': + Frequency-enhanced attention
    - 'adaptive_norm': + Adaptive normalization
    - 'cross_attn': + Cross-attention with exogenous features
    - 'enhanced_head': + Enhanced prediction head
    - 'full': All components (full PatchXFormer)
    """
    
    def __init__(self, configs, patch_len=16, stride=8, ablation_variant=None):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Get patch_len and stride from configs if available, otherwise use defaults
        self.patch_len = getattr(configs, 'patch_len', patch_len) if hasattr(configs, 'patch_len') else patch_len
        self.stride = getattr(configs, 'stride', stride) if hasattr(configs, 'stride') else stride
        self.n_vars = configs.enc_in
        
        # Get ablation variant from configs if not provided
        if ablation_variant is None:
            ablation_variant = getattr(configs, 'ablation_variant', 'full')
        self.ablation_variant = ablation_variant
        
        print(f"Initializing PatchXFormer_Ablation with variant: {self.ablation_variant}")
        
        # Determine which components to use
        self.use_enhanced_patch = ablation_variant in ['enhanced_patch', 'freq_attention', 
                                                        'adaptive_norm', 'cross_attn', 
                                                        'enhanced_head', 'full']
        self.use_freq_attention = ablation_variant in ['freq_attention', 'adaptive_norm', 
                                                       'cross_attn', 'enhanced_head', 'full']
        self.use_adaptive_norm = ablation_variant in ['adaptive_norm', 'cross_attn', 
                                                       'enhanced_head', 'full']
        self.use_cross_attn = ablation_variant in ['cross_attn', 'enhanced_head', 'full']
        self.use_enhanced_head = ablation_variant in ['enhanced_head', 'full']
        
        # Patch embedding
        padding = stride
        if self.use_enhanced_patch:
            self.patch_embedding = EnhancedPatchEmbedding(
                self.n_vars, configs.d_model, patch_len, stride, padding, configs.dropout)
        else:
            self.patch_embedding = StandardPatchEmbedding(
                self.n_vars, configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # Exogenous embedding (only if using cross-attention)
        if self.use_cross_attn:
            self.ex_embedding = DataEmbedding_inverted(
                configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Calculate patch number
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        if not self.use_enhanced_patch:
            self.patch_num = int((configs.seq_len - patch_len) / stride + 1)  # No global token
        
        # Encoder layers
        if ablation_variant == 'baseline':
            # Standard encoder layers
            self.encoder = nn.ModuleList([
                StandardEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_cross_attn=False
                ) for l in range(configs.e_layers)
            ])
        else:
            # Hybrid encoder layers with configurable components
            self.encoder = nn.ModuleList([
                HybridEncoderLayerVariant(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_freq_attention=self.use_freq_attention,
                    use_adaptive_norm=self.use_adaptive_norm,
                    use_cross_attn=self.use_cross_attn
                ) for l in range(configs.e_layers)
            ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(configs.d_model)
        
        # Prediction head
        self.head_nf = configs.d_model * self.patch_num
        
        if self.use_enhanced_head:
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                self.head = EnhancedPredictionHead(configs.enc_in, self.head_nf, configs.pred_len,
                                                 head_dropout=configs.dropout)
            else:
                self.head = EnhancedPredictionHead(configs.enc_in, self.head_nf, configs.seq_len,
                                                 head_dropout=configs.dropout)
        else:
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                self.head = StandardPredictionHead(configs.enc_in, self.head_nf, configs.pred_len,
                                                 head_dropout=configs.dropout)
            else:
                self.head = StandardPredictionHead(configs.enc_in, self.head_nf, configs.seq_len,
                                                 head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # Exogenous embedding (if using cross-attention)
        ex_embed = None
        if self.use_cross_attn and x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)
        
        # Encoder processing
        for layer in self.encoder:
            if self.use_cross_attn:
                enc_out = layer(enc_out, ex_embed)
            else:
                enc_out = layer(enc_out)
        
        # Final normalization
        enc_out = self.final_norm(enc_out)
        
        # Reshape for prediction head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # Prediction
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        # Denormalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

