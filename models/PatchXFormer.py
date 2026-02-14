import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, DataEmbedding_inverted, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np
import math


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


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


class FrequencyEnhancedAttention(nn.Module):
    """Simplified frequency-enhanced attention"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(FrequencyEnhancedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Standard attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Simplified frequency components
        self.freq_weight = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Standard attention computation
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Standard attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Simple frequency enhancement
        if seq_len > 4:  # Only apply FFT if sequence is long enough
            try:
                x_freq = torch.fft.rfft(x, dim=1)
                x_freq_real = torch.fft.irfft(x_freq, n=seq_len, dim=1)
                freq_enhanced = attn_output + self.freq_weight * x_freq_real
            except:
                freq_enhanced = attn_output
        else:
            freq_enhanced = attn_output
        
        return self.out_proj(freq_enhanced), attn_weights


class AdaptiveNormalization(nn.Module):
    """Simplified adaptive normalization"""
    def __init__(self, d_model, eps=1e-5):
        super(AdaptiveNormalization, self).__init__()
        self.eps = eps
        self.d_model = d_model
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        # Simplified adaptation
        self.adaptation_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Standard normalization
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        
        # Apply learnable parameters with adaptation
        adaptation = torch.tanh(self.adaptation_weight)
        adaptive_alpha = self.alpha * (1 + 0.1 * adaptation)
        adaptive_beta = self.beta + 0.1 * adaptation
        
        return normalized * adaptive_alpha + adaptive_beta


class EnhancedHybridEncoderLayer(nn.Module):
    """Simplified but enhanced encoder layer"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="gelu"):
        super(EnhancedHybridEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # Frequency-enhanced attention
        self.freq_attention = FrequencyEnhancedAttention(d_model, n_heads=8, dropout=dropout)
        
        # Enhanced feed-forward
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Improved normalization
        self.norm1 = AdaptiveNormalization(d_model)
        self.norm2 = AdaptiveNormalization(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Enhanced self-attention with frequency components
        freq_attn_out, _ = self.freq_attention(x, x_mask)
        x = x + self.dropout(freq_attn_out)
        x = self.norm1(x)
        
        # Standard self-attention
        std_attn_out = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=delta)[0]
        x = x + self.dropout(std_attn_out)
        x = self.norm2(x)
        
        # Cross-attention with exogenous features (if available)
        if cross is not None:
            B = cross.shape[0]
            L, D = x.shape[1], x.shape[2]
            
            # Extract global token (last token)
            x_glb_ori = x[:, -1, :].unsqueeze(1)  # [bs*nvars, 1, d_model]
            x_glb = x_glb_ori.reshape(B, -1, D)  # [bs, nvars, d_model]
            
            # Cross-attention with exogenous features
            x_glb_attn = self.cross_attention(x_glb, cross, cross, 
                                            attn_mask=cross_mask, tau=tau, delta=delta)[0]
            x_glb_attn = x_glb_attn.reshape(-1, 1, D)  # [bs*nvars, 1, d_model]
            
            # Update global token
            x_glb_final = x_glb_ori + self.dropout(x_glb_attn)
            
            # Replace global token in sequence
            x = torch.cat([x[:, :-1, :], x_glb_final], dim=1)
        
        # Enhanced feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)


class EnhancedPredictionHead(nn.Module):
    """Enhanced prediction head with improved architecture"""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.target_window = target_window
        
        # Enhanced prediction with residual connection
        self.main_path = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(nf // 2, target_window)
        )
        
        # Residual path for better gradient flow
        self.residual_path = nn.Linear(nf, target_window)
        
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        
        # Main prediction path
        main_out = self.main_path(x)
        
        # Residual path
        residual_out = self.residual_path(x)
        
        # Combine with residual connection
        output = main_out + 0.1 * residual_out
        
        return output


class Model(nn.Module):
    """
    PatchXFormer: Simplified but Enhanced Version
    
    Key improvements:
    1. Enhanced patch embedding with better initialization
    2. Frequency-enhanced attention (simplified)
    3. Adaptive normalization
    4. Improved prediction head with residual connections
    5. Better gradient flow and stability
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_vars = configs.enc_in
        
        # Enhanced patch embedding
        padding = stride
        self.patch_embedding = EnhancedPatchEmbedding(
            self.n_vars, configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # Exogenous embedding
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Calculate patch number
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        
        # Enhanced encoder
        self.encoder = nn.ModuleList([
            EnhancedHybridEncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(configs.d_model)
        
        # Enhanced prediction head
        self.head_nf = configs.d_model * (self.patch_num + 1)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = EnhancedPredictionHead(configs.enc_in, self.head_nf, configs.pred_len,
                                             head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = EnhancedPredictionHead(configs.enc_in, self.head_nf, configs.seq_len,
                                             head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Enhanced normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Enhanced patch embedding
        x_enc = x_enc.permute(0, 2, 1)  # [bs, nvars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [bs*nvars, patch_num+1, d_model]
        
        # Exogenous embedding
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)  # [bs, seq_len, d_model]
        
        # Enhanced encoder processing
        for layer in self.encoder:
            enc_out = layer(enc_out, ex_embed)
        
        # Final normalization
        enc_out = self.final_norm(enc_out)
        
        # Reshape for prediction head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num+1]
        
        # Enhanced prediction
        dec_out = self.head(enc_out)  # [bs, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs, pred_len, nvars]
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Enhanced normalization for imputation
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
        
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)
        
        for layer in self.encoder:
            enc_out = layer(enc_out, ex_embed)
        
        enc_out = self.final_norm(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def anomaly_detection(self, x_enc):
        # Same enhancement as forecast
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        for layer in self.encoder:
            enc_out = layer(enc_out)
        
        enc_out = self.final_norm(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)
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
        
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)
        
        for layer in self.encoder:
            enc_out = layer(enc_out, ex_embed)
        
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