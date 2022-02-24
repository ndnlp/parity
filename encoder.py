import torch
import math

class SigmoidAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SigmoidAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj_weight = torch.nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = torch.nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.in_proj_weight)
        torch.nn.init.constant_(self.in_proj_bias, 0.)
        torch.nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value):
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        q = torch.nn.functional.linear(query, w_q, b_q)
        k = torch.nn.functional.linear(key, w_k, b_k)
        v = torch.nn.functional.linear(value, w_v, b_v)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        q = q / math.sqrt(embed_dim)
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = torch.sigmoid(attn)
        attn_output = torch.bmm(attn, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_output = torch.nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        return attn_output, attn

class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.last_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class ScaledTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.last_weights = self.self_attn(
            src*math.log(len(src)),
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
