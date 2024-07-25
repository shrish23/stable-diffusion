import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self,n_heads: int, d_embed: int, in_proj_bias= True, out_proj_bias= True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias= in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias= out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, casual_mask=False):
        # x: (Batch_size, seq_len, dim)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # Projecting the input to the required shape: (Batch_size, seq_len, 3 * d_embed)
        q,k,v = self.in_proj(x).chunk(3, dim=-1)

        # Reshaping the projected input to the required shape: (Batch_size, seq_len, n_heads, d_head)
        # Transposing the reshaped input to the required shape: (Batch_size, n_heads, seq_len, d_head)
        # This is done to perform the attention operation on the last two dimensions
        # (Batch_size, seq_len, dim) -> (Batch_size, n_heads, seq_len, dim / n_heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Performing the scaled dot product attention
        # q: (Batch_size, n_heads, seq_len, d_head)
        # k: (Batch_size, n_heads, d_head, seq_len)
        # v: (Batch_size, n_heads, seq_len, d_head)
        # attention: (Batch_size, n_heads, seq_len, seq_len)

        weight = q @ k.transpose(-1,-2)

        if casual_mask:
            # Mask where the upper triangle (including the diagonal) is zero
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_size, n_heads, seq_len, seq_len) @ (Batch_size, n_heads, seq_len, d_head) -> (Batch_size, n_heads, seq_len, d_head)

        output = weight @ v
        # (Batch_size, n_heads, seq_len, d_head) -> (Batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (Batch_size, seq_len, dim) -> (Batch_size, seq_len, dim)
        return output
    

class CrossAttention(nn.Module):

    def __init__(self,n_head: int, d_embd: int, d_cross: int, in_proj_bias= True, out_proj_bias= True):
        super().__init__()

        self.q_proj = nn.Linear(d_embd, d_embd, bias= in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embd, bias= in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embd, bias= in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias= out_proj_bias)# output projection
        self.n_head = n_head
        self.d_head = d_embd // n_head # the attention each head can pay to the input

    def forward(self, x, y):
        #x: (latent): (Batch_size, seq_len, dim)
        #y: (context): (Batch_size, seq_len, dim) = (batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embd = input_shape

        interim_shape = (batch_size, -1, self.n_head, self.d_head)# (batch_size, 77, 12, 64)

        # Multiply the latent and context by the projection matrices
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # split into n_heads
        q = q.view(interim_shape).transpose(1, 2)# (batch_size, 12, 77, 64)
        k = k.view(interim_shape).transpose(1, 2)# (batch_size, 12, 77, 64)
        v = v.view(interim_shape).transpose(1, 2)# (batch_size, 12, 77, 64)

        weight = q @ k.transpose(-1,-2)# (batch_size, 12, 77, 77)

        weight /= math.sqrt(self.d_head)#scaling

        weight = F.softmax(weight, dim=-1)# softmax: the sum of the weights is 1

        output = weight @ v# (batch_size, 12, 77, 77) @ (batch_size, 12, 77, 64) = (batch_size, 12, 77, 64)

        output = output.transpose(1,2).contiguous()# (batch_size, 77, 12, 64)

        output = output.view(input_shape)# (batch_size, 77, 768)

        output = self.out_proj(output)

        return output
