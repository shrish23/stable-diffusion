import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))# Positional encoding: it encodes the position of the token in the sequence

    def forward(self, tokens):
        # (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x) -> torch.Tensor:
        # x: (batch_size, seq_len, n_embd)
        residue = x

        # SELF ATTENTION

        x = self.layernorm_1(x)

        x = self.attention(x, casual_mask=True)# casual mask is used to prevent the model from looking at the future tokens

        x += residue

        # FEED FORWARD LAYER

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)# Quick GELU activation function

        x = self.linear_2(x)

        x += residue

        return x

class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)# 12 layers of CLIPLayer and the 12 is the number of heads and 768 is the embedding size
        ])

        self.layernorm = nn.LayerNorm(768)# Layer normalization: it normalizes the input to have zero mean and unit variance

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # tokens: (Batch size, seq_len) -> (Batch size, seq_len, 768)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # state: (Batch size, seq_len, 768) -> (Batch size, seq_len, 768)
        output = self.layernorm(state)# does not change the shape

        return output