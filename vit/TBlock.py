import torch.nn as nn

'''
MLP block composed of two linear layers with GELU activation and dropout.
'''
class FeedForward(nn.Module):
    def __init__(self, embedDim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embedDim, mlp_dim*embedDim)
        self.act = nn.Tanh()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim*embedDim, embedDim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
'''
Attention block implemented as in https://arxiv.org/abs/2010.11929
'''
class Attention(nn.Module):
    def __init__(self, embedDim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.headDim = embedDim // heads
        self.scale = self.headDim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(embedDim, embedDim * 3)
        self.proj = nn.Linear(embedDim, embedDim)

        #same value for both the dropout layers
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, GS, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, GS, 3, self.heads, self.headDim)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).flatten(2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

'''
Transformer block composed of an attention block and an MLP block plus normalization layers.
'''
class TBlock(nn.Module):
    def __init__(self, embedDim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedDim)
        self.norm2 = nn.LayerNorm(embedDim)
        self.attn = Attention(embedDim, heads, dropout)
        self.mlp = FeedForward(embedDim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x