# Day 3: attention head를 여러 개 쓰면 서로 다른 관계를 볼 수 있다.
# 우선 작은 head들을 병렬로 계산하고 마지막에 합쳐본다.
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        b, t, _ = x.shape
        def split_heads(layer):
            return layer(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(self.W_query), split_heads(self.W_key), split_heads(self.W_value)
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:t, :t], float('-inf'))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(b, t, -1)
        return self.out_proj(context)

if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.randn(2, 5, 6)
    model = MultiHeadAttention(6, 8, context_length=8, num_heads=2)
    print('입력:', x.shape)
    print('출력:', model(x).shape)  # (2, 5, 8)
