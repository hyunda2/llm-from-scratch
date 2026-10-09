# Day 2: 위에서 한 계산을 nn.Module로 묶어봤다.
# batch가 여러 개여도 동작하도록 만든 버전.
import torch
from torch import nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = self.W_query(x), self.W_key(x), self.W_value(x)
        scores = q @ k.transpose(1, 2) / (k.shape[-1] ** 0.5)
        scores = scores.masked_fill(self.mask[:tokens, :tokens], float('-inf'))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return weights @ v

if __name__ == '__main__':
    torch.manual_seed(123)
    model = CausalAttention(6, 4, context_length=8)
    x = torch.randn(2, 5, 6)
    print('입력:', x.shape)
    print('출력:', model(x).shape)  # (2, 5, 4)
