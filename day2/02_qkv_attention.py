# Day 2: Q, K, V를 따로 만들어보자.
# Q는 뭘 찾는지, K는 어떤 정보인지, V는 실제 가져올 정보라고 이해했다.
import torch
from torch import nn

torch.manual_seed(123)
x = torch.randn(4, 6)
d_in, d_out = 6, 4
W_q = nn.Linear(d_in, d_out, bias=False)
W_k = nn.Linear(d_in, d_out, bias=False)
W_v = nn.Linear(d_in, d_out, bias=False)
q, k, v = W_q(x), W_k(x), W_v(x)
scores = q @ k.T
weights = torch.softmax(scores / (d_out ** 0.5), dim=-1)
context = weights @ v
print('Q/K/V:', q.shape, k.shape, v.shape)
print('weights:', weights)
print('output:', context.shape)
