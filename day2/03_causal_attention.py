# Day 2: GPT는 미래 단어를 미리 보면 안 된다.
# 그래서 오른쪽 위 삼각형을 가려준다.
import torch

torch.manual_seed(123)
q = torch.randn(5, 4)
k = torch.randn(5, 4)
v = torch.randn(5, 4)
scores = q @ k.T / (q.shape[-1] ** 0.5)
mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))
weights = torch.softmax(scores, dim=-1)
context = weights @ v
print('mask:', mask.int())
print('weights:', weights)
print('output:', context.shape)
