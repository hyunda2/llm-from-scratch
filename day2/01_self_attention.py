# Day 2: attention을 가장 단순하게 계산해보기
# 각 단어가 다른 단어를 얼마나 참고하는지 점수로 구한다.
import torch

torch.manual_seed(123)
x = torch.randn(4, 3)  # 토큰 4개, 각 토큰 벡터 길이 3
scores = x @ x.T
weights = torch.softmax(scores, dim=-1)
context = weights @ x
print('attention weights:', weights)
print('context vectors:', context)
print('각 행의 합:', weights.sum(dim=-1))  # 1이어야 함
