import torch

# 创建一个随机的矩阵A，大小为4x3
# A = torch.randn(4, 3)
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 7]], dtype=torch.float)
print("Original matrix A:")
print(A)

# 使用torch.svd进行奇异值分解
U, S, V = torch.svd(A)

# 选择k的值
k = 2

# 取S的前k个值，并构造对角矩阵
S_k = torch.diag(S[:k])

# 取U和V的前k列
U_k = U[:, :k]
V_k = V[:, :k]

# 中间矩阵开根号
S_k_sqrt = torch.sqrt(S_k)

# 构造两个矩阵的乘积
B = torch.mm(U_k, S_k_sqrt)
C = torch.mm(S_k_sqrt, V_k.t())
# B = U_k
# C = torch.mm(S_k, V_k.t())
# B = torch.mm(U_k, S_k)
# C = V_k.t()

print("\nMatrix B:")
print(B)
print("\nMatrix C:")
print(C)

# 验证原始矩阵和分解后矩阵的乘积
A_reconstructed = torch.mm(B, C)
print("\nReconstructed matrix A:")
print(A_reconstructed)

# 验证原始矩阵和重建矩阵的差异
difference = torch.norm(A - A_reconstructed)
print("\nDifference between original and reconstructed matrix:")
print(difference)