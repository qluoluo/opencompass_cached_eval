import torch

# 创建一个示例数据矩阵 (每行是一个样本，每列是一个特征)
X = torch.tensor([[2.5, 2.4],
                  [0.5, 0.7],
                  [2.2, 2.9],
                  [1.9, 2.2],
                  [3.1, 3.0],
                  [2.3, 2.7],
                  [2.0, 1.6],
                  [1.0, 1.1],
                  [1.5, 1.6],
                  [1.1, 0.9]], dtype=torch.float)

# 1. 数据标准化 (使每个特征的均值为0)
X_mean = torch.mean(X, dim=0)
X_centered = X - X_mean

# 2. 计算协方差矩阵
cov_matrix = torch.mm(X_centered.t(), X_centered) / (X_centered.shape[0] - 1)

# 3. 进行SVD分解
U, S, V = torch.svd(cov_matrix)

# 4. 选择前 k 个主成分 (k=1)
k = 1
U_k = U[:, :k]

# 5. 将原始数据投影到低维空间
X_pca = torch.mm(X_centered, U_k)

# 6. 重构数据
X_reconstructed = torch.mm(X_pca, U_k.t()) + X_mean

# 计算重构误差
reconstruction_error = torch.mean((X - X_reconstructed) ** 2)

print("Original Data Matrix X:")
print(X)
print("\nMean-Centered Data Matrix X_centered:")
print(X_centered)
print("\nCovariance Matrix:")
print(cov_matrix)
print("\nU Matrix from SVD (Principal Components):")
print(U)
print("\nSingular Values:")
print(S)
print("\nV Matrix from SVD:")
print(V)
print("\nTransformed Data Matrix X_pca:")
print(X_pca)
print("\nReconstructed Data Matrix X_reconstructed:")
print(X_reconstructed)
print("\nReconstruction Error:")
print(reconstruction_error)