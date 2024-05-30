import torch
from sklearn.decomposition import IncrementalPCA

y = torch.rand((128,256))

print(y.shape)

pca_model = IncrementalPCA(n_components=32)

pca_model.partial_fit(y)

y = torch.rand((12,256))

compressed_y = pca_model.transform(y)

print(f"{compressed_y.shape=}")

y_decompress = pca_model.inverse_transform(compressed_y)

print(y_decompress.shape)