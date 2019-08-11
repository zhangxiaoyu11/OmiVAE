import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.manifold import TSNE
import umap

print('Loading data...')
input_path = 'data/PANCAN/GDC-PANCAN_'
sample_id = np.loadtxt(input_path + 'both_samples.tsv', delimiter='\t', dtype='str')

input_df = pd.read_csv(input_path + 'preprocessed_both.tsv', sep='\t', header=0, index_col=0)
input_df = input_df.T

latent_space_dimension = 2

# PCA
print('PCA')
pca = decomposition.PCA(n_components=latent_space_dimension)
z = pca.fit_transform(input_df.values)
latent_code = pd.DataFrame(z, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_PCA_latent_sapce.tsv'
latent_code.to_csv(output_path, sep='\t')

# KPCA
print('KPCA')
kpca = decomposition.KernelPCA(n_components=latent_space_dimension, kernel='rbf')
z = kpca.fit_transform(input_df.values)
latent_code = pd.DataFrame(z, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_KPCA_latent_sapce.tsv'
latent_code.to_csv(output_path, sep='\t')

# TSNE
print('TSNE')
tsne = TSNE(n_components=latent_space_dimension)
z = tsne.fit_transform(input_df.values)
latent_code = pd.DataFrame(z, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_TSNE_latent_sapce.tsv'
latent_code.to_csv(output_path, sep='\t')

# UMAP
print('UMAP')
umap_reducer = umap.UMAP()
z = umap_reducer.fit_transform(input_df.values)
latent_code = pd.DataFrame(z, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_UMAP_latent_sapce.tsv'
latent_code.to_csv(output_path, sep='\t')
