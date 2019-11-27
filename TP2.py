# -*- coding: utf-8 -*-

import numpy as np
from tp2_aux import images_as_matrix, report_clusters
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    data_res = np.load('feature_res.npz')
    print('Leu')
    pca_data = data_res['pca_data']
    tsne_data = data_res['tsne_data']
    iso_data = data_res['iso_data']
except IOError:
    data = images_as_matrix()
    print('Nao leu')
    pca = PCA(n_components=6)
    pca_data = pca.fit_transform(data)

    tsne = TSNE(n_components=6, method='exact')
    tsne_data = tsne.fit_transform(data)

    iso = Isomap(n_components=6)
    iso_data = iso.fit_transform(data)

    np.savez('feature_res.npz', pca_data=pca_data, tsne_data=tsne_data, iso_data=iso_data)

data_labels = np.loadtxt('labels.txt', delimiter=',')
#stacked_features = np.concatenate((pca_data, tsne_data, iso_data), axis=1)

#labeled = data_labels[data_labels[:,1]>0,:]
#print(pca_data[data_labels[:,1]>0,:])
pca_f, pca_prob = f_classif(pca_data[data_labels[:,1]>0,:], data_labels[data_labels[:,1]>0,1])
tsne_f, tsne_prob = f_classif(tsne_data[data_labels[:,1]>0,:], data_labels[data_labels[:,1]>0,1])
iso_f, iso_prob = f_classif(iso_data[data_labels[:,1]>0,:], data_labels[data_labels[:,1]>0,1])

kmeans = KMeans(n_clusters=4)
lbls = kmeans.fit_predict(pca_data)
report_clusters(np.linspace(0, data_labels.shape[0] - 1, data_labels.shape[0]), lbls, 'teste.html')



