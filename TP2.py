# -*- coding: utf-8 -*-

import numpy as np
from tp2_aux import images_as_matrix, report_clusters
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, DBSCAN,AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

def select_features_MX(f_values, features, th):
    f_max = max(f_values)
    res = features[:,f_values[:]>f_max * th]
    return res

def select_features_NB(f_values, features, n):
    selected = []
    res = []
    max = 0
    for nx in range(n):
        nmax = -1
        max = 0
        for ix in range(f_values.shape[0]):
            if ix not in selected and f_values[ix] > max:
                max = f_values[ix]
                nmax = ix
        selected.append(nmax)
        if len(res) == 0:
            res = features[:, nmax]
        else:
            res = np.column_stack((res, features[:, nmax]))
    return res

def standardize(vec):
    res = 0
    for y in range(vec.shape[1]):
        if y == 0:
            res = (vec[:,y] - np.mean(vec[:,y], axis=0))/np.std(vec[:,y], axis=0)
        else:
            res = np.column_stack((res, (vec[:,y] - np.mean(vec[:,y], axis=0))/np.std(vec[:,y], axis=0)))
        
    return res

def ext_indexes(pred, true):
    n = len(pred)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for ix in range(n - 1):
        for jx in range(ix + 1, n):
            if pred[ix] == pred[jx] and true[ix] == true[jx]:
                tp += 1
            if pred[ix] == pred[jx] and true[ix] != true[jx]:
                fp += 1
            if pred[ix] != pred[jx] and true[ix] == true[jx]:
                fn += 1
            if pred[ix] != pred[jx] and true[ix] != true[jx]:
                tn += 1

    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        rand = (tp + tn) / (n * (n - 1) / 2)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision = 0
        recall = 0
        rand = 0
        f1 = 0
    
    return precision, recall, rand, f1

def label_kmeans(main_arg, reach, x, true_lbls):
    silh_array = []
    ari_array = []
    prcsn_array = []
    rcl_array = []
    rand_array = []
    f_array = []
    center_labels = None
    min = 2
    if main_arg - reach > 1:
        min = main_arg - reach

    max = main_arg + reach + 1
    space = range(min, max)
    for iarg in space:
        kmeans = KMeans(n_clusters=iarg)
        lbls = kmeans.fit_predict(x)
        if iarg == main_arg:
            center_labels = lbls
        silh_array.append(silhouette_score(x, lbls))
        ari_array.append(adjusted_rand_score(true_lbls[true_lbls[:,1]>0,1], lbls[true_lbls[:,1]>0]))
        p, r, a, f = ext_indexes(lbls[true_lbls[:,1]>0], true_lbls[true_lbls[:,1]>0,1])
        prcsn_array.append(p)
        rcl_array.append(r)
        rand_array.append(a)
        f_array.append(f)
    
    plt.figure()
    plt.title('K-Means scores (center: {0:1.0f}; range: {1:1.0f})'.format(main_arg, reach))
    plt.plot(space, silh_array, label='Silhouette score')
    plt.plot(space, ari_array, label='Adjusted Rand score')
    plt.plot(space, prcsn_array, label='Precision score')
    plt.plot(space, rcl_array, label='Recall score')
    plt.plot(space, rand_array, label='Rand score')
    plt.plot(space, f_array, label='F1 score')
    plt.legend()
    plt.savefig('KMeans-plot',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

    return center_labels

def label_dbscan(main_arg, range, precision, x, true_lbls):
    silh_array = []
    ari_array = []
    prcsn_array = []
    rcl_array = []
    rand_array = []
    f_array = []
    center_labels = None
    min = main_arg - range
    max = main_arg + range
    num = 1 + precision * 2
    space = np.linspace(min, max, num)
    for iarg in space:
        dbscan = DBSCAN(eps=iarg)
        lbls = dbscan.fit_predict(x)
        if iarg == main_arg:
            center_labels = lbls
        try:
            silh_array.append(silhouette_score(x, lbls))
        except ValueError:
            silh_array.append(-1)
        ari_array.append(adjusted_rand_score(true_lbls[true_lbls[:,1]>0,1], lbls[true_lbls[:,1]>0]))
        p, r, a, f = ext_indexes(lbls[true_lbls[:,1]>0], true_lbls[true_lbls[:,1]>0,1])
        prcsn_array.append(p)
        rcl_array.append(r)
        rand_array.append(a)
        f_array.append(f)
    
    plt.figure()
    plt.title('DBSCAN scores (center: {0:1.2f}; range: {1:1.2f}; precision: {2:1.0f})'.format(main_arg, range, precision))
    plt.plot(space, silh_array, label='Silhouette score')
    plt.plot(space, ari_array, label='Adjusted Rand score')
    plt.plot(space, prcsn_array, label='Precision score')
    plt.plot(space, rcl_array, label='Recall score')
    plt.plot(space, rand_array, label='Rand score')
    plt.plot(space, f_array, label='F1 score')
    plt.legend()
    plt.savefig('DBSCAN-plot',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

    return center_labels

def label_AC(main_arg, reach, x, true_lbls):
    silh_array = []
    ari_array = []
    prcsn_array = []
    rcl_array = []
    rand_array = []
    f_array = []
    center_labels = None
    min = 2
    if main_arg - reach > 1:
        min = main_arg - reach

    max = main_arg + reach + 1
    space = range(min, max)
    for iarg in space:
        ac = AgglomerativeClustering(n_clusters=iarg)
        lbls = ac.fit_predict(x)
        if iarg == main_arg:
            center_labels = lbls
        silh_array.append(silhouette_score(x, lbls))
        ari_array.append(adjusted_rand_score(true_lbls[true_lbls[:,1]>0,1], lbls[true_lbls[:,1]>0]))
        p, r, a, f = ext_indexes(lbls[true_lbls[:,1]>0], true_lbls[true_lbls[:,1]>0,1])
        prcsn_array.append(p)
        rcl_array.append(r)
        rand_array.append(a)
        f_array.append(f)
    
    plt.figure()
    plt.title('AgglemerativeClustering scores (center: {0:1.0f}; range: {1:1.0f})'.format(main_arg, reach))
    plt.plot(space, silh_array, label='Silhouette score')
    plt.plot(space, ari_array, label='Adjusted Rand score')
    plt.plot(space, prcsn_array, label='Precision score')
    plt.plot(space, rcl_array, label='Recall score')
    plt.plot(space, rand_array, label='Rand score')
    plt.plot(space, f_array, label='F1 score')
    plt.legend()
    plt.savefig('AgglomerativeClustering-plot',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

    return center_labels

try:
    data_res = np.load('feature_res.npz')
    pca_data = data_res['pca_data']
    tsne_data = data_res['tsne_data']
    iso_data = data_res['iso_data']
except IOError:
    data = images_as_matrix()
    pca = PCA(n_components=6)
    pca_data = pca.fit_transform(data)

    tsne = TSNE(n_components=6, method='exact')
    tsne_data = tsne.fit_transform(data)

    iso = Isomap(n_components=6)
    iso_data = iso.fit_transform(data)

    np.savez('feature_res.npz', pca_data=pca_data, tsne_data=tsne_data, iso_data=iso_data)

data_labels = np.loadtxt('labels.txt', delimiter=',')
stacked_features = np.concatenate((pca_data, tsne_data, iso_data), axis=1)
stacked_f, stacked_prob = f_classif(stacked_features[data_labels[:,1]>0,:], data_labels[data_labels[:,1]>0,1])

plt.figure()
plt.bar(range(18), stacked_f, width=.2,
        label=r'Univariate score', color='darkorange',
        edgecolor='black')
plt.legend()
plt.show()
plt.close()

final_features = select_features_NB(stacked_f, stacked_features, 5)
final_features = standardize(final_features)

knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(final_features, data_labels[:, 0])
distances = knn.kneighbors()
distances = np.sort(distances[0][:, -1])
distances = distances[::-1]

plt.figure()
plt.title('Fifth-nearest distance per point')
plt.plot(distances)
plt.show()
plt.close()

dbscan_labels = label_dbscan(.73, .35, 40, final_features, data_labels)
report_clusters(np.linspace(0, data_labels.shape[0] - 1, data_labels.shape[0]), dbscan_labels, 'teste_dbscan.html')

kmeans_labels = label_kmeans(13, 17, final_features, data_labels)
report_clusters(np.linspace(0, data_labels.shape[0] - 1, data_labels.shape[0]), kmeans_labels, 'teste_kmeans.html')

AC_labels = label_AC(3, 40, final_features, data_labels)
report_clusters(np.linspace(0, data_labels.shape[0] - 1, data_labels.shape[0]), AC_labels, 'teste_AC.html')