import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import time, itertools, re
import matplotlib.patches as patches
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, cosine

def clusterPlot(cluster_model, data_tabs, fig_naming, fig_lab_titles):
    fig, ax=plt.subplots(1,2, figsize=(8,4), layout='constrained')
    legend_loc, ncols, fs=fig_lab_titles['legend_loc'], fig_lab_titles['ncols'], fig_lab_titles['fs']
    saveExpName, datasettag=fig_naming[0], fig_naming[1]
    zs_cluster_median_profile,EDtype_per_cluster={},{}
    data_to_plot, df, z_df, cols2train=data_tabs['data_to_plot'], data_tabs['df'], data_tabs['z_df'], data_tabs['cols2train']
    model, train_pred_labs, test_pred_labs=cluster_model['model'], cluster_model['train_pred_labs'], cluster_model['test_pred_labs']
    test_data_to_plot=data_tabs['test_data_to_plot']
    if cluster_model['name']=='Affinity':        
        cluster_centers_indices=cluster_model['cluster_center_indices']       
    n_clusters_=len(np.unique(train_pred_labs))
    if n_clusters_<4:
        colors = plt.cycler("color", plt.cm.Paired(np.linspace(0, 2, n_clusters_+1)))
    else:
        colors = plt.cycler("color", plt.cm.Paired(np.linspace(0, 1, n_clusters_+1)))
    for k, col in zip(range(n_clusters_), colors):
        class_members, class_members_test = train_pred_labs == k, test_pred_labs==k	  
        cls_idx=np.where(class_members)[0]
        zs_cluster_median_profile[k]=z_df[cols2train].iloc[class_members].median(skipna=True)	
        EDtype_per_cluster[k]=df['EDtype'].iloc[class_members].value_counts().sort_index()        
        cluster_center=np.median(data_to_plot[class_members], axis=0)
        pwdists=[np.linalg.norm(cluster_center-row) for row in data_to_plot[cls_idx,:]]
        sort_idx=np.argsort(np.abs(pwdists))
        pts=data_to_plot[cls_idx, 0:2]        
        ax[0].scatter(data_to_plot[class_members, 0], data_to_plot[class_members, 1], color=col["color"], marker="o",s=5)
        #ax[0].scatter(cluster_center[0], cluster_center[1], s=10, color=col["color"], marker="^")
        hull = ConvexHull(pts)#ax211.tricontourf(pts[:, 0], pts[:, 1], levels=15, zorder=0, color=col["color"], alpha=0.3)        
        #ax[2].scatter(data_to_plot[class_members, 0],data_to_plot[class_members, 1], color=col["color"], marker="o", s=8)
        ax[0].scatter(test_data_to_plot[class_members_test, 0],test_data_to_plot[class_members_test, 1], color=col["color"], marker="+")
        ax[0].scatter(cluster_center[0], cluster_center[1], color=col["color"], marker='^', s=16,
                      label='C%d: N=%d (o), %d (+)'%(k+1, np.sum(class_members), np.sum(class_members_test)))        
        ax[0].fill(pts[hull.vertices,0], pts[hull.vertices,1],color=col["color"],alpha=0.35)
        #for x in data_to_plot[class_members]:
	    #    ax[0].plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"], alpha=0.5)
    ax[0].set_xlabel(fig_lab_titles['ax0_xlab'], fontsize=fs)
    ax[0].set_ylabel(fig_lab_titles['ax0_ylab'], fontsize=fs)
    #ax[2].set_xlabel(fig_lab_titles['ax0_xlab'], fontsize=fs)
    #ax[2].set_ylabel(fig_lab_titles['ax0_ylab'], fontsize=fs)
    ax[0].legend(fontsize=fs-3, ncol=ncols, loc=legend_loc, title='Train: o, Test: +', title_fontsize=fs-2)
    ax[0].set_title("%d %s"%(n_clusters_,fig_lab_titles['fig_title']), fontsize=fs)	
    ed_cluster_df=pd.DataFrame.from_dict(EDtype_per_cluster).T
    ed_cluster_df.rename(columns={'EDtype':'Cluster'}, inplace=True)
    ed_cluster_df.plot.bar(ax=ax[1])
    ax[1].set_xlabel('Clusters', fontsize=fs)
    ax[1].set_ylabel('Normalized feature values', fontsize=fs)
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    for idx,label in enumerate(labels):
        labels[idx]='C%d'%(idx+1)
    ax[1].set_xticklabels(labels, fontsize=fs-1)
    ax[1].legend(fontsize=fs-3)	
    plt.savefig('figs/PDFs/clustering/ED_%s_%s_20250402.pdf'%(datasettag, saveExpName), bbox_inches='tight', dpi=200)
    plt.savefig('figs/PNGs/clustering/ED_%s_%s_20250402.png'%(datasettag, saveExpName), bbox_inches='tight', dpi=200)
    zs_ed_cluster_df=pd.DataFrame.from_dict(zs_cluster_median_profile).T
    return zs_ed_cluster_df, ed_cluster_df, colors

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    