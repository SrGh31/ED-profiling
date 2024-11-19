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

def clusterPlot(cluster_model, data_tabs, fig_naming, fig_lab_titles):
    fig, ax=plt.subplots(1,2, figsize=(8,4), layout='constrained')
    legend_loc, ncols, fs=fig_lab_titles['legend_loc'], fig_lab_titles['ncols'], fig_lab_titles['fs']
    saveExpName, datasettag=fig_naming[0], fig_naming[1]
    zs_cluster_median_profile,EDtype_per_cluster={},{}
    data_to_plot, df, z_df, cols2train=data_tabs['data_to_plot'], data_tabs['df'], data_tabs['z_df'], data_tabs['cols2train']
    model, pred_labels=cluster_model['model'], cluster_model['pred_labels']
    if cluster_model['name']=='Affinity':        
        cluster_centers_indices=cluster_model['cluster_center_indices']       
    n_clusters_=len(np.unique(pred_labels))
    if n_clusters_<4:
        colors = plt.cycler("color", plt.cm.Paired(np.linspace(0, 2, n_clusters_+1)))
    else:
        colors = plt.cycler("color", plt.cm.Paired(np.linspace(0, 1, n_clusters_+1)))
    for k, col in zip(range(n_clusters_), colors):
        class_members = pred_labels == k	
        zs_cluster_median_profile[k]=z_df[cols2train].iloc[class_members].median(skipna=True)	
        EDtype_per_cluster[k]=df['EDtype'].iloc[class_members].value_counts().sort_index()
        if cluster_model['name']=='Affinity':
	        cluster_center = data_to_plot[cluster_centers_indices[k]]
        else:
	        cluster_center=data_to_plot[pred_labels == k].mean(axis=0)
        ax[0].scatter(data_to_plot[class_members, 0], data_to_plot[class_members, 1], color=col["color"], marker=".")
        if datasettag=='combinedDataset':
            ax[0].scatter(cluster_center[0], cluster_center[1], s=12, color=col["color"], marker="o", alpha=0.75,
	            label='C%d (%d), ES:%.2f, LAV:%.2f, SQ48:%.2f'%(k+1, np.sum(class_members),
		         df['EDEQ-Score'].iloc[class_members].mean(), df['Lav-Score'].iloc[class_members].mean(),
		                            df['SQ48-Score'].iloc[class_members].mean()))
        else:
            ax[0].scatter(cluster_center[0], cluster_center[1], s=12, color=col["color"], marker="o", alpha=0.75,
	            label='C%d (%d), ES:%.2f'%(k+1, np.sum(class_members),df['EDEQ-Score'].iloc[class_members].mean()))
        for x in data_to_plot[class_members]:
	        ax[0].plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"], alpha=0.5)
    ax[0].set_xlabel(fig_lab_titles['ax0_xlab'], fontsize=fs)
    ax[0].set_ylabel(fig_lab_titles['ax0_ylab'], fontsize=fs)
    ax[0].legend(fontsize=fs-3, ncol=ncols, loc=legend_loc)
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
    plt.savefig('figs/PDFs/ED_%s_%s.pdf'%(datasettag, saveExpName), bbox_inches='tight', dpi=200)
    plt.savefig('figs/PNGs/ED_%s_%s.png'%(datasettag, saveExpName), bbox_inches='tight', dpi=200)
    zs_ed_cluster_df=pd.DataFrame.from_dict(zs_cluster_median_profile).T
    return zs_ed_cluster_df, ed_cluster_df, colors


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
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

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    