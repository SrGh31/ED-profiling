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
import matplotlib as mpl
from scipy.spatial.distance import cdist, cosine
from model_explanations import model_feature_weights
from GetDataReady import getDataNormalized,  get_fname_exp, get_choice



def clf_fig_utils(keyD:float):    
    save_dict=get_fname_exp()
    exp_name=save_dict[keyD]
    dataset=getDataNormalized(keyD,0)    
    X, Y=dataset['zXtrain'], dataset['Ytrain']
    adapted_combo_cols, nclasses=X.columns, len(np.unique(Y))
    ind, fs,show_col_names = np.arange(len(adapted_combo_cols)-1), 10, []
    for idx, col in enumerate(adapted_combo_cols):
        if (col.split('-')[0]=='Main') | (col=='DT-BMI'):
            col=col.split('-')[1]   
        show_col_names.append(' '.join(col.split('_')))
    patterns = [ "/" , "++", "..", "xx", "\\" , "+" , "o", "O", "*", "|"  ]
    colors=[np.array([228,26,28])/256, np.array([55,126,184])/256, np.array([77,175,74])/256, np.array([56,108,176])/256, 
            np.array([255,127,0]) /256, np.array([191,91,23])/256, np.array([231,41,138])/256, np.array([247,129,191])/256]
    fig_utils_dict={'colors': colors, 'patterns':patterns, 'show_col_names':show_col_names, 'fs':fs, 'ind': ind,
                  }
    return fig_utils_dict

def feature_imp_glob(keyD:float, fimp_all:dict, savefig:int):    
    use_permutation_imp, select_keys1=['RF', 'KNN','RSVC', 'LDA','GMLVQ'], ['KNN','LDA','QDA','LSVC','RSVC']
    classifier_name1=['Random Forest', 'K-nearest neighbour', 'SVM w/ RBF','Linear Discr. Analysis','Generalized Matrix LVQ']
    shifts, bar_width, widths, ax_all=[-0.3,-0.15,0.0,0.15,0.30],0.15, [0.18,1],[]    
    save_dict=get_fname_exp()
    exp_name=save_dict[keyD]    
    fig_utils_dict=clf_fig_utils(keyD) #dataset=getDataNormalized(keyD,0)
    fs, ind=fig_utils_dict['fs'], fig_utils_dict['ind']
    colors, patterns, show_col_names=fig_utils_dict['colors'], fig_utils_dict['patterns'], fig_utils_dict['show_col_names']
    fig = plt.figure(constrained_layout=True, figsize=(5,11))
    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=widths)
    mpl.rcParams['hatch.linewidth'] = 0.5
    for colnum in range(2):
        ax_all.append(fig.add_subplot(spec[colnum]))
    ax, rects_all, bmi_all=ax_all[1],{},{}
    for idx,key in enumerate(select_keys1):
        if key in use_permutation_imp:
            bardata, err=np.mean(fimp_all[key]['All'], axis=1), np.std(fimp_all[key]['All'], axis=1)
      #  else:
      #      bardata, err=np.mean(fimp_all[key]['All'], axis=0), np.std(fimp_all[key]['All'], axis=0)        
        rects_all[key] = ax.barh(ind+shifts[idx], bardata[1:], height=bar_width, xerr=err[1:], label=key, 
                               error_kw=dict(lw=0.75, capsize=0.5, capthick=0.3), hatch=patterns[idx], color=colors[idx])        
        ax_all[0].barh(idx, bardata[0], xerr=err[0], hatch=patterns[idx], height=bar_width*2, 
                       color=colors[idx], error_kw=dict(lw=0.75, capsize=0.5, capthick=0.3))        
        if idx==0:
            ax_all[0].annotate('BMI', xy=(0.35, 0.1))
        mean_err=(np.column_stack((bardata, err))).flatten()
       # fimp_tabulate[key]=mean_err
    #fimp_tabulate=pd.DataFrame.from_dict(data=fimp_tabulate).T#fimp_tabulate.columns=res_df_colname
    ax.set_ylabel('Features', fontsize=fs)
    ax.set_yticks(ind)
    ax.set_yticklabels(show_col_names[1:], rotation=0, fontsize=fs)
    ax.legend('')
    ax.grid(axis='x')
    ax.set_xlim(0,0.18)
    ax.set_ylim(-0.5,len(show_col_names)-1.5)
    ax_all[0].set_yticks(np.arange(0,len(select_keys1)))
    ax_all[0].set_yticklabels(classifier_name1, rotation=0, fontsize=fs-1)
    ax_all[0].set_ylabel('Classifiers', fontsize=fs+1)
    ax_all[0].set_xlabel('importance/weight', fontsize=fs)
    ax_all[0].grid(axis='x')
    fig.suptitle('Overall feature importance/weight (%s)'%exp_name, fontsize=fs)
    fig.tight_layout()
    plt.show()
    if savefig>0:
        fig.savefig('figs/PDFs/classification/FIMP1%s_vert.pdf'%exp_name, bbox_inches='tight', transparent=False,
                   pad_inches=0.01)
        fig.savefig('figs/PNGs/classification/FIMP1_%s_vert.png'%exp_name, bbox_inches='tight', transparent=True,
                   pad_inches=0.01)
    
    

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
        hull = ConvexHull(pts)
        ax[0].scatter(test_data_to_plot[class_members_test, 0],test_data_to_plot[class_members_test, 1], color=col["color"], marker="+")
        ax[0].scatter(cluster_center[0], cluster_center[1], color=col["color"], marker='^', s=16,
                      label='C%d: N=%d (o), %d (+)'%(k+1, np.sum(class_members), np.sum(class_members_test)))        
        ax[0].fill(pts[hull.vertices,0], pts[hull.vertices,1],color=col["color"],alpha=0.35)        	    
    ax[0].set_xlabel(fig_lab_titles['ax0_xlab'], fontsize=fs)
    ax[0].set_ylabel(fig_lab_titles['ax0_ylab'], fontsize=fs)
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
    plt.savefig('figs/PDFs/clustering/ED_%s_%s.pdf'%(datasettag, saveExpName), bbox_inches='tight', dpi=200)
    plt.savefig('figs/PNGs/clustering/ED_%s_%s.png'%(datasettag, saveExpName), bbox_inches='tight', dpi=200)
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
    