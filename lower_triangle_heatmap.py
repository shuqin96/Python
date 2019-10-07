#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@Author: Wangshuqin
@Date:   2019-06-14 19:56:07
@Last Modified by:   WangShuqin
@Last Modified time: 2019-06-29 09:41:50
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = '/data/wangs/MutaAnalysis/three.data0223/figure/'
# in alphabetical order按字母顺序排列
methodlist = ['CADD','CHASMplus','CanDrAplus','CONDEL','DANN','Eigen','fitCons','FATHMM','FATHMM-MKL','FATHMM-Cancer','GERP++','GenoCanyon',
    'LRT','M-CAP','MetaSVM','MetaLR','MutPred','MutationTaster','MutationAssessor','phyloP','phastCons','PROVEAN','ParsSNP','PolyPhen2-HDIV','PolyPhen2-HVAR',
    'REVEL','SIFT','SiPhy','VEST3']
## load data
hpm =pd.read_csv(path  + "hpm_method_cor_table.txt",sep='\t')
fmd =pd.read_csv(path  + "fmd_method_cor_table.txt",sep='\t')
fre5 =pd.read_csv(path  + "cancer.census.fre5d_method_cor_table.txt",sep='\t')
fmn =pd.read_csv(path  + "fmn_method_cor_table.txt",sep='\t')
exac =pd.read_csv(path  + "exac_method_cor_table.txt",sep='\t')
random =pd.read_csv(path  + "cancer.random_method_cor_table.txt",sep='\t')
## 构造热图数据框
# hpm
df1 = pd.DataFrame(0,columns=methodlist,index=methodlist)
for i in range(len(methodlist)):
	method1 = methodlist[i]
	for j in range(len(methodlist)):
		method2 = methodlist[j]
		df1.loc[method1,method2] = hpm[(hpm['method1'] == method1) & (hpm['method2'] == method2)]['higher_cor'].values[0]
# fmd
df2 = pd.DataFrame(0,columns=methodlist,index=methodlist)
for i in range(len(methodlist)):
    method1 = methodlist[i]
    for j in range(len(methodlist)):
        method2 = methodlist[j]
        df2.loc[method1,method2] = fmd[(fmd['method1'] == method1) & (fmd['method2'] == method2)]['higher_cor'].values[0]
# fre5
df3 = pd.DataFrame(0,columns=methodlist,index=methodlist)
for i in range(len(methodlist)):
    method1 = methodlist[i]
    for j in range(len(methodlist)):
        method2 = methodlist[j]
        df3.loc[method1,method2] = fre5[(fre5['method1'] == method1) & (fre5['method2'] == method2)]['higher_cor'].values[0]
# fmn
df4 = pd.DataFrame(0,columns=methodlist,index=methodlist)
for i in range(len(methodlist)):
    method1 = methodlist[i]
    for j in range(len(methodlist)):
        method2 = methodlist[j]
        df4.loc[method1,method2] = fmn[(fmn['method1'] == method1) & (fmn['method2'] == method2)]['higher_cor'].values[0]
# exac
df5 = pd.DataFrame(0,columns=methodlist,index=methodlist)
for i in range(len(methodlist)):
    method1 = methodlist[i]
    for j in range(len(methodlist)):
        method2 = methodlist[j]
        df5.loc[method1,method2] = exac[(exac['method1'] == method1) & (exac['method2'] == method2)]['higher_cor'].values[0]
# cancer random
df6 = pd.DataFrame(0,columns=methodlist,index=methodlist)
for i in range(len(methodlist)):
    method1 = methodlist[i]
    for j in range(len(methodlist)):
        method2 = methodlist[j]
        df6.loc[method1,method2] = random[(random['method1'] == method1) & (random['method2'] == method2)]['higher_cor'].values[0]

# background color
# sns.set(style="white")
# f, (ax1,ax2,ax3,axcb) = plt.subplots(1,4,figsize=(54, 15))
# cmap = sns.diverging_palette(150, 10,n=15,as_cmap=False)
# # 1.hpm dataset
# ax1 = plt.subplot(131)
# mask = np.zeros_like(df1, dtype=np.bool)
# mask[np.triu_indices_from(mask,k=1)] = True
# sns.heatmap(df1,mask=mask, cmap=cmap,center=0.5, ax=ax1,cbar=False,vmin=0,square=True)
# ax1.set_title('Correlation Of HPM',fontsize=24)
# ax1.tick_params(labelsize=18) # change xlabel and ylabel tick size
# # 2.fmd
# ax2 = plt.subplot(132)
# mask = np.zeros_like(df2, dtype=np.bool)
# mask[np.triu_indices_from(mask,k=1)] = True
# sns.heatmap(df2,mask=mask, cmap=cmap,center=0.5, ax=ax2,vmin=0,cbar=False,square=True)
# ax2.set_title('Correlation Of FDM',fontsize=24)
# ax2.tick_params(labelsize=18)
# # 3.cancer census frequency>=5
# ax3 = plt.subplot(133)
# mask = np.zeros_like(df3, dtype=np.bool)
# mask[np.triu_indices_from(mask,k=1)] = True
# sns.heatmap(df3,mask=mask, cmap=cmap,center=0.5, ax=ax3,vmin=0,cbar_ax=axcb)
# ax3.set_title('Correlation Of Cancer Census Frequency>=5',fontsize=24)
# ax3.tick_params(labelsize=18)
# cax = plt.gcf().axes[-1]
# cax.tick_params(labelsize=10) # change colorbar tick size
# plt.subplots_adjust(wspace=0.8)
# plt.show()
# f.savefig(path + 'deleterious_data_lower_heatmap.jpg', bbox_inches='tight')


############# deleterious datset heatmap ##############
sns.set(style="white")
plt.rcParams['font.family'] = 'Overpass'  # 设置字体
f, ax = plt.subplots(2,3,figsize=(27, 16))
cmap = sns.diverging_palette(150, 10,n=15,as_cmap=False)
# 1.hpm dataset
ax1 = plt.subplot(231)
mask = np.zeros_like(df1, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df1,mask=mask, cmap=cmap, ax=ax1,cbar=False,vmin=0,square=True)
ax1.set_title('Correlation Of HPM',fontsize=24)
ax1.tick_params(labelsize=18) # change xlabel and ylabel tick size
# 2.fmd
ax2 = plt.subplot(232)
mask = np.zeros_like(df2, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df2,mask=mask, cmap=cmap,ax=ax2,vmin=0,cbar=False,square=True)
ax2.set_title('Correlation Of FDM',fontsize=24)
ax2.tick_params(labelsize=18)
# 3.cancer census frequency>=5
ax3 = plt.subplot(233)
mask = np.zeros_like(df3, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df3,mask=mask, cmap=cmap, ax=ax3,vmin=0)
ax3.set_title('Correlation Of Cancer Census Frequency>=5',fontsize=24)
ax3.tick_params(labelsize=18)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
############# neutral datset heatmap ##############
# 1.fmn
ax4 = plt.subplot(234)
mask = np.zeros_like(df4, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df4,mask=mask, cmap=cmap, ax=ax4,vmin=0,cbar=False,square=True)
ax4.set_title('Correlation Of FNM',fontsize=24)
ax4.tick_params(labelsize=18)
# 2.exac
ax5 = plt.subplot(235)
mask = np.zeros_like(df5, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df5,mask=mask, cmap=cmap, ax=ax5,vmin=0,cbar=False,square=True)
ax5.set_title('Correlation Of ExAC',fontsize=24)
ax5.tick_params(labelsize=18)
# 3.cancer random
ax6 = plt.subplot(236)
mask = np.zeros_like(df6, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df6,mask=mask, cmap=cmap, ax=ax6,vmin=0)
ax6.set_title('Correlation Of Random Mutation',fontsize=24)
ax6.tick_params(labelsize=18)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20) # change colorbar tick size
plt.subplots_adjust(wspace=0.3,hspace=0.5) # 控制上下左右图形之间的间隔
f.savefig(path + 'all_data_lower_heatmap.svg', bbox_inches='tight', dpi=1000)


## ---------
## 分开作图
## ---------

## deleterious
sns.set(style="white")
plt.rcParams['font.family'] = 'Overpass'  # 设置字体
f, ax = plt.subplots(1,3,figsize=(30, 7))
cmap = sns.diverging_palette(150, 10,n=15,as_cmap=False)
# 1.hpm dataset
ax1 = plt.subplot(131)
mask = np.zeros_like(df1, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df1,mask=mask, cmap=cmap, ax=ax1,cbar=False,vmin=0,square=True)
ax1.set_title('Correlation Of HPM',fontsize=24)
ax1.tick_params(labelsize=15) # change xlabel and ylabel tick size
# 2.fmd
ax2 = plt.subplot(132)
mask = np.zeros_like(df2, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df2,mask=mask, cmap=cmap,ax=ax2,vmin=0,cbar=False,square=True)
ax2.set_title('Correlation Of FDM',fontsize=24)
ax2.tick_params(labelsize=15)
# 3.cancer census frequency>=5
ax3 = plt.subplot(133)
mask = np.zeros_like(df3, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df3,mask=mask, cmap=cmap, ax=ax3,vmin=0)
ax3.set_title('Correlation Of Cancer Census Frequency>=5',fontsize=24)
ax3.tick_params(labelsize=15)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.4)
f.savefig(path + 'deleterious_data_lower_heatmap.svg', bbox_inches='tight', dpi=1000)

## neutral
sns.set(style="white")
plt.rcParams['font.family'] = 'Overpass'  # 设置字体
f, ax = plt.subplots(1,3,figsize=(30, 7))
cmap = sns.diverging_palette(150, 10,n=15,as_cmap=False)
# 1.hpm dataset
ax1 = plt.subplot(131)
mask = np.zeros_like(df1, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df4,mask=mask, cmap=cmap, ax=ax1,cbar=False,vmin=0,square=True)
ax1.set_title('Correlation Of FNM',fontsize=24)
ax1.tick_params(labelsize=15) # change xlabel and ylabel tick size
# 2.fmd
ax2 = plt.subplot(132)
mask = np.zeros_like(df2, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df5,mask=mask, cmap=cmap,ax=ax2,vmin=0,cbar=False,square=True)
ax2.set_title('Correlation Of ExAC',fontsize=24)
ax2.tick_params(labelsize=15)
# 3.cancer census frequency>=5
ax3 = plt.subplot(133)
mask = np.zeros_like(df3, dtype=np.bool)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(df6,mask=mask, cmap=cmap, ax=ax3,vmin=0)
ax3.set_title('Correlation Of Random Mutation',fontsize=24)
ax3.tick_params(labelsize=15)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.4)
f.savefig(path + 'neutral_data_lower_heatmap.svg', bbox_inches='tight', dpi=1000)
