#!/usr/bin/python
#-*- coding:utf-8 -*- 
'''
Author: Wangshuqin
Data: 2019-10-07 21:23:41
Last Modified time: 2019-10-07 21:23:41
Description: 将比较的列表数据转变成矩阵格式
'''

import pandas as pd
import numpy as np
import itertools
outpath = "/data/wangs/MutaAnalysis/three.data0223/output/"

# *Including self and self comparison
def findsubsets(S,m):
    return set(itertools.combinations_with_replacement(S, m))


def compare_data_with_other():
    keyname = ['hpmindex', 'fmnindex', 'fmdindex', 'cancerindex', 'hpmrandomindex', 'cancerrandomindex', 'exacindex']
    datalist = ['hpm','fmn','fmd','cancer','hpmrandom','cancerrandom','exac']
    # load data
    hpm = pd.read_table(path + 'hpm_data_have_all_score_clean.tsv',dtype=str)
    functional = pd.read_table(path + 'functional_data_have_all_score_clean.tsv',dtype=str)
    fmd = functional[functional['y']=='1']
    fmn = functional[functional['y']=='0']
    cancer = pd.read_table('/data1/wangs/CanDriver/data/ZONG/output/icgc,tcga,cosmic_have_all_score_ptm.tsv',dtype=str)
    hpmrandom = pd.read_table(path + 'hpm_background_mutation_have_all_score_clean.tsv', dtype=str)
    cancerrandom = pd.read_table(path + 'cancer_background_mutation_have_all_score_clean.tsv', dtype=str)
    exac = pd.read_table("/data1/wangs/CanDriver/data/ExAC/output/ExAC_nonTCGA_have_all_score_clean.tsv", dtype=str)  
    for name in datalist:
        data = eval(name)
        if ('census' in name) or (name == 'cancer'):
            data['index1'] = data['symbol'] +':'+ data['aa_mutation']
        else:   
            data['index1'] = data['gene'] +':'+ data['aachange']
    ## rename and key
    hpmindex = set(hpm['index1'])
    fmnindex = set(fmn['index1'])
    fmdindex = set(fmd['index1'])
    cancerindex = set(cancer['index1'])
    hpmrandomindex = set(hpmrandom['index1'])
    cancerrandomindex = set(cancerrandom['index1'])
    exacindex = set(exac['index1'])
    # 计算重叠部分
    with open(path + 'four_data_have_all_score_file_compare_aachange.xlsx', 'w') as fw:
        fw.write('%s\t%s\t%s\n' % ('data1', 'data2', 'number_of_same'))
        assemble2 = list(findsubsets(keyname, 2))
        for index in assemble2:
            gong = eval(index[0])&eval(index[1])
            name1, name2 = index[0].replace('index', ''), index[1].replace('index', '')
            fw.write('%s\t%s\t%s\n' % (name1,name2,len(gong)))

## format as below，必须是以下格式，否则下一个函数无法运行
# data1 data2 number_of_same
# fmd fmd 1444
# fmd fmn 0
# fmd hpm 43
# fmd exac 5


def list_to_matrix():
    f = pd.read_excel(outpath + 'four_data_have_all_score_file_compare_aachange.xlsx')
    # order by you want
    collist = ['hpm','fmd','fmn','cancer','hpm.random','cancer.random','exac','cancer.census.fre1','cancer.census.fre2+','cancer.census.fre3+','cancer.census.fre5+']
    # 构造一个空数据框
    df = pd.DataFrame(columns = collist, index = collist) # 指定列顺序
    datalist = list(set(f['data1']))
    for data1 in datalist:
        for data2 in datalist:
            # 两个数据集之间比较时，在结果文件中的前后顺序不一样，如（candl,civic）与 (civic,candl)
            col = 'number_of_same'
            try:
                value = f[(f['data1'] == data1) & (f['data2'] == data2)][col].values[0]
            except Exception:
                value = f[(f['data2'] == data1) & (f['data1'] == data2)][col].values[0]
            df.loc[data1,data2] = value
    # 此数据框中上下三角都有值，比较冗余，只要上三角
    df_triu = pd.DataFrame(np.triu(df),columns = collist, index = collist)
    df_triu.to_csv(outpath + "four_data_have_all_score_file_compare_aachange_matrix.txt",sep='\t')