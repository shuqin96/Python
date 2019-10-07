#!/usr/bin/python
#-*- coding:utf-8 -*- 
'''
Author: Wangshuqin
Data: 2019-10-07 21:23:41
Last Modified time: 2019-10-07 21:23:41
Description: 将比较的列表数据转变成矩阵格式
'''

import pandas as pd

def result_become_matrix():
    muta = pd.read_excel(outpath + 'four_data_have_all_score_file_compare_mutation_ref.xlsx')
    aa = pd.read_excel(outpath + 'four_data_have_all_score_file_compare_aachange.xlsx')
    data = [muta, aa]
    namelist = ['mutation_ref', 'aachange']
    # order by you want
    collist = ['hpm','fmd','fmn','cancer','hpm.random','cancer.random','exac','cancer.census.fre1','cancer.census.fre2+','cancer.census.fre3+','cancer.census.fre5+']
    for i in range(len(data)):
        f = data[i]
        name = namelist[i]
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
        df_triu.to_csv(outpath + "four_data_have_all_score_file_compare_" + name + "_matrix.txt",sep='\t')