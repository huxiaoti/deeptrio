# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:21:54 2021

@author: zju
"""

import numpy as np

pro_data = {}
pro_candi = {}
np.random.seed(seed=5)
with open('sa_combine_10.clstr', 'r') as r:
    line = r.readline()
    while line != '':
        if line.startswith('>'):
            pro_cluster = line.split(' ')[1].strip()
            old_name = ''
            count = 0
            
        else:
            pro_name = line.split('\t')[1].split(' ')[1].split('...')[0].lstrip('>')
            flo = line.split('\t')[1].split(' ')[2].strip()
            if flo == '*':
                old_name = pro_name
                pro_data[old_name] = pro_cluster
            else:
                number = line.strip().split('/')[-1].strip('%')
                rd = np.random.randint(1,3)
                if float(number) == 100 and ('reviewed' not in pro_name) and count == 0 and rd == 1:              
                    pro_data.pop(old_name)
                    count += 1
                    pro_data[pro_name] = pro_cluster
        line = r.readline()
pro_data_names = list(pro_data.keys())            
'''
protein_seq = {}        
with open('protein.dictionary.tsv', 'r') as r4:
    line = r4.readline()
    while line != '':
        name = line.split('\t')[0]
        seq = line.split('\t')[1].strip()
        protein_seq[name] = seq
        line = r4.readline()
        
with open('double_sa_MV_database.tsv', 'r') as r5:
    line = r5.readline()
    while line != '':
        name = line.split('\t')[0]
        seq = line.split('\t')[1].strip()
        protein_seq[name] = seq
        line = r5.readline()
'''
#with open('sa_ind_database.tsv', 'w') as w3:
#    for key in list(protein_seq.keys()):
#        w3.write(key + '\t' + protein_seq[key] + '\n')


pro_pair_r = {}
pro_r_names = []
with open('protein.actions.tsv', 'r') as r3:
    line = r3.readline()
    while line != '':
        name_1 = line.split('\t')[0]
        name_2 = line.split('\t')[1]
        if (name_1 in pro_data_names) and (name_2 in pro_data_names):
            if (50 < len(protein_seq[name_1]) < 1500) and (50 < len(protein_seq[name_2]) < 1500):
                pro_pair_r[name_1 + '\t' + name_2] = line.split('\t')[2].strip()
                if name_1 not in pro_r_names:
                    pro_r_names.append(name_1)
                if name_2 not in pro_r_names:
                    pro_r_names.append(name_2)
        line = r3.readline()

pro_pair_s = {}
pro_s_names = []
with open('third_sa_MV_pair.tsv', 'r') as r2:
    line = r2.readline()
    while line != '':
        name_1 = line.split('\t')[0]
        name_2 = line.split('\t')[1]
        if (name_1 in pro_data_names) and (name_2 in pro_data_names):
            if (50 < len(protein_seq[name_1]) < 1500) and (50 < len(protein_seq[name_2]) < 1500):
                pro_pair_s[name_1 + '\t' + name_2] = line.split('\t')[2].strip()
                if name_1 not in pro_s_names:
                    pro_s_names.append(name_1)
                if name_2 not in pro_s_names:
                    pro_s_names.append(name_2)
        line = r2.readline()

with open('ind_train_shuffle.tsv', 'w') as w1:
    
    for key in list(pro_pair_s.keys()):
        w1.write(key + '\t' + pro_pair_s[key] + '\n')
        
    with open('ind_test_random.tsv', 'w') as w2:
        pro_negative = []
        for key in list(pro_pair_r.keys()):
            name_1 = key.split('\t')[0]
            name_2 = key.split('\t')[1]
            if (name_1 not in pro_s_names) and (name_2 not in pro_s_names):
                if pro_pair_r[key] == '0':
                    rd = np.random.randint(1,801)
                    if rd <= 670:
                        pro_negative.append(key)
                elif pro_pair_r[key] == '1':
                    w2.write(key + '\t' + pro_pair_r[key] + '\n')
                else:
                    print('error')
        remove_num = np.random.randint(0,688,size=[21,])
        print(len(pro_negative))
        for n in remove_num:
            pro_negative.pop(n)
        for key in pro_negative:
            w2.write(key + '\t' + pro_pair_r[key] + '\n')