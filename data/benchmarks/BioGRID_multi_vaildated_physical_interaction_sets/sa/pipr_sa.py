# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:34:10 2021

@author: zju
"""
protein_seq = {}
with open('protein.dictionary.tsv', 'r') as r:
    line = r.readline()
    while line != '':
        term_list = line.split('\t')
        protein_seq[term_list[0]] = term_list[1].strip('\n')
        line = r.readline()
#ipq = 0
#imn = 0
#protein_count = []
protein_k = {}

with open('protein.actions.tsv', 'r') as r:
    line = r.readline()
    while line != '':
        name_1 = line.split('\t')[0]
        name_2 = line.split('\t')[1]
        cla = line.split('\t')[2].strip()
        
        if (len(protein_seq[name_1]) <= 2000) and (len(protein_seq[name_2]) <= 2000):
            protein_k[name_1 + '\t' + name_2] = cla
#            if name_1 not in protein_count:
#                protein_count.append(name_1)
#            if name_2 not in protein_count:
#                protein_count.append(name_2)
        line = r.readline()

with open('pipr_P_1.txt', 'w') as w1:
    with open('pipr_P_2.txt', 'w') as w2:
        with open('pipr_N_1.txt', 'w') as w3:
            with open('pipr_N_2.txt', 'w') as w4:
                
                for k in list(protein_k.keys()):
                    if protein_k[k] == '1':
                        w1.write('>' + k.split('\t')[0] + '\n' + protein_seq[k.split('\t')[0]] + '\n')
                        w2.write('>' + k.split('\t')[1] + '\n' + protein_seq[k.split('\t')[1]] + '\n')
                for k in list(protein_k.keys()):
                    if protein_k[k] == '0':
                        w3.write('>' + k.split('\t')[0] + '\n' + protein_seq[k.split('\t')[0]] + '\n')
                        w4.write('>' + k.split('\t')[1] + '\n' + protein_seq[k.split('\t')[1]] + '\n')