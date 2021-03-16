# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:39:59 2021

@author: zju
"""
protein_seq = {}
with open('double_sa_MV_database.tsv', 'r') as r:
    line = r.readline()
    while line != '':
        term_list = line.split('\t')
        protein_seq[term_list[0]] = term_list[1]
        line = r.readline()

protein_tab = {}
with open('third_sa_MV_pair.tsv', 'r') as r2:
    line = r2.readline()
    while line != '':
        name_1 = line.split('\t')[0]
        name_2 = line.split('\t')[1]
        protein_tab[name_1] = protein_seq[name_1]
        protein_tab[name_2] = protein_seq[name_2]
        line = r2.readline()

with open('cd_seq_sa.fasta', 'w') as w:
    for key in list(protein_tab.keys()):
        w.write('>' + key + '\n')
        w.write(protein_tab[key])