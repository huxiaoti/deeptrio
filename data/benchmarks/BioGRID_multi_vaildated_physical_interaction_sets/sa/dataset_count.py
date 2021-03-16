# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:32:45 2021

@author: zju
"""

protein_seq = {}
with open('CeleganDrosophilaEcoli.dictionary.tsv', 'r') as r:
    line = r.readline()
    while line != '':
        term_list = line.split('\t')
        protein_seq[term_list[0]] = term_list[1].strip('\n')
        line = r.readline()
ipq = 0
imn = 0
protein_count = []
protein_k = {}

with open('CeleganDrosophilaEcoli.actions.tsv', 'r') as r:
    line = r.readline()
    while line != '':
        name_1 = line.split('\t')[0]
        name_2 = line.split('\t')[1]
        cla = line.split('\t')[2].strip()
        
        if (len(protein_seq[name_1]) <= 1500) and (len(protein_seq[name_2]) <= 1500):
            protein_k[name_1 + '\t' + name_2] = cla
            if name_1 not in protein_count:
                protein_count.append(name_1)
            if name_2 not in protein_count:
                protein_count.append(name_2)
        line = r.readline()
        
for key in list(protein_k.keys()):
    if protein_k[key] == '1':
        ipq += 1
    elif protein_k[key] == '0':
        imn += 1
        
#print(len(list(protein_k.keys())))
print('the number of proteins: ' + str(len(protein_count)))
print(ipq)
print(imn)