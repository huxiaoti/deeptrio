# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:31:00 2021

@author: zju
"""

protein_seq = {}
pair_p = {}
pair_n = {}
   
with open('Protein_A_P.txt', 'r') as r1:
    with open('Protein_B_P.txt', 'r') as r2:
        with open('Protein_A_N.txt', 'r') as r3:
            with open('Protein_B_N.txt', 'r') as r4:
        
                line_1 = r1.readline()
                line_2 = r2.readline()
                line_3 = r3.readline()
                line_4 = r4.readline()
                
                while line_1 != '':
                    name_P_A = line_1.split('|')[1].split(':')[1]
                    if name_P_A == '':
                        name_P_A = line_1.split('|')[0].split(':')[1]
                    name_P_B = line_2.split('|')[1].split(':')[1]
                    if name_P_B == '':
                        name_P_B = line_2.split('|')[0].split(':')[1]
                    name_N_A = line_3.split('|')[1].split(':')[1]
                    if name_N_A == '':
                        name_N_A = line_3.split('|')[0].split(':')[1]
                    name_N_B = line_4.split('|')[1].split(':')[1]
                    if name_N_B == '':
                        name_N_B = line_4.split('|')[0].split(':')[1]
                    
                    seq_P_A = r1.readline()
                    protein_seq[name_P_A] = seq_P_A.strip()
                    seq_P_B = r2.readline()
                    protein_seq[name_P_B] = seq_P_B.strip()
                    seq_N_A = r3.readline()
                    protein_seq[name_N_A] = seq_N_A.strip()
                    seq_N_B = r4.readline()
                    protein_seq[name_N_B] = seq_N_B.strip()


                    if (name_P_A + '\t' + name_P_B) in list(pair_p.keys()):
                        print((name_P_A + '\t' + name_P_B))
                    if (name_N_A + '\t' + name_N_B) in list(pair_n.keys()):
                        print((name_N_A + '\t' + name_N_B))
                        
                    pair_p[name_P_A + '\t' + name_P_B] = '1'
                    pair_n[name_N_A + '\t' + name_N_B] = '0'
                    
                    line_1 = r1.readline()
                    line_2 = r2.readline()                    
                    line_3 = r3.readline()                   
                    line_4 = r4.readline()
                    
#with open('action_pair.tsv', 'w') as w1:
#    for k in list(pair_p.keys()):
#        if (len(protein_seq[k.split('\t')[0]]) <= 1500) and (len(protein_seq[k.split('\t')[1]]) <= 1500):
#            w1.write(k + '\t' + pair_p[k] + '\n')
#    for k in list(pair_n.keys()):
#        if (len(protein_seq[k.split('\t')[0]]) <= 1500) and (len(protein_seq[k.split('\t')[1]]) <= 1500):
#            w1.write(k + '\t' + pair_n[k] + '\n')
#                    
#with open('action_dictionary.tsv', 'w') as w2:
#    for k in list(protein_seq.keys()):
#        w2.write(k + '\t' + protein_seq[k] + '\n')
        
with open('ep_ppi_P_1.txt', 'w') as w1:
    with open('ep_ppi_P_2.txt', 'w') as w2:
        with open('ep_ppi_N_1.txt', 'w') as w3:
            with open('ep_ppi_N_2.txt', 'w') as w4:
                
                for k in list(pair_p.keys()):
                    if (len(protein_seq[k.split('\t')[0]]) <= 1500) and (len(protein_seq[k.split('\t')[1]]) <= 1500):
                        w1.write('>' + k.split('\t')[0] + '\n' + protein_seq[k.split('\t')[0]] + '\n')
                        w2.write('>' + k.split('\t')[1] + '\n' + protein_seq[k.split('\t')[1]] + '\n')
                for k in list(pair_n.keys()):
                    if (len(protein_seq[k.split('\t')[0]]) <= 1500) and (len(protein_seq[k.split('\t')[1]]) <= 1500):
                        w3.write('>' + k.split('\t')[0] + '\n' + protein_seq[k.split('\t')[0]] + '\n')
                        w4.write('>' + k.split('\t')[1] + '\n' + protein_seq[k.split('\t')[1]] + '\n')