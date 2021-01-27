# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:32:21 2021

@author: zju
"""
import numpy as np

def preprocess(file_1, file_2):
    protein_pairs = {}
    protein_1_list = []
    protein_2_list = []
    
    
    with open(file_1, 'r') as r:
        line = r.readline()
        while line != '':
            protein_1 = line.strip().split('\t')[0]
            protein_1_list.append(protein_1)
            protein_2 = line.strip().split('\t')[1]
            protein_2_list.append(protein_2)
            label = line.strip().split('\t')[2]
            protein_pairs[protein_1 + '\t' + protein_2] = label
            line = r.readline()
            
    all_proteins = protein_1_list + protein_2_list
    all_proteins = list(set(all_proteins))
            
    protein_seq = {}
           
    with open(file_2, 'r') as r2:
        line = r2.readline()
        while line != '':
            name = line.split('\t')[0]
            seq = line.strip().split('\t')[1]
            protein_seq[name] = seq
            line = r2.readline()
            
    
    #with open('single_protein_2021.txt', 'r') as f:
    for key in list(protein_pairs.keys()):
        amino_acid ={'A':1,'C':2,'D':3,'E':4,'F':5,
                     'G':6,'H':7,'I':8,'K':9,'L':10,
                     'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,
                     'T':17,'V':18,'W':19,'Y':20,'U':21,'X':22,'B':0}
    
        #line = f.readline()
        name_1 = key.split('\t')[0]
        name_2 = key.split('\t')[1]
        label = protein_pairs[key]
        seq_1 = protein_seq[name_1]
        seq_2 = protein_seq[name_2]
        k1 = []
        k2 = []
        k3 = []
    
        a1 = np.zeros([1500,], dtype = int)
        a2 = np.zeros([1500,], dtype = int)
        a3 = np.zeros([3,], dtype = float)
    
        k = 0
        for AA in seq_1:
            a1[k] = amino_acid[AA]
            k += 1
        k1.append(a1)
    
        k = 0
        for AA in seq_2:
            a2[k] = amino_acid[AA]
            k += 1
        k2.append(a2)
        
    
        if int(label) == 0:
            a3[1] = 1
        elif int(label) == 1:
            a3[0] = 1
    #    elif int(label) == 2:
    #        a3[2] = 1
        else:
            print('error')
            break
        k3.append(a3)
    
    m1 = np.stack(k1, axis=0)
    m2 = np.stack(k2, axis=0)
    m3 = np.stack(k3, axis=0)
    
    for key in all_proteins:
        amino_acid ={'A':1,'C':2,'D':3,'E':4,'F':5,
                     'G':6,'H':7,'I':8,'K':9,'L':10,
                     'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,
                     'T':17,'V':18,'W':19,'Y':20,'U':21,'X':22,'B':0}
    
        #line = f.readline()
        name_1 = key
        name_2 = key
        label = 2
        seq_1 = protein_seq[key]
        seq_2 = 'B'
        k1 = []
        k2 = []
        k3 = []
    
        a1 = np.zeros([1500,], dtype = int)
        a2 = np.zeros([1500,], dtype = int)
        a3 = np.zeros([3,], dtype = float)
    
        k = 0
        for AA in seq_1:
            a1[k] = amino_acid[AA]
            k += 1
        k1.append(a1)
    
        k = 0
        for AA in seq_2:
            a2[k] = amino_acid[AA]
            k += 1
        k2.append(a2)
        
        if int(label) == 2:
            a3[2] = 1
        else:
            print('error')
            break
        k3.append(a3)
    
    n1 = np.stack(k1, axis=0)
    n2 = np.stack(k2, axis=0)
    n3 = np.stack(k3, axis=0)
    
    return m1, m2, m3, n1, n2, n3
#np.save('single_protein_1', n1)
#np.save('single_protein_2', n2)
#np.save('single_protein_y', n3)