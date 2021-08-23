# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:32:21 2021

@author: zju
"""
import numpy as np

def preprocess(pair_file, seq_file):
    
    with open(pair_file, 'r') as f:
        
        lines = f.readlines()
        
    proteins_1 = [line.strip().split('\t')[0] for line in lines]
    proteins_2 = [line.strip().split('\t')[1] for line in lines]
    labels = [line.strip().split('\t')[2] for line in lines]
    
    protein_list = list(set(proteins_1 + proteins_2))
            
    protein_seq = {}
           
    with open(seq_file, 'r') as f:
        
        lines = f.readlines()
        
    for i in range(len(lines)):
        
        line = lines[i].strip().split('\t')
        protein_seq[line[0]] = line[1]
        
#    return proteins_1, proteins_2, labels, protein_seq
    amino_acid ={'A':1,'C':2,'D':3,'E':4,'F':5,
                'G':6,'H':7,'I':8,'K':9,'L':10,
                'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,
                'T':17,'V':18,'W':19,'Y':20,'U':21,'X':22,'B':0}

# positive and negative setting
    k1 = []
    k2 = []
    k3 = []
    k_h = []

    for i in range(len(labels)):

        protein_1 = proteins_1[i]
        protein_2 = proteins_2[i]
        
        label = labels[i]
        
        seq_1 = protein_seq[protein_1]
        seq_2 = protein_seq[protein_2]

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
        else:
            print('error')
            break
        k3.append(a3)
        
        k_h.append(np.array([protein_1, protein_2]))
    
    m1 = np.stack(k1, axis=0)
    m2 = np.stack(k2, axis=0)
    m3 = np.stack(k3, axis=0)
    m_h = np.stack(k_h, axis=0)

# single protein setting
    k1 = []
    k2 = []
    k3 = []

    for protein in protein_list:

        seq_1 = protein_seq[protein]
        seq_2 = 'B'
        label = 2
    
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
    
    return m1, m2, m3, n1, n2, n3, m_h