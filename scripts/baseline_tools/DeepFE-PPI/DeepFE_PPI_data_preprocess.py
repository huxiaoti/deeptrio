# -*- coding: utf-8 -*-

"""
Created on Wed Oct 24 09:54:33 2018

@author: yaoyu
"""

import os
from time import time
from keras import Model
from keras.models import Sequential
from keras.layers import *
#from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_curve, auc, roc_auc_score,average_precision_score
import numpy as np
#from keras.layers.core import Dense, Dropout
import utils.tools as utils
from keras.regularizers import l2
from gensim.models import Word2Vec
import copy
import h5py
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
#import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras.optimizers import SGD

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)
'''
def plot(length):
    reversed_length =  sorted(length,reverse=True)
    x = np.linspace(0, len(length), len(length))
    plt.plot(x, reversed_length)
     
    plt.title('line chart')
    plt.xlabel('x')
    plt.ylabel('reversed_length')
    
    plt.show()  
'''    
   
def max_min_avg_length(seq):
    length = []
    for string in seq:
        length.append(len(string))
    #plot(length)    
    maxNum = max(length) #maxNum = 5
    minNum = min(length) #minNum = 1
    
    avg = averagenum(length)
    
    print('The longest length of protein is: '+str(maxNum))
    print('The shortest length of protein is: '+str(minNum))
    print('The avgest length of protein is: '+str(avg))

def merged_DBN(sequence_len):
    # left model
    model_left_in = Input(shape=(sequence_len,))
    model_left = Dense(2048, activation='relu',kernel_regularizer=l2(0.01))(model_left_in)
    model_left = BatchNormalization()(model_left)
    model_left = Dropout(0.5)(model_left)
   
    model_left = Dense(1024, activation='relu',kernel_regularizer=l2(0.01))(model_left)
    model_left = BatchNormalization()(model_left)
    model_left = Dropout(0.5)(model_left)
    model_left = Dense(512, activation='relu',kernel_regularizer=l2(0.01))(model_left)
    model_left = BatchNormalization()(model_left)
    model_left = Dropout(0.5)(model_left)
    model_left = Dense(128, activation='relu',kernel_regularizer=l2(0.01))(model_left)
    model_left = BatchNormalization()(model_left)
    
    
    # right model
    model_right_in = Input(shape=(sequence_len,))
    model_right = Dense(2048,activation='relu',kernel_regularizer=l2(0.01))(model_right_in)
    model_right = BatchNormalization()(model_right)
    model_right = Dropout(0.5)(model_right)
     
    model_right = Dense(1024, activation='relu',kernel_regularizer=l2(0.01))(model_right)
    model_right = BatchNormalization()(model_right)
    model_right = Dropout(0.5)(model_right)
    model_right = Dense(512, activation='relu',kernel_regularizer=l2(0.01))(model_right)
    model_right = BatchNormalization()(model_right)
    model_right = Dropout(0.5)(model_right)
    model_right = Dense(128, activation='relu',kernel_regularizer=l2(0.01))(model_right)
    model_right = BatchNormalization()(model_right)
    # together
    merged = Concatenate()([model_left, model_right])
    
    x = Dense(8, activation='relu',kernel_regularizer=l2(0.01))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs = [model_left_in, model_right_in], outputs = x)

    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    #model.summary()
    
    return model

'''    
# define the function
def training_vis(hist,i,plot_dir,swm,be):
    loss = hist.history['loss']
    #val_loss = hist.history['val_loss']
    acc = hist.history['precision']
    #val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    #ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Traingng Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_precision')
    #ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision on Traingng Data')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + swm+be+'/round_'+str(i)+'.png')
'''    
    
def read_file(file_name):
    pro_swissProt = []
    with open(file_name, 'r') as fp:
        protein = ''
        for line in fp:
            if line.startswith('>sp|'):
                pro_swissProt.append(protein)
                protein = ''
            elif line.startswith('>sp|') == False:
                protein = protein+line.strip()
              
    return   pro_swissProt[1:]   

'''    
def get_res2vec_data():
    file_name = 'dataset/uniprot_sprot.fasta'
    pro_swissProt = read_file(file_name)
    
    return pro_swissProt
'''    

    
def token(dataset):
    token_dataset = []
    for i in range(len(dataset)):
        seq = []
        for j in range(len(dataset[i])):
            seq.append(dataset[i][j])

        token_dataset.append(seq)  
                
    return  token_dataset
    
def pandding_J(protein,maxlen):           
    padded_protein = copy.deepcopy(protein)   
    for i in range(len(padded_protein)):
        if len(padded_protein[i])<maxlen:
            for j in range(len(padded_protein[i]),maxlen):
                padded_protein[i].append('J')
    return padded_protein  
    

def protein_representation(wv,tokened_seq_protein,maxlen,size):  
    represented_protein  = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        for j in range(maxlen):
            if tokened_seq_protein[i][j]=='J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))    
    return represented_protein
    
def read_traingingData(file_name):
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==1:
                seq.append(line.split('\n')[0])
            i = i+1       
    return   seq   


              
def get_training_dataset(wv,  maxlen,size):

    file_1 = 'human_P_A_new.fasta'
    file_2 = 'human_P_B_new.fasta'
    file_3 = 'human_N_A_new.fasta'
    file_4 = 'human_N_B_new.fasta'
    # positive seq protein A
    pos_seq_protein_A = read_traingingData(file_1)
    pos_seq_protein_B = read_traingingData(file_2)
    neg_seq_protein_A = read_traingingData(file_3)
    neg_seq_protein_B = read_traingingData(file_4)
    # put pos and neg together
    pos_neg_seq_protein_A = copy.deepcopy(pos_seq_protein_A)   
    pos_neg_seq_protein_A.extend(neg_seq_protein_A)
    pos_neg_seq_protein_B = copy.deepcopy(pos_seq_protein_B)   
    pos_neg_seq_protein_B.extend(neg_seq_protein_B)
    seq = []
    seq.extend(pos_neg_seq_protein_A)
    seq.extend(pos_neg_seq_protein_B)
    max_min_avg_length(seq)
    
    
    # token
    token_pos_neg_seq_protein_A = token(pos_neg_seq_protein_A)
    token_pos_neg_seq_protein_B = token(pos_neg_seq_protein_B)
    # padding
    tokened_token_pos_neg_seq_protein_A = pandding_J(token_pos_neg_seq_protein_A, maxlen)
    tokened_token_pos_neg_seq_protein_B = pandding_J(token_pos_neg_seq_protein_B,maxlen)
    # protein reprsentation
    feature_protein_A  = protein_representation(wv,tokened_token_pos_neg_seq_protein_A,maxlen,size)
    feature_protein_B  = protein_representation(wv,tokened_token_pos_neg_seq_protein_B,maxlen,size)
    feature_protein_AB = np.hstack((np.array(feature_protein_A),np.array(feature_protein_B)))
    #  creat label
    label = np.ones(len(feature_protein_A))
    label[len(feature_protein_AB)//2:] = 0
   
    return feature_protein_AB,label
    #
'''    
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                 
        os.makedirs(path)           
        print("---  new folder...  ---")
        print("---  OK  ---")
 
    else:
        print("---  There is this folder!  ---")
'''        

                                                 
#%%  
if __name__ == "__main__": 
   
    # load dictionary
    model_wv = Word2Vec.load('wv_swissProt_size_20_window_4.model')
    
  
    #plot_dir = 'plot/11188/'
    
    sizes = [20]
    windows = [4]
    maxlens = [850]  # 550,650,750,850
    batch_sizes = [256] # 32,64,128,256
    nb_epoches = [150] 
    
    size = sizes[0]
    window = windows[0]
    maxlen = maxlens[0]
    batch_size = batch_sizes[0]
    nb_epoch = nb_epoches[0]

    sequence_len = size*maxlen

    print(sequence_len) #17000

    # get training data 
    t_start = time()
    '''
    h5_file = h5py.File('dataset/11188/wv_swissProt_size_20_window_4_maxlen_'+str(maxlen)+'.h5','r')
    train_fea_protein_AB =  h5_file['trainset_x'][:]
    train_label = h5_file['trainset_y'][:]
    h5_file.close()
    '''
    train_fea_protein_AB,train_label= get_training_dataset(model_wv.wv, maxlen,size)  
    np.save('train_fea_protein_AB', train_fea_protein_AB)
    np.save('ep_ppi_label', train_label)