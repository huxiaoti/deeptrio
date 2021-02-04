import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from build_my_layer import MyMaskCompute, MySpatialDropout1D
import os
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from colour import Color
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import rcParams
import matplotlib
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='run DeepTrio for visualizarion')
parser.add_argument('-p1', '--protein1', required=True, type=str, help='configuration of the first protein in fasta format with path')
parser.add_argument('-p2', '--protein2', required=True, type=str, help='configuration of the second protein in fasta format with path')
parser.add_argument('-m', '--model',  required=True, type=str, help='configuration of the DeepTrio model with its path')
# parser.add_argument('-o', '--output', default='default', type=str, help='configuration of the name of output without a filename extension')

static_args = parser.parse_args()

error_report = 0

file_1_path = './' + static_args.protein1
file_2_path = './' + static_args.protein2

model_path = static_args.model

print('\nWelcome to use our tool')
print('\nVersion: 1.0.0')
print('\nAny problem, please contact mchen@zju.edu.cn')
print('\nStart to process the raw data')

def read_file(file_path):
    namespace = {}
    with open(file_path, 'r') as r:
        line = r.readline()
        while line != '':
            if line.startswith('>'):
                name = line.strip()
                namespace[name] = ''
                line = r.readline()
            else:
                namespace[name] += line.strip().upper()
                line = r.readline()
    return namespace

def arr_split_n(arr):
    threshold = arr.shape[0]
    b = np.array([0])    
    list_n = []
    arr_n = []
    length = arr.shape[0]
    len_each = int(np.ceil(length/threshold))
    length = np.nonzero(arr)[0].shape[0]
    gap = np.tile(b, 1)
    for n in range(threshold):
        if n != threshold-1:
            list_n.append(arr.copy()[n*len_each:(n+1)*len_each])
        else:
            list_n.append(arr.copy()[n*len_each:])
            
    rate = 0 #**********
    for n1 in range(length - rate):        
        mutant_n = []        
        for n2 in range(threshold):
            if n1 <= n2 <= n1 + rate:
                mutant_n.append(gap)
            else:
                mutant_n.append(list_n[n2])                
        arr_n.append(np.concatenate(mutant_n,axis=None))    
    arr_n = np.stack(arr_n, axis=0)
        
    return arr_n

p1_group = read_file(file_1_path)
if len(p1_group.keys()) != 1:
    raise Exception('The format of protein_1 is incorrect')
p2_group = read_file(file_2_path)
if len(p2_group.keys()) != 1:
    raise Exception('The format of protein_2 is incorrect')

p1_name_list = list(p1_group.keys())
p2_name_list = list(p2_group.keys())
        
def to_arr(seq):
    amino_acid ={'A':1,'C':2,'D':3,'E':4,'F':5,
                 'G':6,'H':7,'I':8,'K':9,'L':10,
                 'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,
                 'T':17,'V':18,'W':19,'Y':20,'U':21,'X':22,'B':0}

    length = len(seq)
    a1 = np.zeros([length,], dtype = int)
    k = 0
    for AA in seq:
        a1[k] = amino_acid[AA]
        k += 1
    return a1

def pad_arr(arr):
    arr=np.pad(arr,(0,1500-len(arr)))
    return arr


def convert_numerical_list(name_list, protein_group):

    for name in name_list:
        seq = to_arr(protein_group[name])
        seq = pad_arr(seq)

    return seq


p1_arr_seq = convert_numerical_list(p1_name_list, p1_group)
p2_arr_seq = convert_numerical_list(p2_name_list, p2_group)

len1 = np.nonzero(p1_arr_seq)[0].shape[0]
len2 = np.nonzero(p2_arr_seq)[0].shape[0]

ac1 = p1_arr_seq.copy()
ac2 = p2_arr_seq.copy()

am1 = arr_split_n(ac1)[:len1, :]
am2 = arr_split_n(ac2)[:len2, :]

as1 = np.tile(p1_arr_seq, (am2.shape[0],1))
as2 = np.tile(p2_arr_seq, (am1.shape[0],1))

ae1 = np.expand_dims(p1_arr_seq, axis=0)
ae2 = np.expand_dims(p2_arr_seq, axis=0)

print('\nModel loading')
model = tf.keras.models.load_model(model_path, custom_objects={'MyMaskCompute':MyMaskCompute, 'MySpatialDropout1D':MySpatialDropout1D})

output1 = model.predict([am1,as2], verbose = 0)

output2 = model.predict([am2,as1], verbose = 0)

output_standerd = model.predict([ae1,ae2], verbose = 0)

baseline = output_standerd[0][1]
b_1 = output1[:,1] - baseline
b_2 = output2[:,1] - baseline

df_1_max = b_1.max()
df_1_min = b_1.min()

df_1_seq = p1_group[p1_name_list[0]]
df_2_seq = p2_group[p2_name_list[0]]


if df_1_max > (-4 * df_1_min):
    v_1_max = -4 * df_1_min
else:
    v_1_max = df_1_max

df_2_max = b_2.max()
df_2_min = b_2.min()

if df_2_max > (-4 * df_2_min) and df_2_min < 0:
    v_2_max = -4 * df_2_min
else:
    v_2_max = df_2_max

distance = 20

while len(b_1) % distance:
    b_1 = np.concatenate([b_1,np.array([-10])])
    
df_1 = pd.DataFrame()
for n in range(int(len(b_1)/distance)):
    df_1[str(n)] = b_1[n*20:(n+1)*20]
    
df_1 = df_1.T

while len(b_2) % distance:
    b_2 = np.concatenate([b_2,np.array([-10])])
    
df_2 = pd.DataFrame()
for n in range(int(len(b_2)/distance)):
    df_2[str(n)] = b_2[n*20:(n+1)*20]
    
df_2 = df_2.T


p1_file =  str(p1_name_list[0]).lstrip('>')
p2_file =  str(p2_name_list[0]).lstrip('>')

def draw(pandas_data, seque, v_max, output_name_1, output_name_2):
    prtein_len = len(seque)

    if (prtein_len / 20) > (np.floor(prtein_len / 20)):
        row_number = int(np.floor(prtein_len / 20) +1)
    else:
        row_number = int(np.floor(prtein_len / 20))
    row_index = []
    for n in range(1,20 * row_number,20):
        row_index.append(int(n))

    xx=np.zeros((row_number,20))
    xx=xx.astype(np.str)
    ik = 0
    for n1 in range(row_number):
        for n2 in range(20):
            if ik < len(seque):
                xx[n1][n2] = seque[ik]
                ik += 1

    matplotlib.use('Agg')           
    fig, ax = plt.subplots(figsize=(0.667 * row_number, 12))
    sns.set_style('white')
    cdict = [(0,'#0000ff'),(0.5,'#FFFFFF'), (1,'#ff0000')]
    col=LinearSegmentedColormap.from_list('',cdict)

    sns.heatmap(data=pandas_data, linewidths = 0.1, annot=xx, annot_kws = {'fontsize':15}, fmt='', linecolor = '#DCDCDE', vmax=v_max, vmin=-v_max, ax=ax, cmap=col, yticklabels=1, square=True, mask=(pandas_data<=-10), cbar=False)# , cbar_kws={'shrink':0.5, 'aspect':10, 'pad':0.08})

    #row_index = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281, 301, 321, 341, 361, 381, 401, 421, 441]

    ax.set_ylim([row_number, 0])
    ax.set_facecolor('#DCDCDE')
    ax.set_yticklabels(row_index, rotation=360)
    
    ax.tick_params(bottom = False, labelbottom = False)
    for edge in ['top','bottom','left','right']:
        ax.spines[edge].set_visible(True)
        ax.spines[edge].set_color('black')

    ax.tick_params(axis='y', labelsize=16, pad = 0.5)
    visualization_name = output_name_1 + '_with_respect_to_' + output_name_2 + '_importance_map.svg'   
    fig.savefig(visualization_name, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    
    return visualization_name
print('\nStart to draw importance maps ...')
name_1 = draw(df_1, df_1_seq, v_1_max, p1_file, p2_file)
import importlib
importlib.reload(matplotlib); importlib.reload(plt); importlib.reload(sns)

name_2 = draw(df_2, df_2_seq, v_2_max, p2_file, p1_file)

path = os.getcwd()
print('\nCongratulations, the visualization results are saved in ' + path)
print(name_1 + '\t' + name_2)