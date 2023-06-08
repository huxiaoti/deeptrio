import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from build_my_layer import MyMaskCompute, MySpatialDropout1D
import numpy as np
import os
import argparse
import warnings
import json
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='run DeepTrio for PPI prediction')
parser.add_argument('-p1', '--protein1', required=True, type=str, help='configuration of the first protein group in fasta format with its path, which can contain multiply sequences')
parser.add_argument('-p2', '--protein2', required=True, type=str, help='configuration of the second protein group in fasta format with its path, whcih can contain multiply sequences')
parser.add_argument('-m', '--model',  required=True, type=str, help='configuration of the DeepTrio model with its path')
parser.add_argument('-o', '--output', default='default', type=str, help='configuration of the name of output without a filename extension')

static_args = parser.parse_args()

error_report = 0

file_1_path = static_args.protein1
file_2_path = static_args.protein2
file_output = static_args.output + '.txt'

model_path = static_args.model

print('\nWelcome to use our tool')
print('\nVersion: 1.0.0')
print('\nAny problem, please contact mchen@zju.edu.cn')
print('\nStart to process the raw data')
# if static_args.model in ['human','yeast','general']:
#     pass
# else:
#     error_report = 1
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

p1_group = read_file(file_1_path)
p2_group = read_file(file_2_path)

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
    seq_list = []
    for name in name_list:
        seq = to_arr(protein_group[name])
        seq = pad_arr(seq)
        seq_list.append(seq)
    return seq_list
        
p1_seq_list = convert_numerical_list(p1_name_list, p1_group)
p2_seq_list = convert_numerical_list(p2_name_list, p2_group)

group_seq_1 = []
group_seq_2 = []

group_name = []

for n1 in range(len(p1_name_list)):
    for n2 in range(len(p2_name_list)):
        group_seq_1.append(p1_seq_list[n1])
        group_seq_2.append(p2_seq_list[n2])
        group_name.append(p1_name_list[n1] + '\t' + p2_name_list[n2])

group_arr_1 =  np.array(group_seq_1)
group_arr_2 =  np.array(group_seq_2)

# print(group_arr_1)
# print(group_arr_2)
# print(group_arr_1.shape)
# print(group_arr_2.shape)
# print(group_name)

print('\nModel loading')
model = tf.keras.models.load_model(model_path, custom_objects={'MyMaskCompute':MyMaskCompute, 'MySpatialDropout1D':MySpatialDropout1D})

predictions_test = model.predict([group_arr_1, group_arr_2])

# print(predictions_test)

# with open(file_output, 'w') as w:
#     for n1 in range(len(predictions_test)):
#         w.write(group_name[n1])
#         for n2 in range(len(predictions_test[n1])):
#             w.write('\t' + str(predictions_test[n1][n2]))
#         w.write('\n')
print('\nPrediction results:')

with open(file_output, 'w') as w:
    w.write('protein_1\tprotein_2\tprobability\tresult\n')
    print('protein_1\tprotein_2\tprobability\tresult')
    output_data = {}
    for n in range(len(predictions_test)):
        output_data[group_name[n]] = {}
        # output_data[group_name[n]]['model'] = static_args.model
        output_data[group_name[n]]['binding_probability'] = str(predictions_test[n][0])
        if predictions_test[n][0] >= 0.5:
            output_data[group_name[n]]['result'] = 'binding'
        elif predictions_test[n][2] >= 0.5:
            output_data[group_name[n]]['result'] = 'single-protein'
        else:
            output_data[group_name[n]]['result'] = 'non-binding'
        p1_name = group_name[n].split('\t')[0].lstrip('>')
        p2_name = group_name[n].split('\t')[1].lstrip('>')
        w.write(p1_name + '\t' + p2_name + '\t')
        print(p1_name + '\t' + p2_name + '\t', end='')
        for key in list(output_data[group_name[n]].keys()):
            w.write(output_data[group_name[n]][key] + '\t')
            print(output_data[group_name[n]][key] + '\t', end='')
        print('')
        w.write('\n')
path = os.getcwd()
print('predcition file is saved in ' + path)
print(file_output)

print('\nThank you for using')
