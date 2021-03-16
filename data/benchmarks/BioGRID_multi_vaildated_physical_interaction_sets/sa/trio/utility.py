import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

def random_arr(arr):

    np.random.seed(5)
    np.random.shuffle(arr)
    np.random.seed(55)
    np.random.shuffle(arr)

def array_split(num, arr_ppi_1, arr_ppi_2, arr_ppi_y, arr_single_1, arr_single_2, arr_single_y, protein_dict, protein_names):

    x_a_1 = {}
    x_t_1 = {}
    x_a_2 = {}
    x_t_2 = {}
    y_a = {}
    y_t_s = {}
    kfnum = 0
    spl_num = num

    kf = KFold(n_splits=spl_num, shuffle=True, random_state=5)
    for train_index, test_index in kf.split(arr_ppi_y):
        x_a_1[kfnum], x_t_1[kfnum] = arr_ppi_1[train_index], arr_ppi_1[test_index]
        x_a_2[kfnum], x_t_2[kfnum] = arr_ppi_2[train_index], arr_ppi_2[test_index]
        y_a[kfnum], y_t_s[kfnum] = arr_ppi_y[train_index], arr_ppi_y[test_index]
        
        protein_list = []
        protein_seq = protein_dict
        
        for k in train_index:
            if protein_names[k].split('\t')[0] not in protein_list:
                protein_list.append(protein_names[k].split('\t')[0])

            if protein_names[k].split('\t')[1] not in protein_list:
                protein_list.append(protein_names[k].split('\t')[1])

        amino_acid ={'A':1,'C':2,'D':3,'E':4,'F':5,
                     'G':6,'H':7,'I':8,'K':9,'L':10,
                     'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,
                     'T':17,'V':18,'W':19,'Y':20,'U':21,'X':22,'B':0}

        k1 = []
        k2 = []
        k3 = []

        for key in protein_list:

            label = 2
            seq_1 = protein_seq[key]

            seq_2 = 'B'
        
            a1 = np.zeros([2000,], dtype = int)
            a2 = np.zeros([2000,], dtype = int)
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

        print(n3.shape)
        print(arr_single_y.shape)

        x_a_1[kfnum] = np.concatenate([n1, x_a_1[kfnum]], 0)
        x_a_2[kfnum] = np.concatenate([n2, x_a_2[kfnum]], 0)
        y_a[kfnum] = np.concatenate([n3, y_a[kfnum]], 0)

        random_arr(x_a_1[kfnum])
        random_arr(x_a_2[kfnum])
        random_arr(y_a[kfnum])

        kfnum += 1

    return x_a_1, x_a_2, y_a, x_t_1, x_t_2, y_t_s