import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

def random_arr(arr):

    np.random.seed(5)
    np.random.shuffle(arr)
    np.random.seed(55)
    np.random.shuffle(arr)

def array_split(num, arr_ppi_1, arr_ppi_2, arr_ppi_y, arr_single_1, arr_single_2, arr_single_y):

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
        kfnum += 1

    for ks in range(spl_num):
        x_a_1[ks] = np.concatenate([arr_single_1, x_a_1[ks]], 0)
        x_a_2[ks] = np.concatenate([arr_single_2, x_a_2[ks]], 0)
        y_a[ks] = np.concatenate([arr_single_y, y_a[ks]], 0)
        random_arr(x_a_1[ks])
        random_arr(x_a_2[ks])
        random_arr(y_a[ks])

    return x_a_1, x_a_2, y_a, x_t_1, x_t_2, y_t_s