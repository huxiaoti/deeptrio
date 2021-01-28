from sklearn.model_selection import KFold, ShuffleSplit

def random_arr(arr):
    np.random.seed(5)
    np.random.shuffle(arr)
    np.random.seed(55)
    np.random.shuffle(arr)

x_a_1 = {}
x_t_1 = {}
x_a_2 = {}
x_t_2 = {}
y_a = {}
y_t_s = {}
kfnum = 0
spl_num = 5
kf = KFold(n_splits=spl_num, shuffle=True, random_state=5)
for train_index, test_index in kf.split(y_train):
    x_a_1[kfnum], x_t_1[kfnum] = x_train_1[train_index], x_train_1[test_index]
    x_a_2[kfnum], x_t_2[kfnum] = x_train_2[train_index], x_train_2[test_index]
    y_a[kfnum], y_t_s[kfnum] = y_train[train_index], y_train[test_index]
    kfnum += 1

for ks in range(spl_num):
    x_a_1[ks] = np.concatenate([s_1, x_a_1[ks]], 0)
    x_a_2[ks] = np.concatenate([s_2, x_a_2[ks]], 0)
    y_a[ks] = np.concatenate([s_y, y_a[ks]], 0)
    random_arr(x_a_1[ks])
    random_arr(x_a_2[ks])
    random_arr(y_a[ks])