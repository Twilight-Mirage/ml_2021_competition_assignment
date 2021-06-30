import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepforest import CascadeForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle


def read_one_params_data(path):
    with open(path) as f:
        data = np.loadtxt(path, delimiter='\t', dtype=np.double)
    return data

def read_one_nosiy_data(path):
    with open(path) as f:
        content = f.read()
        content = content.replace("# star_temp: ","").replace("# star_logg: ","").replace("# star_rad: ","").replace("# star_mass: ","").replace("# star_k_mag: ","").replace("# period: ","")
        content = content.replace("\n", "\t")
        data = np.fromstring(content, sep='\t', dtype=np.double)
    return data

def get_file_list():
    files = []
    txtpath = "ml_data_challenge_database\\noisy_train.txt"
    fd = open(txtpath)
    chosen = np.random.randint(0,10)
    for index,line in enumerate(fd):
        if index % 10 == chosen:
            files.append(line[12:-1])
    return files

def get_test_list(i):
    testfiles=[]
    txtpath = "ml_data_challenge_database\\noisy_test.txt"
    fd = open(txtpath)
    for index,line in enumerate(fd):
        if index >= i*20000 and index <= (i+1)*20000:
            testfiles.append(line[11:-1])
    if i == 2:
        testfiles[-1]+='t'
    return testfiles

def walkFile(file, mode, type, i): # 若内存不足，需修改成读部分数据
    data_list = []
    file_list = []
    if type == "train":
        file_list = get_file_list()
    elif type == "test":
        file_list = get_test_list(i)
    for f in tqdm(file_list):
        path = os.path.join(file, f)
        if mode == "noisy":
            data = read_one_nosiy_data(path)
        elif mode == "params":
            data = read_one_params_data(path)
        data_list.append(data)
    return np.array(data_list)

if __name__ == '__main__':

    df_y_file = "df_y.pk"
    if os.path.exists(df_y_file):
        with open(df_y_file, "rb") as f:
            df_y = pickle.load(f)
    else:
        data_list = walkFile("ml_data_challenge_database\\params_train", "params", "train", 0)
        df_y = pd.DataFrame(data_list)
        with open(df_y_file, "wb") as f:
            pickle.dump(df_y, f, protocol = 4)

    df_X_file = "df_X.pk"
    if os.path.exists(df_X_file):
        with open(df_X_file, "rb") as f:
            df_X = pickle.load(f)
    else:
        data_list = walkFile("ml_data_challenge_database\\noisy_train", "noisy", "train", 0)
        df_X = pd.DataFrame(data_list)
        with open(df_X_file, "wb") as f:
            pickle.dump(df_X, f, protocol = 4)

    x_train, x_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.40)

    DF21 = CascadeForestRegressor(n_jobs=32,n_estimators=6,max_layers=5)
    DF21.fit(x_train.values, y_train.values)
    y_pred = DF21.predict(x_test.values)

    '''model = BaggingRegressor(n_jobs=16)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)'''

    
    mse = mean_squared_error(y_test.values, y_pred)
    print("\nTesting MSE of Extra Trees Regressor: {:.9f}".format(mse))
    

    df_test_file = "df_test1.pk"
    if os.path.exists(df_test_file):
        with open(df_test_file, "rb") as f:
            df_test = pickle.load(f)
    else:
        data_list = walkFile("ml_data_challenge_database\\noisy_test", "noisy", "test", 0)
        df_test = pd.DataFrame(data_list)
        with open(df_test_file, "wb") as f:
            pickle.dump(df_test, f, protocol = 4)
    y_pred = DF21.predict(df_test.values)
    df_output = pd.DataFrame(y_pred)
    df_output.to_csv("output.txt", sep='\t',float_format='%.12f', header=False, index=False, mode='a+')
    
    
    df_test_file = "df_test2.pk"
    if os.path.exists(df_test_file):
        with open(df_test_file, "rb") as f:
            df_test = pickle.load(f)
    else:
        data_list = walkFile("ml_data_challenge_database\\noisy_test", "noisy", "test", 1)
        df_test = pd.DataFrame(data_list)
        with open(df_test_file, "wb") as f:
            pickle.dump(df_test, f, protocol = 4)
    y_pred = DF21.predict(df_test.values)
    df_output = pd.DataFrame(y_pred)
    df_output.to_csv("output.txt", sep='\t',float_format='%.12f', header=False, index=False, mode='a+')
    

    df_test_file = "df_test3.pk"
    if os.path.exists(df_test_file):
        with open(df_test_file, "rb") as f:
            df_test = pickle.load(f)
    else:
        data_list = walkFile("ml_data_challenge_database\\noisy_test", "noisy", "test", 2)
        df_test = pd.DataFrame(data_list)
        with open(df_test_file, "wb") as f:
            pickle.dump(df_test, f, protocol = 4)
    y_pred = DF21.predict(df_test.values)
    df_output = pd.DataFrame(y_pred)
    df_output.to_csv("output.txt", sep='\t',float_format='%.12f', header=False, index=False, mode='a+')
    