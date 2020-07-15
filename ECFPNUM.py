import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from molecule_feature_prediction.feature import molecules

import argparse


def train(input_file, epochs, property, normalize):
    ''' 
    the argument poperty is a list including the output property
    '''  
    data = pd.read_csv(input_file)
    if normalize:
        print("start to normalize")
        scaler_Y = StandardScaler()
        scaler_Y.fit(data[property])
        data[property] = scaler_Y.transform(data[property])
    
    x = molecules(data['smiles'].tolist()).ECFPNUM()
    y = data[property].values

    np.random.seed(0)
    perm_L = np.random.permutation(x.shape[0])


    x_train, x_test = np.split(x[perm_L], [int(len(x)*0.9)])
    y_train, y_test = np.split(y[perm_L], [int(len(y)*0.9)])


        # building model
    model = Sequential()
    model.add(Dense(output_dim=int(x_train.shape[1]/2), input_dim=x_train.shape[1],activation='relu'))
    model.add(Dense(output_dim=int(x_train.shape[1]/6),activation='relu'))
    model.add(Dense(output_dim=int(x_train.shape[1]/12),activation='relu'))
    model.add(Dense(output_dim=int(y_train.shape[1])))
    model.compile(loss='mae', optimizer='adam',metrics=['mae'])
    mcp_save = ModelCheckpoint('ECFPNUM_model', save_best_only=True, monitor='val_loss', mode='min')
    print('Training -----------')
    model.fit(x_train, y_train, verbose=1, epochs=epochs, validation_data=(x_test, y_test), callbacks=[mcp_save])


    for j in range(len(property)):
        if normalize:
            print(property[j],mean_absolute_error(scaler_Y.inverse_transform(y_test)[:,j],scaler_Y.inverse_transform(model.predict(x_test))[:,j]))
            print(property[j],mean_squared_error(scaler_Y.inverse_transform(y_test)[:,j],scaler_Y.inverse_transform(model.predict(x_test))[:,j],squared=False))
        else:
            print(property[j],mean_absolute_error(y_test[:,j],model.predict(x_test)[:,j]))
            print(property[j],mean_squared_error(y_test[:,j],model.predict(x_test)[:,j],squared=False))
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--data_file',
                        help='the file including smiles and corresponding IE and EA', default='MP_clean_canonize_cut.csv')
    parser.add_argument('-e', '--epochs',
                        help='how many epochs would be trained', default=100, type=int)
    parser.add_argument('-p', '--property', nargs='+',
                        help='which property do you want to train')
    
    parser.add_argument('--normalize',action='store_true')
    
    args = vars(parser.parse_args())
    
    if args["normalize"]:
        train(input_file=args["data_file"], epochs=args["epochs"], property=args["property"], normalize=args["normalize"])
    else:
        train(input_file=args["data_file"], epochs=args["epochs"], property=args["property"], normalize=False)
