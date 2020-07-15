from __future__ import print_function
import numpy as np
import pandas as pd
import RNN_property_predictor
from molecule_feature_prediction.feature import molecules
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import argparse
try:
    import tensorflow.compat.v1 as tf 
    tf.compat.v1.disable_v2_behavior()
except:
    import tensorflow as tf


def train(input_file, epochs, property, normalize):
    char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", "s", "O",
              "[", "Cl", "Br", "\\"]
    
    data = pd.read_csv(input_file)
    if normalize:
        print("start to normalize")
        scaler_Y = StandardScaler()
        scaler_Y.fit(data[property])
        data[property] = scaler_Y.transform(data[property])
    
    x, ls_smi_new = molecules(data['smiles'].tolist()).one_hot(char_set=char_set)
    y = data[property].values

    np.random.seed(0)
    perm_L = np.random.permutation(x.shape[0])


    x_train, x_test = np.split(x[perm_L], [int(len(x)*0.9)])
    y_train, y_test = np.split(y[perm_L], [int(len(y)*0.9)])
       
    

    print('::: model training')

    seqlen_x = x_train.shape[1]
    dim_x = x_train.shape[2]
    dim_y = y_train.shape[1]
    dim_z = 100
    dim_h = 250

    n_hidden = 3
    batch_size = 200


    model = RNN_property_predictor.Model(seqlen_x = seqlen_x, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, dim_h = dim_h,
                        n_hidden = n_hidden, batch_size = batch_size, char_set = char_set)
    with model.session:
        model.train(trnX_L=x_train, trnY_L=y_train, valX_L=x_test, valY_L=y_test, epochs=epochs)
        # model.saver.save(model.session, "./RNN_model_EA.ckpt")

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
