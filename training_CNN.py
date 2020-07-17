import numpy as np
import pandas as pd
import RNN_property_predictor
from molecule_feature_prediction.feature import molecules
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import argparse
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Flatten, RepeatVector, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras import backend as K
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint

def train(input_file, epochs, property, normalize):

    print("###########",property,"################")

    char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", "s", "O",
              "[", "Cl", "Br", "\\"]

    data = pd.read_csv(input_file)

    if normalize:
        print("start to normalize")
        scaler_Y = StandardScaler()
        scaler_Y.fit(data[property])
        data[property] = scaler_Y.transform(data[property])
    
    x, ls_smi_new, ls_index_new = molecules(data['smiles'].tolist()).one_hot(char_set=char_set)
    y = data[property].iloc[ls_index_new].values
    
    np.random.seed(0)
    perm_L = np.random.permutation(x.shape[0])


    x_train, x_test = np.split(x[perm_L], [int(len(x)*0.9)])
    y_train, y_test = np.split(y[perm_L], [int(len(y)*0.9)])


    
        
    x_in = Input(shape=(x_train.shape[1], x_train.shape[2]), name='input_molecule_smi')
    x = Convolution1D(filters=15, kernel_size=2, activation='tanh', name="encoder_conv0")(x_in)
    x = BatchNormalization(axis=-1, name="encoder_norm0")(x)
    for j in range(1, 3):
        x = Convolution1D(filters=int(15 * 1.15875438383 ** (j)),
                          kernel_size=int(2 * 1.1758149644 ** (j)),
                          activation='tanh',
                          name="encoder_conv{}".format(j))(x)
    x = Flatten()(x)
    hidden_dim = 100
    # Middle layers
    middle = Dense(int(hidden_dim * 1.4928245388), activation='tanh', name='encoder_dense0')(x)
    
    #middle = BatchNormalization(axis=-1, name='encoder_dense0_norm')(middle)

    z_mean = Dense(int(hidden_dim), name='z_mean_sample')(middle)
    prop_mid = Dense(int(hidden_dim/2), activation='tanh')(z_mean)
    
    for p_i in range(1, 3):
        prop_mid = Dense(int((hidden_dim/2) * 0.8 ** p_i),
                            activation='tanh',
                            name="property_predictor_dense{}".format(p_i))(prop_mid)
        #prop_mid = BatchNormalization()(prop_mid)

    reg_prop_pred = Dense(y_test.shape[1], activation='linear',name='reg_property_output')(prop_mid)
    model = Model(inputs=x_in, outputs=reg_prop_pred)

    model.compile(loss='mae', optimizer='adam',metrics=['mae'])
    mcp_save = ModelCheckpoint('CNN_model', save_best_only=True, monitor='val_loss', mode='min')
    print('Training -----------')
    model.fit(x_train, y_train, verbose=1, epochs=epochs, validation_data=[x_test, y_test], callbacks=[mcp_save])
    

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
