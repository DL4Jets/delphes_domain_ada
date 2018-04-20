
import keras
from keras.layers import Dense, Concatenate ,Dropout
from keras.layers import Input
from keras.models import Model
from Layers import GradientReversal

#
# defines the blocks for the models.
# It is important to always return a descriptor of the OUTPUT shape
# of the model, such that it can be used to define the INPUT to a subsequent
# block consistently
#

def common_features(Inputs, dropoutrate=0.03):
    X=Inputs
    X = Dense(30, activation='relu', name='common_dense_0')(X)
    X = Dropout(dropoutrate)(X)

    X = Dense(20, activation='relu')(X)
    X = Dropout(dropoutrate)(X)

    X = Dense(20, activation='relu')(X)
    X = Dropout(dropoutrate)(X)
    
    X = Dense(15, activation='relu')(X)
    X = Dropout(dropoutrate)(X)
    
    #please always define and return this (shape of the last layer)
    outputDescription=Input((15,))
    
    return Model(inputs=Inputs, outputs=X, name = 'common_features'), outputDescription


def datamc_discriminator(XInputs, rev_grad=10, dropoutrate=0.03):
    
    Ad = GradientReversal(hp_lambda=rev_grad, name='Ad_gradrev')(XInputs)
    Ad = Dense(20, activation='relu')(Ad)
    Ad = Dropout(dropoutrate)(Ad)
    Ad = Dense(15, activation='relu')(Ad)
    Ad = Dropout(dropoutrate)(Ad)
    Ad = Dense(15, activation='relu')(Ad)
    Ad = Dropout(dropoutrate)(Ad)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dropout(dropoutrate)(Ad)
    Ad = Dense(1, activation='sigmoid', name = 'Add' )(Ad)
    
    #just for compatibility for now
    Ad1 = Dense(1, activation='linear',use_bias=False,trainable=False  , name = 'Add1' )(Ad)
    
    outputDescription=[Input((1,)),Input((1,))]
    
    return Model(inputs=XInputs, outputs=[Ad,Ad1], name="datamc_discriminator"),outputDescription

def btag_discriminator(XInputs, dropoutrate=0.03):
    X = XInputs
    X = Dense(15, activation='relu')(X)
    X = Dropout(dropoutrate)(X)
    
    X = Dense(15, activation='relu')(X)
    X = Dropout(dropoutrate)(X)
    
    X = Dense(1, activation='sigmoid', name = 'mc')(X)
    
    X1= Dense(1, activation='linear',use_bias=False, trainable=False,
            kernel_initializer='Ones', name = 'data') (X)
    
    outputDescription=[Input((1,)),Input((1,))]
            
    return Model(inputs=XInputs, outputs=[X,X1], name="btag_discriminator"),outputDescription



