from keras.initializers import RandomNormal
from keras.layers import Dense, Dropout, Multiply, Add, \
    Concatenate, Reshape, LocallyConnected1D, Flatten
from keras.models import Model
from Layers import GradientReversal

#
# defines the blocks for the models.
# It is important to always return a descriptor of the OUTPUT shape
# of the model, such that it can be used to define the INPUT to a subsequent
# block consistently
#


def common_features(Inputs, dropoutrate=0.03):
    X = Inputs
    X = Dense(30, activation='relu', name='common_dense_0')(X)
    X = Dropout(dropoutrate)(X)
    X = Dense(20, activation='relu', name='common_dense_1')(X)
    X = Dropout(dropoutrate)(X)
    X = Dense(20, activation='relu', name='common_dense_2')(X)
    X = Dropout(dropoutrate)(X)
    X = Dense(15, activation='relu', name='common_dense_3')(X)
    X = Dropout(dropoutrate)(X)
    return Model(inputs=Inputs, outputs=X, name='common_features')


def datamc_discriminator(XInputs, rev_grad=10, dropoutrate=0.03, weight_rev=10.):
    Ad = GradientReversal(hp_lambda=rev_grad, name='ada_gradrev')(XInputs[0])
    Ad = Dense(20, activation='relu', name='ada_dense_0')(Ad)
    Ad = Dropout(dropoutrate)(Ad)
    Ad = Dense(15, activation='relu', name='ada_dense_1')(Ad)
    Ad = Dropout(dropoutrate)(Ad)
    Ad = Dense(15, activation='relu', name='ada_dense_2')(Ad)
    Ad = Dropout(dropoutrate)(Ad)
    Ad = Dense(10, activation='relu', name='ada_dense_3')(Ad)
    Ad = Dropout(dropoutrate)(Ad)
    Ad = Dense(1, activation='sigmoid', name='ada_out')(Ad)
    # just for compatibility for now
    Ad1 = Dense(
        1, activation='linear', use_bias=False, trainable=False,
        name='Add1')(Ad)

    #make tunable weight
    weight = Reshape((1,1),name='weight_reshape')(XInputs[1])
    weight = LocallyConnected1D(
	1,1, 
	activation='linear', use_bias=False, 
	kernel_initializer='zeros',
	name="weight_layer",
	#kernel_regularizer=boundary_regularizer,
	) (weight)    
    weight= Flatten()(weight)
    weight = GradientReversal(name='weight_reversal', hp_lambda=weight_rev)(weight)
    Ad = Concatenate(name='ada_weight')([Ad, weight]) 

    return Model(inputs=XInputs, outputs=[Ad, Ad1], name="datamc_discriminator")


def btag_discriminator(XInputs, dropoutrate=0.03):
    X = XInputs
    X = Dense(15, activation='relu')(X)
    X = Dropout(dropoutrate)(X)
    X = Dense(15, activation='relu')(X)
    X = Dropout(dropoutrate)(X)
    X = Dense(1, activation='sigmoid', name='mc')(X)
    X1 = Dense(
        1, activation='linear', use_bias=False, trainable=False,
        kernel_initializer='Ones', name='data')(X)
    return Model(inputs=XInputs, outputs=[X, X1], name="btag_discriminator")


def mc_correction(XInputsList, nodes, layers=1, dropoutrate=0.03):

    # the output shape of the correction should be the same as the input
    outshape = XInputsList[0].get_shape().as_list()[1]

    X = XInputsList[0]
    isMC = XInputsList[1]
    noise = XInputsList[2]
    isBs = XInputsList[3]

    X = Concatenate()([X, noise, isBs])

    for i in range(layers):
        X = Dense(nodes, activation='relu')(X)
        X = Dropout(dropoutrate)(X)

    X = Dense(outshape, activation='relu',
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(X)
    X = Dropout(dropoutrate)(X)
    X = Multiply()([X, isMC])
    X = Add()([XInputsList[0], X])
    return Model(inputs=XInputsList, outputs=X, name="mc_correction")
