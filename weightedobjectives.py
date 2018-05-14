from keras import backend as K
from tensorflow import where, greater, abs, zeros_like, exp
import tensorflow as tf
from keras import losses
from keras import metrics
from pdb import set_trace

def tf2np(fcn, session, *args):
   '''tf2np - TensorFlow to NumPy
Allows to use TF-based objective functions with numpy inputs and outputs to ease debugging.
Usage tf2np(fcn, session, *args):
fcn - TF-based function (accepts and handles tensors)
session - TF session
*args - TF-based function arguments, NumPy format
   '''
   args = [tf.convert_to_tensor(i) for i in args]
   return fcn(*args).eval(session=session)

def binary_crossentropy_labelweights(y_true, y_pred):
    """Modification of the binary cross-entropy to use fitted weights, 
which are passed as one additional column(s) to the y_pred
    """
    printAll=False

    if printAll:
        print('in binary_crossentropy_labelweights_Delphes')
    
    # the prediction if it is data or MC is in the first index (see model)
    isMCpred = y_pred[:,:1]

    #the weights are in the remaining parts of the vector
    Weightpred = y_pred[:,1:]
        
    # the truth if it is data or MC
    isMCtrue = y_true[:,:1]
    # labels: B, C, UDSG - not needed here, but maybe later
    # labels_true = y_true[:,1:]
    if printAll:
        print('isMCpred ', isMCpred)
        print('Weightpred ', Weightpred)
        print('isMCtrue ', isMCtrue)

    #only apply label weight deltas to MC, for data will be 1 (+1)
    #as a result of locally connected if will be only !=0 for one label
    #set_trace()
    nmc = K.sum(isMCtrue)
    flat_mc = K.flatten(isMCtrue)
    #compute weights for all
    #set_trace()
    mc_weights = K.clip(flat_mc * K.sum(Weightpred, axis=1)+1, K.epsilon(), 5)
    #remove data and compute normalization
    mc_weights -= (1-flat_mc)
    mc_factor = nmc/K.sum(mc_weights)
    mc_weights *= mc_factor
    #add back data
    weights = mc_weights+(1-K.flatten(isMCtrue))
    weighted_xentr = weights*K.flatten(losses.binary_crossentropy(isMCtrue, isMCpred))
    
    if printAll:
        print('weightsum', weightsum)
        print('weighted_xentr', weighted_xentr.get_shape())

    return weighted_xentr


def binary_accuracy_labelweights(y_true, y_pred):
   '''Modification of the binary accuracy to use fitted weights, 
which are passed as one additional column(s) to the y_pred
   '''
   acc = metrics.binary_accuracy(y_true, y_pred[:,:1])
   fitted_weights = y_pred[:,1:]
   
   #only apply label weight deltas to MC, for data will be 1 (+1)
   #as a result of locally connected if will be only !=0 for one label
   #set_trace()
   nmc = K.sum(y_true)
   flat_mc = K.flatten(y_true)
   #compute weights for all
   mc_weights = K.clip(flat_mc * K.sum(fitted_weights, axis=1)+1, K.epsilon(), 5)
   #remove data and compute normalization
   mc_weights -= (1-flat_mc)
   mc_factor = nmc/K.sum(mc_weights)
   mc_weights *= mc_factor
   #add back data
   weights = mc_weights+(1-K.flatten(flat_mc))
   weighted_acc = tf.cast(weights, tf.float32)*K.flatten(acc)
   return weighted_acc
