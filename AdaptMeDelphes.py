from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras import backend as K, callbacks
import sys
import tensorflow as tf
import os
from pdb import set_trace
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
from Layers import *
global_loss_list={}

parser = ArgumentParser()
parser.add_argument('outputDir')
parser.add_argument(
    'method', choices = [
        'MC_training',
        'data_training',
        'stepwise_domain_adaptation',
        'corrected_domain_adaptation'
    ]
)
parser.add_argument("-i",  help="input directory", default='/data/ml/jkiesele/pheno_domAda/numpyx2/', dest='indir')
parser.add_argument("--addsv", action='store_true')
parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
parser.add_argument("--nopred", help="do not compute and store predictions", action='store_true')
parser.add_argument("--lr", help="learning rate", type=float, default=0.0005)
parser.add_argument("--weight", help="domain adaptation weight", type=float, default=2)
parser.add_argument("--lmb", help="domain adaptation lambda", type=float, default=30)
parser.add_argument("--lmb2", help="mc correction lambda", type=float, default=30)
parser.add_argument("--sgdf", help="SGD factor", type=float, default=1)
parser.add_argument("--dainj", help="point to inject", type=int, default=2)
parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=0.17)
parser.add_argument("--nepochs", help="number of epochs total", type=int, default=200)
parser.add_argument("--nepochscorr", help="number of epochs for MC correction", type=int, default=100)
parser.add_argument("--batchsize", help="batch size", type=int, default=10000)

args = parser.parse_args()


if args.gpu<0:
    import imp
    try:
        imp.find_module('setGPU')
        import setGPU
    except ImportError:
        found = False
else:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('running on GPU '+str(args.gpu))

if args.gpufraction>0 and args.gpufraction<1:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    print('using gpu memory fraction: '+str(args.gpufraction))


import numpy as np
from make_samples import make_sample
import pandas as pd
from keras.optimizers import Adam, SGD


def save(df, fname):
    dname = os.path.dirname(fname)
    if not os.path.isdir(dname):
        os.makedirs(dname)
    records = df.to_records(index=False)
    records.dtype.names = [str(i) for i in records.dtype.names]
    np.save(fname, records)
    


def schedule(x):
    lr=args.lr
    if x>100: lr=args.lr*0.8
    if x>200: lr=args.lr*0.5
    if x>400: lr=args.lr*0.1
    if x>600: lr=args.lr*0.05
    print('epoch', x, 'learning rate ',lr)
    return lr

learning_rate = keras.callbacks.LearningRateScheduler(schedule)


def run_model(outdir, training_method=1):
        
    nepochstotal=args.nepochs 
    nepochscorr=args.nepochscorr
    batchsize=args.batchsize

    
    Inputs = Input((10,))
        
    global_loss_list['GradientReversal']=GradientReversal()
    X_traintest, isB_traintest , isMC_traintest = make_sample(args.indir, args.addsv)
    X_all, X_test, isB_all, isB_test, isMC_all, isMC_test = train_test_split(X_traintest, isB_traintest , isMC_traintest, test_size=0.1, random_state=42)

    
    onesarray=np.ones(isB_all.shape[0])
    optimizer=Adam(lr=args.lr)
    from block_models import common_features, btag_discriminator,datamc_discriminator
    
    
    #for compatibility
    def modelIverseGrad(Inputs,rev_grad):
        #build the models
        feat = common_features(Inputs)
        feat_outshape = Input(feat.get_layer(index=-1).output_shape)
        btag = btag_discriminator(feat_outshape,dropoutrate=0.03)
        datamc = datamc_discriminator(feat_outshape,rev_grad=rev_grad,dropoutrate=0.03)
        
        return Model(inputs=Inputs, outputs=btag(feat(Inputs))+datamc(feat(Inputs)),name='fullModel')
    
    model = modelIverseGrad(Inputs,rev_grad=args.lmb)
    
    
    
    if(training_method == 'MC_training'):
        model.compile(
            loss=['binary_crossentropy']*4,
            optimizer=optimizer,
            loss_weights=[1., 0., 0., 0.],
            weighted_metrics=['accuracy']
            )
        history = model.fit(
            X_all,
            [isB_all, isB_all, isMC_all, isMC_all],
            batch_size=batchsize, epochs=nepochstotal, verbose=2, validation_split=0.2,
            sample_weight=[isMC_all.ravel(), 1-isMC_all.ravel(), 
                   1+0.5*isB_all.ravel(), 1-0.5*isB_all.ravel()],
            )
 
         
    elif(training_method == 'data_training'):
        model.compile(
            loss=['binary_crossentropy']*4,
            optimizer=optimizer, 
            loss_weights=[0.,1.,0.,0.],
            weighted_metrics=['accuracy']
        )
        history = model.fit(
            X_all,
            [isB_all, isB_all, isMC_all, isMC_all], 
            batch_size=batchsize, epochs=nepochstotal, verbose=2, validation_split=0.2,
            sample_weight = [
                isMC_all.ravel(),
                1-isMC_all.ravel(), 
                1+0.5*isB_all.ravel(), 
                1-0.5*isB_all.ravel()],
            
        )
    elif(training_method == 'stepwise_domain_adaptation' or training_method == 'corrected_domain_adaptation'):
        
        from block_models import mc_correction
        
        da_epochs = nepochstotal
        
        if training_method == 'corrected_domain_adaptation':
            da_epochs=nepochstotal-nepochscorr
        
        #build the models
        feat = common_features(Inputs)
        feat_outshape = Input(shape=(feat.get_layer(index=-1).output_shape[1:]))
        btag = btag_discriminator(feat_outshape,dropoutrate=0.03)
        datamc = datamc_discriminator(feat_outshape,rev_grad=args.lmb,dropoutrate=0.03)
        
            
        modelallopen = Model(inputs=Inputs, outputs=btag(feat(Inputs))+datamc(feat(Inputs)),name='fullModel')

        #print(feat.summary())
        #print(btag.summary())
        #print(datamc.summary())
    
        modelallopen.compile(
            loss = ['binary_crossentropy']*4,
            optimizer = optimizer, 
            loss_weights = [1.,0.,args.weight,0.],
            weighted_metrics=['accuracy']
        )
        
    
        feat.trainable = False
        btag.trainable = False
        
        modelfixedbtag = Model(inputs=Inputs, outputs=btag(feat(Inputs))+datamc(feat(Inputs)),name='fullModel2')
        
        modelfixedbtag.compile(
            loss = ['binary_crossentropy']*4,
            optimizer = optimizer, 
            loss_weights = [1.,0.,args.weight,0.],
            weighted_metrics=['accuracy']
        )
        
        from training_tools import train_models_adversarial
        history=train_models_adversarial([modelallopen,modelfixedbtag], 
                                [X_all], 
                                [isB_all, isB_all, isMC_all, isMC_all], 
                                da_epochs, 
                                sample_weight=[isMC_all.ravel(),
                                               1-isMC_all.ravel(), 
                                               onesarray,
                                               onesarray], 
                                batch_size=batchsize, 
                                validation_split=0.2, 
                                nstepspermodel=None, 
                                custombatchgenerators=None, 
                                verbose=1)

        print ('history type ', type(history))
        history=pd.DataFrame(history)
        
        model=modelallopen
        
        if(training_method == 'corrected_domain_adaptation'):
        
            #copy the trained model and add a correction for MC
            
            AllInputs=[Input((10, )),Input((1, )),Input((1, )),Input((1, ))]
            mccorr = mc_correction([feat_outshape,Input((1, )),Input((1, )),Input((1, ))], 
                                20, layers=2, dropoutrate=0.03)
            
            #fixLayers
            feat.trainable = False
            btag.trainable = False
            datamc.trainable = True
            mccorr.trainable = True
            
            datamc.get_layer('Ad_gradrev').hp_lambda=args.lmb2
            
            mcmodel = Model(inputs=AllInputs, 
                        outputs=
                        btag(mccorr([feat(AllInputs[0]),AllInputs[1],AllInputs[2],AllInputs[3]]))
                        +datamc(mccorr([feat(AllInputs[0]),AllInputs[1],AllInputs[2],AllInputs[3]])),
                        name='fullModelcorr')
            
            # the correction gets truth information, but 
            # the b-tag loss is weighted by 0, so this information
            # is only used for corrections
            
            mcmodel.compile(
                loss = ['binary_crossentropy']*4,
                optimizer = optimizer, 
                loss_weights = [0.,0.,args.weight,0.],
                weighted_metrics=['accuracy']
            )
            
            mccorr.trainable = False
            
            datamcdiscretrainmodel=Model(inputs=AllInputs, 
                        outputs=
                        btag(mccorr([feat(AllInputs[0]),AllInputs[1],AllInputs[2],AllInputs[3]]))
                        +datamc(mccorr([feat(AllInputs[0]),AllInputs[1],AllInputs[2],AllInputs[3]])),
                        name='fullModelcorrdmc')
            
            
            datamcdiscretrainmodel.compile(
                loss = ['binary_crossentropy']*4,
                optimizer = optimizer, 
                loss_weights = [0.,0.,args.weight,0.],
                weighted_metrics=['accuracy']
            )
            
            print('correcting MC')
            
            
            def generate_noise(nsamples):
                return np.random.normal(0, 1, nsamples)
            
            noise=generate_noise(1)#placeholder
            #fit the rest
            history2=train_models_adversarial([mcmodel,datamcdiscretrainmodel], 
                                [X_all,isMC_all,noise,isB_all], 
                                [isB_all, isB_all, isMC_all, isMC_all], 
                                nepochscorr, 
                                sample_weight=[isMC_all.ravel(),
                                               1-isMC_all.ravel(), 
                                               onesarray,
                                               onesarray], 
                                batch_size=batchsize, 
                                validation_split=0.2, 
                                nstepspermodel=None, 
                                custombatchgenerators=[None,None,generate_noise,None], 
                                verbose=1)
            history2=pd.DataFrame(history2)
            history=pd.concat([history,history2], ignore_index=True)
            model=mcmodel
            
            noise=generate_noise(X_test.shape[0])
            X_test=[X_test,isMC_test,noise,isB_all]
        
        
    else:
        raise ValueError('%s is an unknown run option' % training_method)

    if hasattr(history, 'history'):
        history = pd.DataFrame(history.history)
    save(history, '%s/history.npy' %outdir)
    if args.nopred:
        return history

    predictions = model.predict(X_test)
    preds = pd.DataFrame()
    preds['prediction'] = predictions[0].ravel()
    preds['isB'] = isB_test
    preds['isMC'] = isMC_test
    save(preds, '%s/predictions.npy' %outdir)
    
    
    return history

run_model(
    args.outputDir, 
    training_method=args.method
)