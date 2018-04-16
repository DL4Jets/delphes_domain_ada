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

def offsetLoss(y_true, y_pred):
	return K.mean(K.binary_crossentropy(y_true, y_pred),axis=-1)-0.672

def invertedLossXentrCut(y_true, y_pred):
	loss = 0.68 - K.binary_crossentropy(y_true, y_pred)
	loss=K.square(loss)
	return K.mean(loss) #clip at random and add to be in normal range

def makeEpochPlot(idstring,da_history,outdir):
	
	
	fig = plt.figure()

	plt.plot(da_history['val_data_'+idstring],label='data DA w 50 L .04', c='blue')

	plt.plot(da_history['val_mc_'+idstring],label='mc DA w 50 L .04', c='green',linestyle=':')
	
	plt.plot(da_history['val_Add_'+idstring]-0.5,label='ada', c='fuchsia',linestyle='--')
	
	plt.ylabel(''+idstring+'')
	plt.xlabel('epochs')
	plt.legend(ncol=2, loc='best')
	fig.savefig('%s/%s.png' % (outdir, idstring))
	fig.savefig('%s/%s.pdf' % (outdir, idstring))
	plt.clf()

parser = ArgumentParser()
parser.add_argument('outputDir')
parser.add_argument(
	'method', choices = [
		'domain_adaptation_two_samples',
		'MC_training',
		'data_training',
		'domain_adaptation_one_sample',
		'domain_adaptation_one_sample_lambdap5',
		'domain_adaptation_two_samples_w50_l.25',
		'domain_adaptation_two_samples_w50_l.04',
		'domain_adaptation_two_samples_w25_l.5',
		'domain_adaptation_two_samples_w05_l1',
		'stepwise_domain_adaptation',
		'pretrained_domain_adaptation'
	]
)
parser.add_argument("-i",  help="input directory", default='/data/ml/jkiesele/pheno_domAda/numpyx2/', dest='indir')
parser.add_argument("--addsv", action='store_true')
parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
parser.add_argument("--nopred", help="do not compute and store predictions", action='store_true')
parser.add_argument("--lr", help="learning rate", type=float, default=0.0005)
parser.add_argument("--weight", help="domain adaptation weight", type=float, default=2)
parser.add_argument("--lmb", help="domain adaptation lambda", type=float, default=10)
parser.add_argument("--sgdf", help="SGD factor", type=float, default=1)
parser.add_argument("--dainj", help="point to inject", type=int, default=2)
parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=0.17)

args = parser.parse_args()

loss_weigth = 150
lambda_reversal = .1
if args.method.startswith('domain_adaptation_two_samples_'):
	cfg = args.method[len('domain_adaptation_two_samples_'):]
	winfo, linfo = tuple(cfg.split('_'))
	loss_weigth = float(winfo[1:])
	lambda_reversal = float(linfo[1:])
	args.method = 'domain_adaptation_two_samples'
else:
	loss_weigth = args.weight
	lambda_reversal = args.lmb

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

from keras.engine import Layer
from Layers import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from make_samples import make_sample
#from DL4Jets.DeepJet.modules.Losses import  weighted_loss
from keras.layers import Dense, Concatenate ,Dropout
from keras.layers import Input
from keras.models import Model
import pandas as pd
from keras.optimizers import Adam, SGD


def save(df, fname):
	dname = os.path.dirname(fname)
	if not os.path.isdir(dname):
		os.makedirs(dname)
	records = df.to_records(index=False)
	records.dtype.names = [str(i) for i in records.dtype.names]
	np.save(fname, records)
	
def injectDA(layer,X,Xa, doit):
	if doit:
		Xa = layer(X)
		X = Xa
		print('injected DA in layer', layer.name)
	else:
		X = layer(X)
	return X, Xa


def schedule(x):
	lr=args.lr
	if x>100: lr=args.lr*0.8
	if x>200: lr=args.lr*0.5
	if x>400: lr=args.lr*0.1
	if x>600: lr=args.lr*0.05
	print('epoch', x, 'learning rate ',lr)
	return lr

learning_rate = keras.callbacks.LearningRateScheduler(schedule)

def modelIverseGrad(Inputs, rev_grad=.1, dropoutrate=0.03):
	X=Inputs
	Xa=0
	X,Xa = injectDA(Dense(30, activation='relu', name='common_dense_0'),
				X,Xa,args.dainj==0)	
	X = Dropout(dropoutrate)(X)

	X = Dense(20, activation='relu', name='common_dense_1')(X)

	X,Xa = injectDA(Dense(20, activation='relu', name='common_dense_2'),
				X,Xa,args.dainj==1)
	X = Dropout(dropoutrate)(X)
	
	X = Dense(15, activation='relu', name='common_dense_3')(X)
	X = Dropout(dropoutrate)(X)

	X,Xa= injectDA(Dense(15, activation='relu', name='common_dense_4'),
				X,Xa,args.dainj==2)
	X = Dropout(dropoutrate)(X)
	
	X,Xa = injectDA(Dense(15, activation='relu', name='btag_classifier_0'),
				X,Xa,args.dainj==3)
	X = Dropout(dropoutrate)(X)
	
	X,Xa = injectDA(Dense(1, activation='sigmoid', name = 'mc'),
				X,Xa,args.dainj==4)
	
	X1= Dense(1, activation='linear',use_bias=False, trainable=False,
			kernel_initializer='Ones', name = 'data') (X)
	
	Ad = GradientReversal(hp_lambda=rev_grad, name='Ad_gradrev')(Xa)
	Ad = Dense(20, activation='relu', name='Ad_0')(Ad)
	Ad = Dropout(dropoutrate)(Ad)
	Ad = Dense(15, activation='relu', name='Ad_1')(Ad)
	Ad = Dropout(dropoutrate)(Ad)
	Ad = Dense(15, activation='relu', name='Ad_2')(Ad)
	Ad = Dropout(dropoutrate)(Ad)
	Ad = Dense(10, activation='relu', name='Ad_3')(Ad)
	Ad = Dropout(dropoutrate)(Ad)
	Ad = Dense(1, activation='sigmoid', name = 'Add' )(Ad)
	
	Ad1 = Xa #GradientReversal(hp_lambda=-1, name='Ad1_gradrev')(Xa)
	Ad1 = Dense(1, activation='sigmoid', name = 'Add_1' ,trainable=False)(Ad1)
	predictions = [X,X1,Ad,Ad1]
	model = Model(inputs=Inputs, outputs=predictions)
	return model


nepochstotal=200
batchsize=10000


def run_model(outdir, Grad=1, known = 1,AdversOn=1,diffOn = 1):
	Inputs = Input((10,))
	global_loss_list={}
	global_loss_list['GradientReversal']=GradientReversal()
	X_traintest, isB_traintest , isMC_traintest = make_sample(args.indir, args.addsv)
	X_all, X_test, isB_all, isB_test, isMC_all, isMC_test = train_test_split(X_traintest, isB_traintest , isMC_traintest, test_size=0.1, random_state=42)
	advers_weight = 25.
	if AdversOn==0:
		advers_weight = 0.
		
	onesarray=np.ones(isB_all.shape[0])
	optimizer=Adam(lr=args.lr)
	#optimizer.
	
	model = modelIverseGrad(Inputs)
	
	print(model.summary())
	
	#configure optimiser
	
	
	# gradiant loss
	if(Grad == 'domain_adaptation_two_samples'):
		exit()
		model = modelIverseGrad(Inputs,rev_grad=lambda_reversal)
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer=optimizer, 
			weighted_metrics=['accuracy'],
			loss_weights=[1., 0., loss_weigth, loss_weigth]
		)
		history = model.fit(
			X_all, 
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=batchsize, epochs=nepochstotal,  verbose=0, validation_split=0.2, 
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				1-isB_all.ravel()*0.75, 
				1+isB_all.ravel()*0.75],
		)
	elif(Grad == 'MC_training'):
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer=optimizer, 
			loss_weights=[1.,0.,0.,0.],
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
	elif(Grad == 'data_training'):
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
	elif(Grad == 'domain_adaptation_one_sample'):
		model = modelIverseGrad(Inputs,rev_grad=args.lmb)
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer=optimizer, 
			weighted_metrics=['accuracy'],
			loss_weights=[1.,0.,args.weight,0.]
		)
		history = model.fit(
			X_all, 
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=batchsize, epochs=nepochstotal, verbose=2, validation_split=0.2, 
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				onesarray,
				onesarray],		
					
		)
	elif(Grad == 'stepwise_domain_adaptation' or Grad == 'pretrained_domain_adaptation'):
		model = modelIverseGrad(Inputs,rev_grad=args.lmb)
		from modeltools import fixLayersContaining, setAllTrainable
		
		
		
		print('train b-tag disciminator')
		#model = fixLayersContaining(model,['Ad_'])
		
		class_epochs   =1
		domdiscr_epochs=1
		da_epochs      =nepochstotal-class_epochs-domdiscr_epochs
		
	
		
		
		
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer = optimizer, 
			loss_weights = [1.,0.,0.,0.],
			weighted_metrics=['accuracy']
		)
		history = model.fit(
			X_all, 
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=batchsize, epochs=class_epochs,  verbose=0, validation_split=0.2,
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				onesarray,
				onesarray],		
					
		)
		history = pd.DataFrame(history.history)
		
		
		
		print('train domain adaptation discriminator part')
		model = setAllTrainable(model) #open all
		model = fixLayersContaining(model,['common_','btag_classifier_'])
		

		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer = optimizer, 
			loss_weights = [0.,0.,1.,0.],
			weighted_metrics=['accuracy']
		)
		history2=pd.DataFrame(model.fit(
			X_all, 
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=batchsize, epochs=domdiscr_epochs+class_epochs,  verbose=0, validation_split=0.2,
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				onesarray,
				onesarray],	
			initial_epoch=class_epochs,
						
		).history)
		
		history=pd.concat([history,history2], ignore_index=True)
		
		print('train domain adaptation full')
		
		def setLambda(m,lambda_grad):
			for layidx in range(len(m.layers)):
				name = m.get_layer(index=layidx).name
				if 'gradrev' in name:
					m.get_layer(index=layidx).hp_lambda=lambda_grad
		
		
		setLambda(model,args.lmb)
		switch=0
		stepda  =2
		stepbtag=2
		##
		currentepoch=0
		step=0
		#make the optimised the same
		weight=args.weight
		lr=optimizer.lr
		while Grad == 'stepwise_domain_adaptation':
			
			
			if switch % 2 == 0:
				print('re-adjust btag discriminator')
				#small DA loss weight, high lambda
				
				model = setAllTrainable(model)#opens all layers
				step=stepbtag
				weight=args.weight
				
				#optimizer.lr=lr/(weight*args.lmb)
				model.compile(
					loss = ['binary_crossentropy']*4,#+[offsetLoss]+['binary_crossentropy'],
					optimizer = optimizer, 
					loss_weights = [1.,0.,weight,0.],
					weighted_metrics=['accuracy']
				)
				
				
				
			else:
				if currentepoch>da_epochs:
					break
				print('re-adjust data/mc discriminator')
				#small DA loss weight, high lambda
				
				model = setAllTrainable(model)
				model = fixLayersContaining(model,['common_','btag_classifier_'])
				step=stepda
				#optimizer.lr=lr	
				weight=1
				model.compile(
					loss = ['binary_crossentropy']*4,#+[offsetLoss]+['binary_crossentropy'],
					optimizer = optimizer, 
					loss_weights = [1.,0.,weight,0.],
					weighted_metrics=['accuracy']
				)
				
			switch=switch+1
			
			
			print('train from ',currentepoch, ' to ', currentepoch+step)
			
			
			
			history2=pd.DataFrame(
				model.fit(
				X_all, 
				[isB_all, isB_all, isMC_all, isMC_all], 
				batch_size=batchsize, epochs=domdiscr_epochs+class_epochs+currentepoch+step,  
				verbose=2, validation_split=0.2,
				sample_weight = [
					isMC_all.ravel(),
					1-isMC_all.ravel(), 
					onesarray,
					onesarray],
				initial_epoch=class_epochs+domdiscr_epochs+currentepoch,
										
			).history)
			
			
			currentepoch=currentepoch+step
		
			history=pd.concat([history,history2], ignore_index=True)
			
			
		if Grad == 'pretrained_domain_adaptation':
			model = setAllTrainable(model)
			model.compile(
					loss = ['binary_crossentropy']*4,#+[offsetLoss]+['binary_crossentropy'],
					optimizer = optimizer, 
					loss_weights = [1.,0.,args.weight,0.],
					weighted_metrics=['accuracy']
				)
			history2=pd.DataFrame(
				model.fit(
				X_all, 
				[isB_all, isB_all, isMC_all, isMC_all], 
				batch_size=batchsize, epochs=domdiscr_epochs+class_epochs+da_epochs,  
				verbose=2, validation_split=0.2,
				sample_weight = [
					isMC_all.ravel(),
					1-isMC_all.ravel(), 
					onesarray,
					onesarray],
				initial_epoch=class_epochs+domdiscr_epochs,
										
			).history)
			
			history=pd.concat([history,history2], ignore_index=True)
		
		
		
	else:
		raise ValueError('%s is an unknown run option' % Grad)

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
	
	
	makeEpochPlot('weighted_acc',history,outdir)
	makeEpochPlot('loss',history,outdir)
	
	return history

#print history.history.keys()
run_model(
	args.outputDir, 
	Grad=args.method, 
	known = 1, 
	AdversOn=1,
	diffOn = 1
)
###   #print ('damain adaptation with tw sources')
###   history2 = run_model(Grad=2, known = 1,AdversOn=1,diffOn = 1)
###   #print ('train on sources')
###   history3 = run_model(Grad=3, known = 1,AdversOn=1,diffOn = 1)
###   #history4 = run_model(Grad=4, known = 1,AdversOn=1,diffOn = 1)
###   #history5 = run_model(Grad=5, known = 1,AdversOn=1,diffOn = 1)
###   
###   
###   fig = plt.figure()
###   plt.plot(history1.history['val_data_loss'],label='data DA 0.25')
###   plt.plot(history1.history['val_mc_loss'],label='mc DA 0.25')
###   plt.plot(history2.history['val_data_loss'],label='data mc')
###   plt.plot(history2.history['val_mc_loss'],label='mc mc')
###   plt.plot(history3.history['val_data_loss'],label='data data')
###   plt.plot(history3.history['val_mc_loss'],label='mc data')
###   #plt.plot(history4.history['val_data_loss'],label='data DA 0.1')
###   #plt.plot(history4.history['val_mc_loss'],label='mc DA 0.1')
###   #plt.plot(history5.history['val_data_loss'],label='data DA 0.5')
###   #plt.plot(history5.history['val_mc_loss'],label='mc DA 0.5')
###   
###   
###   plt.ylabel('loss')
###   plt.xlabel('epochs')
###   plt.legend()
###   fig.savefig('myPlot')
###   #plt.figure(2)
###   
###   #plt.plot(history.history['val_dense_8_loss'],label='data')
###   #	plt.plot(history.history['val_dense_7_loss'],label='mc')
###   #	plt.legend()
###   #plt.plot(history.history['val_dense_12_loss'])
###   #plt.figure(3)
###   #plt.plot(history.history['val_loss'],label='full loss')
###   #plt.plot(history.history['val_dense_12_loss'])
###   #plt.legend()
###   #plt.show()

