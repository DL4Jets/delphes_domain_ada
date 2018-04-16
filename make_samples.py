import numpy as np
import sys, os
from glob import glob

def make_sample(input_dir, add_svs):
	x_file = glob('%s/*features_0.npy' % input_dir)[0]
	y_file = glob('%s/*truth_1.npy' % input_dir)[0]
	X_all = np.load(x_file)
	Y_all = np.load(y_file)
	
	print X_all.shape, Y_all.shape
	
	# drop Cs or anything else
	Y_C = Y_all[:,2:3]<0.1
	print Y_C.shape
	
	Y_all = Y_all[Y_C.ravel()]
	X_all = X_all[Y_C.ravel()]
	
	# get the right labels
	isB_all = Y_all[:,1:2]
	isMC_all = Y_all[:,0:1]
	
	print(isMC_all.shape)
	isMC_all =isMC_all.ravel()
	print(isMC_all.shape)
	#select MC only
	
	#create random isMC vector
	#np.random.seed(1234)
	#isMC_all = np.random.randint(2, size=X_all.shape[0])
	#isMC_all = np.reshape(isMC_all,(isMC_all.shape[0],1))
	
	#print(isMC_all.shape)

	X_ptRel= X_all[:,:3]#5
	X_2Ds= X_all[:,5:8]#10
	X_3Ds= X_all[:,10:11]#15
	X_ptPro= X_all[:,15:18]#20
	# now we can increase the smearing
	noise = np.random.randn(X_all.shape[0],5)*0.5
	noise2 = np.random.randn(X_all.shape[0],5)*0.5
	noise_uni = np.random.rand(X_all.shape[0],1) > 0.666666
	
	def addStretch(Xin, stretch, data):
		if data:
			return Xin + Xin* stretch * (isMC_all<.1) * (isB_all>.1)
		else:
			return Xin + Xin* stretch * (isMC_all>.1) * (isB_all>.1)
	
	#X_3Ds=addStretch(X_3Ds, 0.05,False)

	
	poisson_b = (np.random.rand(X_all.shape[0],1) > 0.15)*isB_all
	poisson_qcd = (np.random.rand(X_all.shape[0],1) > 0.6)*(1-isB_all)
	SV = poisson_qcd+poisson_b if add_svs else np.random.rand(X_all.shape[0],1)
	
	#X_2Ds_0 = X_2Ds_0 + noise*(isMC_all<.1)
	#X_3Ds_0 = X_3Ds_0 + noise2*(isMC_all<.1)
	#X_3Ds =  X_3Ds + noise * X_3Ds # * X_3Ds * (isMC_all<.1)
	#X_2Ds =  X_2Ds + noise * X_2Ds #* X_2Ds * (isMC_all<.1)
	#X_ptRel= noise #* X_3Ds * (isMC_all<.1)
	#X_ptPro= noise #* X_3Ds * (isMC_all<.1)
	return np.concatenate([X_ptRel,X_2Ds,X_3Ds,X_ptPro], axis=1), isB_all , isMC_all,
#return make_sample_old()


def make_sample_old():
	X_all = np.load('large_dataset_features_0.npy')
	Y_all = np.load('large_dataset_truth_1.npy')

	print X_all.shape, Y_all.shape

	Y_C = Y_all[:,2:3]<0.1
	print Y_C.shape

	Y_all = Y_all[Y_C.ravel()]
	X_all = X_all[Y_C.ravel()]

	print X_all.shape, Y_all.shape

	isB_all = Y_all[:,1:2]
	isMC_all = Y_all[:,0:1]
	
	

#np.save('X_all.npy',X_all)
#	np.save('isB_all.npy',isB_all)
#	np.save('isMC_all.npy',isMC_all)

#	np.save('X_mc.npy',X_all[(isMC_all>.1).ravel()])
#	np.save('isB_mc.npy',isB_all[(isMC_all>.1).ravel()])
#	np.save('isMC_mc.npy',isMC_all[(isMC_all>.1).ravel()])

#	np.save('X_data.npy',X_all[(isMC_all<.1).ravel()])
#	np.save('isB_data.npy',isB_all[(isMC_all<.1).ravel()])
#	np.save('isMC_data.npy',isMC_all[(isMC_all<.1).ravel()])

	return X_all, isB_all , isMC_all,


