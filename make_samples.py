import numpy as np
from glob import glob


def make_sample(input_dir, add_svs):

    x_file = glob('%s/*features_0.npy' % input_dir)[0]
    y_file = glob('%s/*truth_1.npy' % input_dir)[0]
    X_all = np.load(x_file)
    Y_all = np.load(y_file)

    # drop Cs or anything else
    Y_C = Y_all[:, 2:3] < 0.1
    Y_all = Y_all[Y_C.ravel()]
    X_all = X_all[Y_C.ravel()]
    # get the right labels
    isB_all = Y_all[:, 1:2]
    isMC_all = Y_all[:, 0:1]
    print(isMC_all.shape)
    isMC_all = isMC_all.ravel()
    print(isMC_all.shape)
    # select MC only
    
    if False:
        np.random.seed(1234)
        isMC_all = np.random.randint(2, size=X_all.shape[0])
        isMC_all = np.reshape(isMC_all,(isMC_all.shape[0],1))

    X_ptRel = X_all[:, :1]#3
    X_2Ds = X_all[:, 5:6]#8
    X_3Ds = X_all[:, 10:11]
    X_ptPro = X_all[:, 15:16]#18
    # now we can increase the smearing
    # noise = np.random.randn(X_all.shape[0],5)*0.5
    # noise2 = np.random.randn(X_all.shape[0],5)*0.5
    # noise_uni = np.random.rand(X_all.shape[0],1) > 0.666666
    
    def addMCStretch(Xin, stretch, data=False):
        selected = np.array(isMC_all.ravel(), dtype='float32')#important to copy here
        if data:
            selected-=1
            selected=np.abs(selected)
        selected *= isB_all.ravel()
        selected *= stretch
        selected += 1
        selected=np.reshape(selected, (selected.shape[0],1))
        Xin = np.multiply(Xin,selected)
        return Xin

    # X_2Ds=addMCStretch(X_2Ds, 5.5)
    # X_3Ds=addMCStretch(X_3Ds, 5.5)
    
    
    # poisson_b = (np.random.rand(X_all.shape[0], 1) > 0.15) * isB_all
    # poisson_qcd = (np.random.rand(X_all.shape[0], 1) > 0.6) * (1 - isB_all)
    # SV = poisson_qcd + poisson_b if add_
    # svs else np.random.rand(X_all.shape[0], 1)
    # X_2Ds_0 = X_2Ds_0 + noise*(isMC_all<.1)
    # X_3Ds_0 = X_3Ds_0 + noise2*(isMC_all<.1)
    # X_3Ds =  X_3Ds + noise * X_3Ds # * X_3Ds * (isMC_all<.1)
    # X_2Ds =  X_2Ds + noise * X_2Ds #* X_2Ds * (isMC_all<.1)
    # X_ptRel= noise #* X_3Ds * (isMC_all<.1)
    # X_ptPro= noise #* X_3Ds * (isMC_all<.1)
    return np.concatenate(
        [X_ptRel, X_2Ds, X_3Ds, X_ptPro], axis=1), isB_all, isMC_all
