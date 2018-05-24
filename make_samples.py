import numpy as np
np.random.seed(1234567)
from glob import glob
from pdb import set_trace

def make_sample(input_dir, add_svs, keep_probability=None): #default value tuned for current sample
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
        isMC_all = np.random.randint(2, size=X_all.shape[0])
        isMC_all = np.reshape(isMC_all,(isMC_all.shape[0],1))

    X_ptRel = X_all[:, :3]
    X_2Ds = X_all[:, 5:8]

    X_3Ds = X_all[:, 10:11]
    X_ptPro = X_all[:, 15:18]
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
    
    X_all = np.concatenate([X_ptRel, X_2Ds, X_3Ds, X_ptPro], axis=1)
    #
    # flavour weights stuff
    #
    ismc = (isMC_all.ravel() > 0.5)
    isb = (isB_all.ravel() > 0.5)
    #compute scaling to get same composition
    cnst = float((isb & np.invert(ismc)).sum())/np.invert(ismc).sum()
    nb = (isb & ismc).sum()
    nl = (np.invert(isb) & ismc).sum()
    fraction = (nb/cnst - nb)/nl
    if fraction > 1.:
        fraction = -1*nl/(nb/cnst - nb)
    print ' -- DATASET --'
    print 'Before masking:'
    print 'MC fraction: %.3f' % (float(ismc.sum())/ismc.shape[0])
    print 'Data B fraction: %.3f' % (float((isb & np.invert(ismc)).sum())/np.invert(ismc).sum())
    print 'MC B fraction: %.3f' % (float((isb & ismc).sum())/ismc.sum())
    print '\n'
    if keep_probability is None:
        print 'Forcing fractions to be the same'
        keep_probability = fraction

    if keep_probability > 0:
        mask = np.invert(ismc) | \
               (ismc & isb) | \
               (ismc & np.invert(isb) & (np.random.rand(isMC_all.shape[0]) < keep_probability))
    else:
        mask = np.invert(ismc) | \
               (ismc & np.invert(isb)) | \
               (ismc & isb & (np.random.rand(isMC_all.shape[0]) < -1*keep_probability))
    
    X_all    = X_all[mask]		
    isB_all  = isB_all[mask]
    isMC_all = isMC_all[mask]
    ismc = (isMC_all.ravel() > 0.5)
    #make data and mc 50%
    mc_frac = (float(ismc.sum())/ismc.shape[0])
    if mc_frac > 0.5:
        dropout = float(np.invert(ismc).sum())/float(ismc.sum())
        mask = np.invert(ismc) | (ismc & (np.random.rand(isMC_all.shape[0]) < dropout))
    else:
        dropout = float(ismc.sum())/float(np.invert(ismc).sum())
        mask = ismc | (np.invert(ismc) & (np.random.rand(isMC_all.shape[0]) < dropout))

    X_all    = X_all[mask]		
    isB_all  = isB_all[mask]
    isMC_all = isMC_all[mask]
    ismc = (isMC_all.ravel() > 0.5)
    isb  = (isB_all.ravel() > 0.5)
    print 'After masking:'
    print 'MC fraction: %.3f' % (float(ismc.sum())/ismc.shape[0])
    print 'Data B fraction: %.3f' % (((isb & np.invert(ismc)).sum())/float(np.invert(ismc).sum()))
    print 'MC B fraction: %.3f' % (((isb & ismc).sum())/float(ismc.sum()))
    print '\n'
    input_weights = ((isb == 1) & ismc).astype(float)
    input_weights = input_weights.reshape((input_weights.shape[0],1))

    nb_mc = (isb & ismc).sum()
    nl_mc = (ismc & np.invert(isb)).sum()
    data_b_frac = (np.invert(ismc) & isb).sum() / float((np.invert(ismc)).sum())
    print 'MC: #B: %d #L: %d' % (nb_mc, nl_mc)
    exp_f = data_b_frac*nl_mc/(nb_mc*(1-data_b_frac))
    exp_f -= 1
    print 'Expected weight: %.3f' % exp_f
    
    return X_all, isB_all, isMC_all, input_weights, exp_f
