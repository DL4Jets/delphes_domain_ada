
from __future__ import print_function
import numpy as np
from sklearn.utils import shuffle
from random import randint


def train_models_adversarial(modellist,
                             x,
                             y,
                             epochs,
                             sample_weight,
                             batch_size,
                             validation_split,
                             nstepspermodel=None,
                             custombatchgenerators=None,
                             modify_labels=None,
                             verbose=0):
    '''
    modellist,x,y,sample_weight need to be lists, even if only one entry
   
    nstepspermodel must be a list with same length of the model list, defining the number of
    batches per model, before the training resumes at the first model again. 
    If none is given, each model gets trained one batch
    
    custombatchgenerators (opt) is a list of funtions (not generators) that generate a custom input for x
    the function replaces the corresponding item of the list 'x' per batch, e.g.
    x = [arr1, arr2]
    custombatchgenerators = [None, myfunct]
    then arr2 will be replaced with the output of myfunct(batchsize) for each batch.
    The function given to the list must have the batch size as first argument.
    
    modify_labels: applies user defined function to the label list. Needs to be a list
    with same length as the model list. This function will be called for each batch, so keep it simple
    
    '''

    # create the batches
    totalsize = x[0].shape[0]
    ntrainbatches = int((1-validation_split)*totalsize/batch_size)

    split = []
    # make split - last split will be val sample
    for i in range(ntrainbatches):
        split.append((i+1)*batch_size)

    # this function makes sure the training sample
    # comprises exactly ntrainbatches batches
    def split_list(in_list):
        if not hasattr(in_list, '__getitem__'):
            return None, None
        out = [None for i in range(len(in_list))]
        val = [None for i in range(len(in_list))]
        for i in range(len(in_list)):
            sp = np.split(in_list[i], split)
            for j in range(len(sp)-1):
                if j:
                    out[i] = np.concatenate((out[i],sp[j]))
                else:
                    out[i] = sp[j]
            val[i] = sp[-1]
        return out, val
    
    def split_to_batches(in_list):
        if not hasattr(in_list, '__getitem__'):
            return [None for i in range(ntrainbatches)]
        out = [[None for i in range(len(in_list))] for i in range(ntrainbatches)]
        for i in range(len(in_list)):
            sp = np.split(in_list[i], split)
            for j in range(len(sp)-1):
                out[j][i] = sp[j]
        return out

    x_train, x_val = split_list(x)
    y_train, y_val = split_list(y)
    sw_train, sw_splitval = split_list(sample_weight)

    # input is prepared
    # re-write to shuffle the whole list and split again
    def randomise_array_lists():
        rstate = randint(0, 1e4)
        for i in range(len(x_train)):
            x_train[i] = shuffle(x_train[i], random_state=rstate)
        for i in range(len(y_train)):
            y_train[i] = shuffle(y_train[i], random_state=rstate)
        if sw_train:
            for i in range(len(sw_train)):
                sw_train[i] = shuffle(sw_train[i], random_state=rstate)
            

    # do this global for epochs
    modelit = 0
    trainhist = []
    valhist = []
    model = modellist[0]

    usedmetrics = []
    metrics_idx = dict((i, 1) for i in set(model.metrics_names))
    for name in model.metrics_names:
        if model.metrics_names.count(name) > 1:
            usedmetrics.append('%s_%d' % (name, metrics_idx[name]))
            metrics_idx[name] = metrics_idx[name] + 1
        else:
            usedmetrics.append(name)

    for e in range(epochs):
        
        randomise_array_lists()
        x_split = split_to_batches(x_train)
        y_split = split_to_batches(y_train)
        sw_split = split_to_batches(sw_train)
        
        epochtrainhist = []
        currentbatch = 0
        print('starting adversarial epoch ' + str(e+1) + '/' + str(epochs))
        while True:
            nsteps = 1
            if nstepspermodel:
                nsteps = nstepspermodel[modelit]

            model = modellist[modelit]

            while nsteps:
                # prepare extra function
                if custombatchgenerators:
                    for i in range(len(custombatchgenerators)):
                        if custombatchgenerators[i]:
                            x_split[currentbatch][i] = custombatchgenerators[i](batch_size)
                            if verbose > 2:
                                print('launched generator function '+str(i)+' for batch '+str(currentbatch))

                nsteps = nsteps-1
                if modify_labels:
                    if modify_labels[modelit]:
                        y_split[currentbatch]=modify_labels[modelit](y_split[currentbatch])
                            
                

                if verbose > 2:
                    print('training model ', modelit, ' on batch ', currentbatch)
                # take last batch for training history
                epochtrainhist = model.train_on_batch(x_split[currentbatch],
                                                      y_split[currentbatch],
                                                      sample_weight = sw_split[currentbatch])
                if verbose > 2:
                    print(epochtrainhist)

                # adjust for next batch
                if nsteps:
                    currentbatch = currentbatch+1
                if currentbatch == ntrainbatches:
                    break

            modelit = modelit+1
            if modelit == len(modellist):
                modelit = 0
            currentbatch = currentbatch+1
            if currentbatch >= ntrainbatches:
                break

        trainhist.append(epochtrainhist)
        print('end of epoch '+str(e+1) +' : validation')

        # prepare extra function
        if custombatchgenerators:
            for i in range(len(custombatchgenerators)):
                if custombatchgenerators[i]:
                    x_val[i] = custombatchgenerators[i](x_val[0].shape[0])

        valhist.append(model.test_on_batch(x_val, y_val, 
                                           sample_weight = sw_splitval))

        if verbose > 0:
            for i in range(len(usedmetrics)):
                print(usedmetrics[i]+": "+str(trainhist[-1][i]), end=' - ')
            for i in range(len(usedmetrics)):
                print("val_"+usedmetrics[i]+": "+str(valhist[-1][i]), end=' - ')
            print('\n')

    history = {}

    # use usedmetrics to make dict out of it
    for i in range(len(usedmetrics)):
        # make list
        eplisttrain = []
        eplistval = []
        for j in range(len(trainhist)):
            eplisttrain.append(trainhist[j][i])
            eplistval.append(valhist[j][i])
        history[usedmetrics[i]] = eplisttrain
        history["val_"+usedmetrics[i]] = eplistval

    return history
