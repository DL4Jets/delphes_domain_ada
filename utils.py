'''collection of useful functions'''
import os
import numpy as np
import pandas as pd
from fnmatch import fnmatch

def save(df, fname):
    'saves a pandas DataFrame to a numpy file'
    dname = os.path.dirname(fname)
    if not os.path.isdir(dname):
        os.makedirs(dname)
    records = df.to_records(index=False)
    records.dtype.names = [str(i) for i in records.dtype.names]
    np.save(fname, records)

def fix_layers(my_model):
    'sets all the layers in a model as non trainable'
    for layer in my_model.layers:
        layer.trainable = False
    return my_model
		
def open_layers(my_model):
    'sets all the layers in a model as trainable'
    for layer in my_model.layers:
        layer.trainable=True
    return my_model

def set_trainable(m, patterns, value=True):
    '''sets single layers are (non)-trainable, depending on the name.
POSIX regex allowed as single string or list. Defaults to trainable'''
    if isinstance(patterns, basestring):
        patterns = [patterns]
    for layer in m.layers:
        name = layer.name
        if any(fnmatch(name, i) for i in patterns):
            layer.trainable = value
    return m
