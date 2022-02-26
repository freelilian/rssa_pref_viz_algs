# train explicit MF model
import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
# There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
        
import numpy as np
import pandas as pd
from lenskit.algorithms import als
#from . import setpath
import setpath
import time
import pickle
from load_npz import load_trainset_npz
#from .load_npz import load_trainset_npz
import os

# import data
data_path = setpath.set_working_path()
# fullpath_train = data_path + 'train.npz'
fullpath_train = os.path.join(data_path, 'train.npz')
attri_name = ['user', 'item', 'rating', 'timestamp']
ratings_train = load_trainset_npz(fullpath_train, attri_name)

# train MF
model_filename = os.path.join(data_path, 'explicitMF.pkl')
f = open(model_filename, 'wb')
print('Start training the MF model ...')
start = time.time()
algo = als.BiasedMF(20)
algo.fit(ratings_train)
end = time.time() - start
print("\nThe MF model is trained. Time spent: %0.0fs\n" % end)
print('\nExporting the trained MF model - an object - as a pkl file')
pickle.dump(algo, f)
f.close() 
print('\nDone\n')