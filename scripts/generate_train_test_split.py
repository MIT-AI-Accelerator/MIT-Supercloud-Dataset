#########
# Modules
#########

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

#######
# Paths
#######

ROOT_PATH = <ROOT DIRECTORY>
DATA_DIR = <LOCATION OF DATASETS>
RESULTS_DIR = <RESULTS DIRECTORY>

############
# Parameters
############

DATA_FILE = sys.argv[1]
DATA_FULL_PATH = os.path.join(ROOT_PATH,DATA_DIR,DATA_FILE,'ml.npz')

EXPERIMENT_NAME = '{}-results'.format(DATA_FILE)

#############
# Directories
#############

RESULTS_PATH = os.path.join(ROOT_PATH,RESULTS_DIR)
try:
    os.makedirs(RESULTS_PATH)
except:
    pass

sys.stdout = open(os.path.join(RESULTS_PATH,EXPERIMENT_NAME+'.txt'), 'w')

##############
# Load ML Data
##############

ml_data = np.load(DATA_FULL_PATH)
X,y,model = ml_data['X'],ml_data['y'],ml_data['model']

##################
# Train-Test Split
##################

sss = StratifiedShuffleSplit(n_splits=1, 
                             test_size=0.2,
                             random_state=37)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model_train, model_test = model[train_index], model[test_index]
print(X_train.shape,y_train.shape,model_train.shape)
print(X_test.shape,y_test.shape,model_test.shape)

#############
# Save Splits
#############

NPZ_OUTPUT_PATH = os.path.join(RESULTS_PATH,DATA_FILE)
try:
    os.makedirs(NPZ_OUTPUT_PATH)
except:
    pass

np.savez(os.path.join(NPZ_OUTPUT_PATH,'ml.npz'),
         X_train=X_train,
         y_train=y_train,
         model_train=model_train,
         X_test=X_test,
         y_test=y_test,
         model_test=model_test)
