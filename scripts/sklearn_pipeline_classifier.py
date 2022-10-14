#########
# Modules
#########

import os
import sys
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from functions import cov_matrix_vectorized

#######
# Paths
#######

ROOT_PATH = <ROOT DIRECTORY>
DATA_DIR = <LOCATION OF DATDASETS>
RESULTS_DIR = <RESULTS DIRECTORY>

############
# Parameters
############

DATA_FILE = sys.argv[1]
DATA_PATH = os.path.join(ROOT_PATH,DATA_DIR,DATA_FILE,'ml.npz')
WITH_PCA = True if sys.argv[2]=='True' else False

EXPERIMENT_NAME = '{}-PCA-{}-{}-results'.format(DATA_FILE,WITH_PCA)

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

ml_data = np.load(DATA_PATH)
X_train, y_train, X_test, y_test = ml_data['X_train'], ml_data['y_train'],ml_data['X_test'],ml_data['y_test']

if WITH_PCA:
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)
else:
    # standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    # covariance
    X_train = np.array(list(map(cov_matrix_vectorized,X_train)))
    X_test = np.array(list(map(cov_matrix_vectorized,X_test)))
    
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

################
# Build pipeline
################

pipeline_arg_list = [
        ('clf', SVC()), 
]

if WITH_PCA:
    pipeline_arg_list.insert(0,('scaler', StandardScaler()),)
    pipeline_arg_list.insert(1,('pca', PCA()),)

pipeline = Pipeline(pipeline_arg_list)

parameters_list = [
    {
        'clf':(SVC(),),
        'clf__C':(0.1,1.0,10.0),
        'clf__kernel':('linear',),
    },
    {
        'clf':(RandomForestClassifier(),),
        'clf__n_estimators':(50,100,250),
    },
]

if WITH_PCA:
    for params in parameters_list:
        params['pca__n_components'] = (28,64,256,512,)

#############
# Grid Search
#############

for parameters in parameters_list:

    print('****************************************')
    print('Running {}'.format(parameters['clf'][0]))
    print('****************************************')

    grid_search = GridSearchCV(pipeline,
                               parameters,
                               n_jobs=-1,
                               verbose=1,
                               cv=10)

    print('Performing grid search...\n')
    print('Pipeline: {}\n'.format([name for name, _ in pipeline.steps]))
    print('Pipeline parameters:\n')
    for k,v in parameters.items():
        print('  {}:{}'.format(k,v))
    print('\r')
    t0 = time()

    # Fit the GridSearch on the training data
    grid_search.fit(X_train,y_train)
    print('Done in {:0.3f}s\n'.format(time()-t0))
          
    print('Best training score: {:0.4f}\n'.format(grid_search.best_score_))
    best_model_str = [ str(tup[1])[:-2] for tup in grid_search.best_estimator_.__dict__['steps'] if tup[0]=='clf' ][0]
    best_model_params = [params for params in parameters_list if best_model_str==str(params['clf'][0])[:-2]][0]
    best_params = grid_search.best_estimator_.get_params()
    print('Best parameters:\n')
    for param_name in sorted(best_model_params.keys()):
        print('  {}: {}'.format(param_name, best_params[param_name]))
    print('\nTEST SET ACCURACY USING BEST HYPERPARAMETERS {:0.4f}\n'.format(grid_search.score(X_test,y_test)))
