#########
# Modules
#########

import os
import sys
import numpy as np
from time import time
import xgboost as xgb
xgb.set_config(verbosity=0)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
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

EXPERIMENT_NAME = 'xgb-{}-{}-results'.format(DATA_FILE)

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

# standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# covariance
X_train = np.array(list(map(cov_matrix_vectorized,X_train)))
X_test = np.array(list(map(cov_matrix_vectorized,X_test)))
    
################
# Build pipeline
################

xgb_clf = xgb.XGBClassifier(learning_rate=0.1, 
                            objective='multi:softmax',
                            silent=True, 
                            num_boost_round=3, 
                            early_stopping_rounds=10,
                            max_depth=3,
                            nthread=1,
                            num_class=np.unique(y_train).shape[0],
                            label_encoder=False)

xgb_params = {
    'gamma':[0.05,0.5,1],
    'alpha':[0.05,0.5,1],
    'lambda':[0.05,0.5,1]    
}

#############
# Grid Search
#############

print('*******************')
print('Running Grid Search')
print('*******************')

grid_search = GridSearchCV(estimator=xgb_clf, 
                           param_grid=xgb_params,
                           n_jobs=-1,
                           verbose=1,
                           scoring='accuracy',
                           cv=10)
      
# Fit the GridSearch on the training data
t0 = time()
grid_search.fit(X_train,y_train)
print('Done in {:0.3f}s\n'.format(time()-t0))

print('Best training score: {:0.4f}\n'.format(grid_search.best_score_))
best_params = grid_search.best_estimator_.get_params()
print('Best parameters:\n')
for param_name in sorted(xgb_params.keys()):
    print('  {}: {}'.format(param_name, best_params[param_name]))
print('\nTest set accuracy using best hyperparameters {:0.4f}\n'.format(grid_search.score(X_test,y_test)))

####################
# Feature Importance
####################

best_estimator = grid_search.best_estimator_
feature_importance = best_estimator.feature_importances_

feature_importance_matrix = np.zeros((7,7))
feature_importance_matrix[np.triu_indices(7)] = feature_importance

features = [
    '',
    'utilization gpu pct',
    'utilization memory pct',
    'memory free MiB',
    'memory used MiB',
    'temperature gpu',
    'temperature memory',
    'power draw W'
]
plt.set_cmap(cmap='inferno')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticklabels(features,rotation=90)
ax.set_yticklabels(features,rotation=None)
cax = ax.matshow(feature_importance_matrix, 
                 interpolation='nearest')
fig.colorbar(cax)
plt.savefig(os.path.join(RESULTS_PATH,'xgb-feature-importance-{}.jpg'.format(DATA_FILE)),
            bbox_inches='tight',
            dpi=150)
plt.close()
