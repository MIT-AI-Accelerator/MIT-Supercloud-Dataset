#########
# Modules
#########

import numpy as np
import os
import sys
import datetime
import torch
import torch.utils.data as torch_data
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from nn_library import dccDataset, train, model_registry

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
NN_MODEL = sys.argv[2]
HIDDEN_SIZE = int(sys.argv[3])
RNN_LAYERS = int(sys.argv[4])
TIMESTAMP = datetime.datetime.fromtimestamp(int(sys.argv[5])).strftime('%Y%m%d%H%M%S')
DATA_PATH = os.path.join(DATA_PATH,DATA_DIR,DATA_FILE,'ml.npz')
EXPERIMENT_NAME = 'nn-{}-{}-{}-{}'.format(DATA_FILE,NN_MODEL,HIDDEN_SIZE,RNN_LAYERS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_PATH = os.path.join(ROOT_PATH,RESULTS_DIR,'nn/{}'.format(TIMESTAMP))
try: os.makedirs(RESULTS_PATH)
except: pass

sys.stdout = open(os.path.join(RESULTS_PATH,EXPERIMENT_NAME+'.txt'), 'w')

#################
# Hyperparameters
#################

seed = 2022
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

max_workers = 0
batch_size = 32

##############
# Load ML Data
##############

ml_data = np.load(ML_DATA_PATH)
X_train, y_train, X_test, y_test = ml_data['X_train'], ml_data['y_train'],ml_data['X_test'],ml_data['y_test']
n_classes = len(np.unique(y_train))

# standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# build datasets
dset_train = dccDataset(X_train,y_train)
dset_test = dccDataset(X_test,y_test)

# torch dataloaders
train_dl = torch_data.DataLoader(dset_train, batch_size=batch_size, num_workers=max_workers, shuffle=True)
test_dl = torch_data.DataLoader(dset_test, batch_size=batch_size, num_workers=max_workers, shuffle=False)

#######
# Train
#######

model = model_registry[NN_MODEL](timesteps=540,
                                 series=7,
                                 hidden_size=HIDDEN_SIZE,
                                 output_size=n_classes,
                                 rnn_layers=RNN_LAYERS).double().to(DEVICE)

writer = SummaryWriter()

train_results = train(train_dl,
                      test_dl,
                      model,
                      NN_MODEL,
                      num_epochs=1000,
                      writer=writer)
