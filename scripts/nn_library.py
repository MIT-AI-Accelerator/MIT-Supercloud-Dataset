#########
# Modules
#########

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

######################
# dcc_classifiers code
######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_registry = {}
def register(cls):
    model_registry[cls.__name__] = cls
    return cls

@register
class ts_class_naive(nn.Module):
    def __init__(self, timesteps, series, hidden_size, output_size, rnn_layers, dropout=0.5):
        super().__init__()
        
        self.timesteps = timesteps
        self.series = series
        self.hidden_size = hidden_size   
        self.output_size = output_size
        self.rnn_layers = rnn_layers
        
        self.rnn = nn.LSTM(series, hidden_size, rnn_layers,
                           batch_first=True, bidirectional=True, dropout=dropout) # [N, L, D*h]
        self.fc = nn.Linear(2*timesteps*hidden_size, timesteps)
        self.out = nn.Linear(timesteps, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        
    def forward(self, input):
        # input: [batch, timesteps, series]

        hidden, cell = self.init_hidden(input)
        batch = input.size(0)
        
        output, (hidden, cell) = self.rnn(input, (hidden, cell)) # [batch, timesteps, 2*hidden]
        output = output.contiguous().view(batch, -1) # [batch, 2*timesteps*hidden]
        
        output = self.fc(output) # [batch, timesteps]
        output = self.relu(self.dropout(output)) # [batch, timesteps]
        
        output = self.out(output) # [batch, out_size]
        
        return output
        
    def init_hidden(self, input):
        hidden = torch.zeros(2*self.rnn_layers, input.size(0), self.hidden_size, dtype=torch.double).to(device)
        cell = torch.zeros(2*self.rnn_layers, input.size(0), self.hidden_size, dtype=torch.double).to(device)
        return hidden, cell
        
@register
class ts_class_cnnlstm(nn.Module):
    def __init__(self, timesteps, series, hidden_size, output_size, rnn_layers, dropout=0.5):
        super().__init__()
        
        self.timesteps = timesteps
        self.series = series
        self.hidden_size = hidden_size   
        self.output_size = output_size
        self.rnn_layers = rnn_layers
        
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=10, kernel_size=5, stride=1, padding=2) # (N, C, L)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3) # (N, C, L)
        self.conv2 = nn.Conv1d(10, 14, 5, 1, 2)

        self.rnn = nn.LSTM(series*2, hidden_size, rnn_layers,
                           batch_first=True, bidirectional=True, dropout=dropout) # [N, L, D*h]

        self.fc = nn.Linear(2*60*hidden_size, timesteps)
        self.out = nn.Linear(timesteps, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        
    def forward(self, input):
        # input: [batch, timesteps, series]

        batch = input.size(0)
        
        input = input.permute(0, 2, 1)
        
        conv = self.conv1(input) # [batch, series, timesteps]
        conv = self.relu(conv)
        conv = self.pool(conv)
        
        conv = self.conv2(conv)
        conv = self.relu(conv)
        conv = self.pool(conv)
        
        conv = conv.permute(0, 2, 1)
        
        hidden, cell = self.init_hidden(conv)

        output, (hidden, cell) = self.rnn(conv, (hidden, cell)) # [batch, timesteps, 2*hidden]
        
        output = output.contiguous().view(batch, -1) # [batch, 2*timesteps*hidden]
        output = self.fc(output) # [batch, timesteps]
        output = self.relu(self.dropout(output)) # [batch, timesteps]
        output = self.out(output) # [batch, out_size]
        
        return output
        
    def init_hidden(self, input):
        hidden = torch.zeros(2*self.rnn_layers, input.size(0), self.hidden_size, dtype=torch.double).to(device)
        cell = torch.zeros(2*self.rnn_layers, input.size(0), self.hidden_size, dtype=torch.double).to(device)
        return hidden, cell

@register
class ts_class_cnnlstm_smallwindow(nn.Module):
    def __init__(self, timesteps, series, hidden_size, output_size, rnn_layers, dropout=0.5):
        super().__init__()
        
        self.timesteps = timesteps
        self.series = series
        self.hidden_size = hidden_size   
        self.output_size = output_size
        self.rnn_layers = rnn_layers
        
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=10, kernel_size=3, stride=1, padding=2) # (N, C, L)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2) # (N, C, L)
        self.conv2 = nn.Conv1d(10, 14, 3, 1, 2)

        self.rnn = nn.LSTM(series*2, hidden_size, rnn_layers,
                           batch_first=True, bidirectional=True, dropout=dropout) # [N, L, D*h]

        self.fc = nn.Linear(138240, timesteps)
        self.out = nn.Linear(timesteps, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        
    def forward(self, input):
        # input: [batch, timesteps, series]

        batch = input.size(0)
        
        input = input.permute(0, 2, 1)
        
        conv = self.conv1(input) # [batch, series, timesteps]
        conv = self.relu(conv)
        conv = self.pool(conv)
        
        conv = self.conv2(conv)
        conv = self.relu(conv)
        conv = self.pool(conv)
        
        conv = conv.permute(0, 2, 1)
        
        hidden, cell = self.init_hidden(conv)

        output, (hidden, cell) = self.rnn(conv, (hidden, cell)) # [batch, timesteps, 2*hidden]
        
        output = output.contiguous().view(batch, -1) # [batch, 2*timesteps*hidden]
        output = self.fc(output) # [batch, timesteps]
        output = self.relu(self.dropout(output)) # [batch, timesteps]
        output = self.out(output) # [batch, out_size]
        
        return output
        
    def init_hidden(self, input):
        hidden = torch.zeros(2*self.rnn_layers, input.size(0), self.hidden_size, dtype=torch.double).to(device)
        cell = torch.zeros(2*self.rnn_layers, input.size(0), self.hidden_size, dtype=torch.double).to(device)
        return hidden, cell
        
###################
# dcc_dadtaset code
###################

class dccDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]
    
############
# train code
############

class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
    
def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    
    return scheduler

def train(train_dl, test_dl, model, model_name, num_epochs, writer, 
          save_model=True, learning_rate=0.01, patience=100, print_every=None):
    
    print_every = len(train_dl)/2
    
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sched = CyclicLR(optim, cosine(t_max=print_every * 2, eta_min=learning_rate/100))
    
    train_loss_all = []
    test_loss_all = []
    train_accuracy_all = []
    test_accuracy_all = []
    
    best_acc = 0
    trials = 0
        
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_test_loss = 0

        train_correct, train_total = 0, 0
        for i, (data_x, data_y) in enumerate(train_dl):
            model.train()
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optim.zero_grad()
            train_out = model(data_x)
            
            train_loss = criterion(train_out, data_y)
            writer.add_scalar("Loss/train", train_loss, epoch)
            total_train_loss += float(train_loss.detach()/data_x.size(0))
            
            train_preds = F.log_softmax(train_out, dim=1).argmax(dim=1)
            train_total += data_y.size(0)
            train_correct += (train_preds == data_y).sum().item()
            
            train_loss.backward()
            optim.step()
            sched.step()

        train_loss_all.append(total_train_loss)
        
        train_accuracy = train_correct/train_total
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        train_accuracy_all.append(train_accuracy)
            
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for test_x, test_y in test_dl:
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                
                test_out = model(test_x)
                
                test_loss = criterion(test_out, test_y)
                writer.add_scalar("Loss/test", test_loss, epoch)
                total_test_loss += float(test_loss.detach()/test_x.size(0))
                
                test_preds = F.log_softmax(test_out, dim=1).argmax(dim=1)
                test_total += test_y.size(0)
                test_correct += (test_preds == test_y).sum().item()
                
            test_loss_all.append(total_test_loss)
            
            test_accuracy = test_correct/test_total
            writer.add_scalar("Accuracy/test", test_accuracy, epoch)
            test_accuracy_all.append(test_accuracy)
        
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Train Loss: {total_train_loss:.4f}. Test Loss: {total_test_loss:.4f}. Train Acc.: {train_accuracy:2.2%} test Acc.: {test_accuracy:2.2%}')

        if test_accuracy > best_acc:
            trials = 0
            best_acc = test_accuracy
            savename_best = "./models/" + model_name + "_best.pth"
            if save_model:
                torch.save(model.state_dict(), savename_best)
                print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                print(f'Best accuracy: {best_acc:2.2%} at epoch {epoch-patience}')
                break

    return (train_loss_all, test_loss_all, test_accuracy_all)
