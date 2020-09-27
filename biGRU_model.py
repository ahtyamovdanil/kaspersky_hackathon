from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from jupyterplot import ProgressPlot
import numpy as np

class Dataset(TorchDataset):
    
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __getitem__(self, index):
        
        """
        Returns one tensor pair (source and target). The source tensor corresponds to the input word,
        with "BEGIN" and "END" symbols attached. The target tensor should contain the answers
        for the language model that obtain these word as input.        
        """
        source = torch.tensor([self.vocab['BEGIN']] + [self.vocab[ch] for ch in self.data[index, 0]])
        if self.data.shape[1] == 1:
            return source
        else:
            target = torch.tensor([self.data[index, 1]])
            return source, target

    def __len__(self):
        return len(self.data)    

    
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers = 1, device='cuda'):
        super(BiGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.kaiming_uniform_(self.embedding.weight)
        nn.init.kaiming_uniform_(self.fc.weight)
        
    def forward(self, inputs, hidden = None):
        if(hidden == None):
            hidden = self.init_hidden(inputs.shape[0])
        embd = self.embedding(inputs)
        out_rnn, hidden = self.rnn(embd, hidden)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        out = self.sigmoid(self.fc(hidden))
        return out
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
    
    
class Padder:
    def __init__(self, dim=0, pad_symbol=0):
        self.dim = dim
        self.pad_symbol = pad_symbol
    
    def pad_tensor(self, vec, length, dim, pad_symbol):
        """
        Pads a vector ``vec`` up to length ``length`` along axis ``dim`` with pad symbol ``pad_symbol``.
        """
        pad_size = list(vec.shape)
        pad_size[dim] = length - vec.size(dim)
        
        return torch.cat([vec, torch.zeros(*pad_size, dtype=torch.long).fill_(pad_symbol)], dim=dim) 
    
    def __call__(self, batch):
        #if len(batch[0].shape) == 1:
        if type(batch[0]) == torch.Tensor:
            max_len = max([x.shape[self.dim] for x in batch])
            batch = [[self.pad_tensor(x, length=max_len, dim=self.dim, pad_symbol=self.pad_symbol)] \
                     for x in batch]
            xs = torch.stack([row[0] for row in batch], dim=0)
            return xs
        else:
            max_len = max([x.shape[self.dim] for x,y in batch])
            batch = [[self.pad_tensor(x, length=max_len, dim=self.dim, pad_symbol=self.pad_symbol), y] \
                     for x,y in batch]
            xs = torch.stack([row[0] for row in batch], dim=0)
            ys = torch.tensor([row[1] for row in batch]) 
            return xs, ys
        
        raise IndexError('index out of bounds')
    
    
def validate_on_batch(model, criterion, x, y):
    out = model(x)
    return criterion(out.view(-1), y.view(-1))


def train_on_batch(model, criterion, x, y, optimizer):
    optimizer.zero_grad()
    loss = validate_on_batch(model, criterion, x, y)
    loss.backward()
    optimizer.step()
    return loss


def get_scores(model, test_loader, device='cuda'):
    prob_all = np.empty(0)
    pred_labels = np.empty(0)
    y_all = np.empty(0)
    with torch.no_grad():
        for x, y in test_loader:
            pred_prob = model(x.to(device)).view(-1).cpu().detach().numpy()
            prob_all = np.concatenate([prob_all, pred_prob])
            y_all = np.concatenate([y_all, y.view(-1).cpu().detach().numpy()])
            pred_labels = np.concatenate([pred_labels, np.array([0 if item < 0.5 else 1 for item in pred_prob])])
    print('roc_auc ', roc_auc_score(y_all, prob_all))
    print('f1 ', f1_score(y_all, pred_labels))

    
def predict_proba(model, vocab, data, device='cuda'):
    data = DataLoader(Dataset(data, vocab), batch_size=50, collate_fn=Padder(dim=0, pad_symbol=vocab['PAD']))
    prob_all = np.empty(0)
    with torch.no_grad():
        for x in data:
            pred_prob = model(x.to(device)).view(-1).cpu().detach().numpy()
            prob_all = np.concatenate([prob_all, pred_prob])
            
    return prob_all


def train_gru(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs, device):
    pp = ProgressPlot(line_names=["train", "valid"])

    for ep in range(1, epochs+1):
        loss_list = []
        for input, target in train_loader:
            loss = train_on_batch(model, criterion, input.to(device), 
                                  target.type(torch.FloatTensor).to(device), optimizer)
            loss_list.append(loss.item())
        scheduler.step()

        with torch.no_grad():
            valid_list = []
            for i, [x_valid, y_valid] in enumerate(val_loader):
                valid_loss = validate_on_batch(model, criterion, 
                                               x_valid.to(device), 
                                               y_valid.type(torch.FloatTensor).to(device))
                valid_list.append(valid_loss.item())

        train_loss = np.mean(loss_list)
        valid_loss = np.mean(valid_list)
        pp.update([[train_loss, valid_loss]])
        print('epoch: {ep} \t train: {train} \t valid: {valid}'.format(ep=ep, 
                                                                       train=round(train_loss, 4),
                                                                       valid=round(valid_loss, 4)))