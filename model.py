import torch
import numpy as np
from torch.utils.data import Dataset
import copy
import csv
import config
import pandas as pd

def CollateFn(tripsData):
    trainData = []
    labelData = []
    for tripData in tripsData:
        trainData.append(tripData[0])
        labelData.append(tripData[1])
    trainData.sort(key=lambda data: len(data), reverse=True)
    length = [len(data) for data in trainData]
    trainData = torch.nn.utils.rnn.pad_sequence(trainData, batch_first=True, padding_value=0)
    return trainData, length, torch.Tensor(np.array(labelData))

class TripsData(Dataset):
    def __init__(self, tripPath, labelPath,dim=config.TRIP_DIM):
        self.trainData = []
        self.labelData = []
        trips = csv.reader(open(tripPath,'r',encoding='utf-8'))
        for trip in trips:
            cnt = int(trip[0])
            vecTmp = []
            for i in range(cnt):
                vecTmp.append(trip[i*dim+1:(i+1)*dim+1])
            t = torch.Tensor(np.array(copy.deepcopy(vecTmp),dtype=float))
            self.trainData.append(t)
        labels = pd.read_csv(labelPath,header = None)
        for index, row in labels.iterrows():
            self.labelData.append((np.array(row,dtype=float)))

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        return [self.trainData[idx],self.labelData[idx]]

class SubG(torch.nn.Module):
    '''
    train examples: trip
    real: group (fake)
    '''
    def __init__(self):
        super(SubG, self).__init__()
        self.gru = torch.nn.GRU(config.TRIP_DIM, 44,batch_first=True)
        self.l1 = torch.nn.Linear(44,80)
        self.l2 = torch.nn.Linear(80,160)
        self.l3 = torch.nn.Linear(160,config.LABEL_DIM)

    def forward(self, x):#x BATCH_SIZE*2(trip counts a man..cmu)*10(input features)
        output ,h = self.gru(x) # h : batch,num_layer,11
        h = h.contiguous().view(-1,44)
        #h, out_len = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = torch.tanh(self.l1(h))
        h = torch.tanh(self.l2(h))
        h = torch.tanh(self.l3(h))
        return h

class G(torch.nn.Module):
    def __init__(self,subModelPath):
        super(G, self).__init__()
        self.enGRU = torch.nn.GRU(config.TRIP_DIM, 44,batch_first=True)
        self.enL1 = torch.nn.Linear(44,80)
        self.enL2 = torch.nn.Linear(80,40)
        self.enL3 = torch.nn.Linear(40,20)
        self.enL4 = torch.nn.Linear(20,config.EMBEDDING_DIM)
        self.deL1 = torch.nn.Linear(config.EMBEDDING_DIM,20)
        self.deL2 = torch.nn.Linear(20,40)
        self.deL3 = torch.nn.Linear(40,80)
        self.deL4 = torch.nn.Linear(80,44)
        self.deGRU = torch.nn.GRU(22, config.TRIP_DIM,batch_first=True)

        self.subG = SubG()
        self.subG.load_state_dict(torch.load(subModelPath))

    def forward(self, x):
        output ,h = self.enGRU(x)
        h = h.contiguous().view(-1,44)
        h = torch.tanh(self.enL1(h))
        h = torch.tanh(self.enL2(h))
        h = torch.tanh(self.enL3(h))
        embedding = torch.tanh(self.enL4(h))
        h = torch.tanh(self.deL1(embedding))
        h = torch.tanh(self.deL2(h))
        h = torch.tanh(self.deL3(h))
        h = torch.tanh(self.deL4(h))

        h = h.contiguous().view(-1,2,22)

        output,h = self.deGRU(h)

        ans = self.subG(output)

        return ans,embedding

class D(torch.nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(config.LABEL_DIM,512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1,config.LABEL_DIM)
        x = self.model(x)
        return x

class EM(torch.nn.Module):
    '''
    Estimate mobility
    Compared experiment
    '''
    def __init__(self,inputDim,outputDim):
        super(EM, self).__init__()

        self.l1 = torch.nn.Linear(inputDim,20)
        self.l2 = torch.nn.Linear(20,40)
        self.l3 = torch.nn.Linear(40,70)
        self.l4 = torch.nn.Linear(70,outputDim)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        
        return x
    
class EMTripsData(Dataset):
    def __init__(self, emtripPath, labelPath):
        self.trainData = torch.FloatTensor(pd.read_csv(emtripPath).values)
        #self.labelData = torch.LongTensor(pd.read_csv(labelPath).values)
        self.labelData = torch.FloatTensor(pd.read_csv(labelPath).values)
        

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        return [self.trainData[idx],self.labelData[idx]]
    