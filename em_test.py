import torch
import time
from torch.utils.data import DataLoader
import model
import numpy as np
import config



if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # Load data
    emtrips = model.EMTripsData('E:\\key\\final\\0417ttEMTrain.csv','E:\\key\\final\\0417ttEMLabel1.csv')
    loader = DataLoader(dataset=emtrips, batch_size=config.BATCH_SIZE)
    #Test
    em = model.EM(11,121)
    em.load_state_dict(torch.load('E:\\key\\final\\SAVED_MODEL\\0417tt_em1.pth'))
    preLabels = []
    trueLabels = []
    for i,[test,label] in enumerate(loader):
        pre = em(test)
        em.eval()
        pre = pre.detach().numpy()
        preLabel = np.argmax(pre,axis=1)
        preLabels.extend(preLabel)
        trueLabels.extend(label.detach().numpy())
    preLabels = np.array(preLabels,dtype = int).reshape(-1,1)
    trueLabels = np.array(trueLabels,dtype = int)
    rate = trueLabels - preLabels
    rate = sum(rate==0)
    print(rate)
    out = np.concatenate((preLabels,trueLabels),axis=1)
    np.savetxt('E:\\key\\final\\RES\\0417EM.csv',out,delimiter=',',fmt='%d')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))