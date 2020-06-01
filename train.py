import torch
import time
from torch.utils.data import DataLoader
import model
import config
import matplotlib.pyplot as plt




if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # Load data
    trips = model.TripsData(tripPath = 'E:\\key\\final\\0403_train.csv',labelPath = 'E:\\key\\final\\real0403_nor.csv')
    loader = DataLoader(dataset=trips, batch_size=config.BATCH_SIZE,collate_fn=model.CollateFn)
    #Train
    g = model.G(subModelPath='E:\\key\\final\\SUBG\\0403t_train.pth')
    d = model.D()
    criterion = torch.nn.BCELoss()
    DOptimizer = torch.optim.Adam(d.parameters(), lr=0.0003)
    GOptimizer = torch.optim.Adam(g.parameters(), lr=0.0001)
    gLossPlot = []
    dLossPlot = []
    for epoch in range(config.EPOCH):
        for i,[data,length,real] in enumerate(loader):
            realLabel = torch.autograd.Variable(torch.ones(len(data),1))
            fakeLabel = torch.autograd.Variable(torch.zeros(len(data),1))
            data = torch.nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True)
            DOptimizer.zero_grad()
            GOptimizer.zero_grad()
            # train G
            ansG,embedding = g(data.float()) # batchsize*labeldim 128*242
            gLoss = criterion(d(ansG),realLabel)
            gLoss.backward(retain_graph=True)
            GOptimizer.step()
            # train D
            dLoss = (criterion(d(real),realLabel) +criterion(d(ansG),fakeLabel)) /2
            dLoss.backward(retain_graph=True)
            DOptimizer.step()
        gLossPlot.append(gLoss)
        dLossPlot.append(dLoss)
        print('EPOCH: ',epoch,' GLOSS: ' ,gLoss.data,' DLOSS: ' ,dLoss.data)
    pltRange = range(0,config.EPOCH)
    plt.subplot(2,1,1)
    plt.plot(pltRange,gLossPlot)
    plt.subplot(2,1,2)
    plt.plot(pltRange,dLossPlot)
    torch.save(g.state_dict(),'E:\\key\\final\\G_D\\0403_g.pth')
    torch.save(d.state_dict(),'E:\\key\\final\\G_D\\0403_d.pth')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))