import torch
import time
from torch.utils.data import DataLoader
import model
import config
import matplotlib.pyplot as plt




if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # Load data
    emtrips = model.EMTripsData('E:\\key\\final\\0417EMTrain.csv','E:\\key\\final\\0417EMLabel.csv')
    loader = DataLoader(dataset=emtrips, batch_size=config.BATCH_SIZE)
    #Train
    em = model.EM(11,121)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(em.parameters(), lr=0.0005)
    lossPlot = []
    for epoch in range(config.EPOCH):
        for i,[train,label] in enumerate(loader):
            train = torch.autograd.Variable(train,requires_grad=True)
            label = torch.autograd.Variable(label.squeeze())
            pre = em(train)
            optimizer.zero_grad()
            loss = criterion(pre,label)
            loss.backward()
            optimizer.step()
        lossPlot.append(loss)
        print('EPOCH: ',epoch,' LOSS: ' ,loss.data)
    pltRange = range(0,config.EPOCH)
    plt.plot(pltRange,lossPlot)
    torch.save(em.state_dict(),'E:\\key\\final\\SAVED_MODEL\\0417_em.pth')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))