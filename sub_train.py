import torch
import time
from torch.utils.data import DataLoader
import model


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))        
    # Load data
    trips = model.TripsData(tripPath = 'E:\\key\\final\\0403t_train.csv',labelPath = 'E:\\key\\final\\fake0403t_nor.csv')
    loader = DataLoader(dataset=trips, batch_size=64,collate_fn=model.CollateFn)
    '''
    net = torch.nn.LSTM(11, 20, batch_first=True)
    for data, length in loader:
        data = torch.nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True)
        output, hidden = net(data)
        output, out_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        print(output.shape)
    '''
    #Train
    subG = model.SubG()  
    optimizer = torch.optim.SGD(subG.parameters(), lr=20)
    criterion = torch.nn.MSELoss()
    for epoch in range(500):
    	for i,[data,length,label] in enumerate(loader):
            data = torch.nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True)
            optimizer.zero_grad() 
            ans = subG(data.float()) # batchsize*labeldim 128*242
            loss = criterion(ans,label)
            loss.backward()
            optimizer.step()
    	print('EPOCH: ',epoch,' LOSS: ' ,loss.data)
    #Save the model
    torch.save(subG.state_dict(),'E:\\key\\final\\SUBG\\0403t_train_1.pth')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))