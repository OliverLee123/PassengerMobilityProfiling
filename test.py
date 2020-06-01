import torch
import time
from torch.utils.data import DataLoader
import model
import config
import numpy as np




if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # Load data
    #trips = model.TripsData(tripPath = 'E:\\key\\final\\0417_train.csv',labelPath = 'E:\\key\\final\\real0417t_nor.csv')
    trips = model.TripsData(tripPath = 'E:\\key\\final\\0417_train.csv',labelPath = 'E:\\key\\final\\real0417_nnor.csv')
    loader = DataLoader(dataset=trips, batch_size=config.BATCH_SIZE,collate_fn=model.CollateFn)
    #Test
    g = model.G(subModelPath='E:\\key\\final\\SUBG\\0403t_train.pth')
    g.load_state_dict(torch.load('E:\\key\\final\\G_D\\0403_g.pth'))
    g.eval()
    embeddings = []
    for i,[data,length,real] in enumerate(loader):
        data = torch.nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True)
        ansG,embedding = g(data.float()) # batchsize*labeldim 128*242
        embeddings.extend(embedding.detach().numpy())
    embeddings = np.array(embeddings,dtype = float)
    embeddings = embeddings.reshape(-1,config.EMBEDDING_DIM)
    np.savetxt('E:\\key\\final\\RES\\0417.csv',embeddings,delimiter=',')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))