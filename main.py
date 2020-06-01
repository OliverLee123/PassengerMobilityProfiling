import trip
import time
import apriori
import numpy as np
import tools

'''
pipeline:
1. ExtractFP
2. CreateFrequnentGrid
3. CreateIndividualMotifVector(include CreateFPTrips)
4. CreateGroupTrips(slow and need to accelarate)
5. ConstructTrainedTrip(tools.ConTrainData)
6. Normalization(tools.Normalization) real0403->real0403_nor   fake0403->fake0403_nor
7.Cluster
8.Compared experiment
'''
def ExtractFP():
    # Paras
    path = "C:\\Users\\lyc\\Desktop\\final\\20170403.csv" # trip path
    startStopIndex = 2
    endStopIndex = 6
    # Create dict and data
    stopsDict = trip.StopsDict()
    stopsDict.CreateStopsDictFromFile(path, startStopIndex, endStopIndex)
    tripsData = trip.TripsData()
    tripsData.CreateRawODTrips(path, stopsDict.stopsDict, startStopIndex, endStopIndex)
    # Apriori
    apr = apriori.Apriori(tripsData.rawODTrips)
    apr.Process()
    apr.FPToCSV("C:\\Users\\lyc\\Desktop\\final\\0403fp_0.0005.csv") # output path

def CreateFrequnentGrid():
    # Init frequent grid from trip 
    grid = trip.Grid(476300,2484500,559500,2524600)
    grid.InitGridOccurFromFile('C:\\Users\\lyc\\Desktop\\final\\20170403.csv') # trip path
    fgrid = trip.FrequentGrid(maxOccur=100000, mGW=2500, mGH=2500)# max occur, max grid width,max grid height
    fgrid.FilterByQuatree([grid])
    fgrid.FGridsToFile('C:\\Users\\lyc\\Desktop\\final\\fgrid0403.csv') # output path

def CreateFPTrips():
    # Paras
    tripPath = "D:\\20170417.csv"
    dicPath = "E:\\key\\final\\20170403.csv"
    FPPath = "E:\\key\\final\\0403fp_0.0005.csv"
    startStopIndex = 2
    endStopIndex = 6
    # Create dict and data
    stopsDict = trip.StopsDict()
    stopsDict.CreateStopsDictFromFile(dicPath, startStopIndex, endStopIndex)
    tripsData = trip.TripsData()
    tripsData.CreateFPODTrips(tripPath, FPPath, stopsDict.stopsDict, startStopIndex, endStopIndex)
    return stopsDict, tripsData

def CreateIndividualMotifVector():
    vs = []
    cnt = 0
    # Load the frequent grid
    fgrid = trip.FrequentGrid()
    fgrid.InitFGridsFromFile('E:\\key\\final\\fgrid0403.csv') # frequent grid path
    stopsDict, tripsData = CreateFPTrips()
    stopsGridDict,vLen = tools.StopsMapToFGrid(stopsDict,fgrid.fGrids)
    for fptrip in tripsData.FPODTrips:
        # each person has a motif matrix
        # matrix(i,j) means O-i, D-j
        matrix = np.zeros((vLen,vLen))
        tripLen = 0.5*len(fptrip)
        for i in range(int(tripLen)):          
            matrix[stopsGridDict[fptrip[2*i]]][stopsGridDict[fptrip[2*i+1]]] += 1
        
        # cal row(O) vector
        rowV = np.sum(matrix,axis=1)
        # cal col(D) vector
        colV = np.sum(matrix,axis=0)
        rowV = rowV.reshape((1,vLen))
        colV = colV.reshape((1,vLen))
        v = np.concatenate((rowV,colV),axis=1)
        v = v.reshape(v.shape[1])
        vs.append(v)
        
        cnt += 1
    vs = np.array(vs)
    np.savetxt('E:\\key\\final\\real0417.csv',vs,delimiter=',',fmt='%d')
    print('Create individual motif vector successfully.Total count:'+str(cnt))
    return vs

def CreateGroupTrips():
    path = "E:\\key\\final\\tdata.csv" # trip path
    qpath = "E:\\key\\final\\20170403.csv" # query trip path
    wpath = "E:\\key\\final\\20170403.csv" # whole path

    stopsDict = trip.StopsDict()
    stopsDict.CreateStopsDictFromFile(wpath, 2, 6)
    tripsData = trip.TripsData()
    tripsData.CreateGroupTrips(path, qpath, stopsDict.stopsDict, 1000, 1) 
    
    vs = []
    cnt = 0
    # Load the frequent grid
    fgrid = trip.FrequentGrid()
    fgrid.InitFGridsFromFile('E:\\key\\final\\fgrid0403.csv') # frequent grid path
    stopsGridDict,vLen = tools.StopsMapToFGrid(stopsDict,fgrid.fGrids)
    for fptrip in tripsData.groupTrips:
        # each person has a motif matrix
        # matrix(i,j) means O-i, D-j
        matrix = np.zeros((vLen,vLen))
        tripLen = 0.5*len(fptrip)
        for i in range(int(tripLen)):          
            matrix[stopsGridDict[fptrip[2*i]]][stopsGridDict[fptrip[2*i+1]]] += 1
        # cal row(O) vector
        rowV = np.sum(matrix,axis=1)
        # cal col(D) vector
        colV = np.sum(matrix,axis=0)
        rowV = rowV.reshape((1,vLen))
        colV = colV.reshape((1,vLen))
        v = np.concatenate((rowV,colV),axis=1)
        v = v.reshape(v.shape[1])
        vs.append(v)
        cnt += 1
    vs = np.array(vs)
    np.savetxt('E:\\key\\final\\fake0403t.csv',vs,delimiter=',',fmt='%d')
    print('Create group motif vector successfully.Total count:'+str(cnt))



if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    #CreateIndividualMotifVector()
    #tools.Normalization('E:\\key\\final\\real0417t.csv','E:\\key\\final\\real0417t_nor.csv')
    #tools.ConTrainData("D:\\20170422.csv",'E:\\key\\final\\0422_train.csv')
    #tools.CleanTrip('D:\\20170417.csv','D:\\201704171.csv')
    tools.ClusterTrips('D:\\0417tt.csv','E:\\key\\final\\RES\\0417tt.csv','E:\\key\\final\\RES\\0417tt_c.csv',5)
    #tools.ConCELabel('E:\\key\\final\\fgrid0403.csv','D:\\0417tt.csv','E:\\key\\final\\0417ttEMLabel1.csv')
    #tools.ConCETrip('E:\\key\\final\\RES\\0417.csv','D:\\0417.csv','E:\\key\\final\\0417EMTrain.csv')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

 
