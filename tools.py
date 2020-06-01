import csv
import copy
import math
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import trip

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
This package contains some function in a mess
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def LoadFP(path): # load frequent pattern from file
    fpAns = []
    fps = csv.reader(open(path,'r',encoding='utf-8'))
    for fp in fps:
        if(len(fp)>1): # ignore 1 length fp
            fp = list(map(eval,fp))
            fpAns.append(copy.deepcopy(fp))
    return fpAns


def TransferToFPTrip(trip, fps):
    '''
    Judge if a trip contains fp and transfer to fp trip
    Consider only 2 length fp temporarily
    If a trip contains fp,then transfer to fp; else ratain the whole raw trip
    '''
    fpTrip = []
    ifTrans = True
    for fp in fps:
        for i in range(len(trip)-1):
            if(trip[i]==fp[0] and trip[i+1]==fp[1]):
                for f in fp:
                    fpTrip.append(f)
    if(len(fpTrip)>0):
        return fpTrip, ifTrans
    else:
        ifTrans = False
        return trip, ifTrans

def StopsMapToFGrid(stopsDict, fGrid):
    '''
    map stops to frequent grids
    '''
    # Create clean frequent grid
    fgrid = []
    index = 0
    for fg in fGrid:
        fgrid.append([index,fg[0],fg[1],fg[2],fg[3]])
        index += 1
    stopsFGridDict = {} # stopsFGridDict[stopsID] = fgridID(fgrid[0])
    for key,value in stopsDict.stopsPosDict.items():
        for f in fgrid:
            if(value[0]>=f[1] and value[0]<=f[3] and value[1]>=f[2] and value[1]<=f[4]):
                stopsFGridDict[key] = f[0]
    print('Mapping stop to frequent grid successfully.Total dict lenght:'+str(index))
    return stopsFGridDict,index

def StrTimeToFloatTime(strTime):
    '''
    unit : hour
    '''
    floatT = strTime.split(' ')[1].split(':')
    ans = float(floatT[0]) + float(floatT[1])/60 + float(floatT[1])/3600
    return ans

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Process trip for training
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def Trip2Vec(trip):
    '''
    input: trip - str
    output: vec - float
    '''
    Otime = StrTimeToFloatTime(trip[1])
    # a cycle is 24 hours
    Otimex = math.cos(Otime*math.pi*2/24)
    Otimey = math.sin(Otime*math.pi*2/24)
    Dtime = StrTimeToFloatTime(trip[5])
    Dtimex = math.cos(Dtime*math.pi*2/24)
    Dtimey = math.sin(Dtime*math.pi*2/24)
    return trip[0],[Otimex,Otimey,float(trip[3]),float(trip[4]),Dtimex,Dtimey,float(trip[7]),float(trip[8]),float(trip[11]),float(trip[12]),float(trip[13])]


def ConTrainData(path,outputPath):
    '''
    Make train examples of trip
    input: trip path
    output: trip examples for training
    '''
    idData = []
    tripData = []
    cnt = 0
    trips = csv.reader(open(path,'r',encoding='utf-8'))
    for trip in trips:
        userid,trip = Trip2Vec(trip)
        idData.append(userid)
        tripData.append(trip)
        cnt += 1
    tripData = np.array(tripData)
    tripData = preprocessing.scale(tripData)
    tripTmp = []
    data = []
    for i in range(cnt):
        if(i==0):
            tripTmp.append(tripData[i].tolist())
        else:
            if(idData[i]==idData[i-1]):
                tripTmp.append(tripData[i].tolist())
            else:
                data.append(copy.deepcopy(tripTmp))
                tripTmp.clear()
                tripTmp.append(tripData[i].tolist())
    if(len(tripTmp)>0):
        data.append(copy.deepcopy(tripTmp))
        tripTmp.clear()
    '''
    data eg :[[[1,2],[1,2]],[[1,2],[1,2]]]
    '''
    with open(outputPath, 'w', encoding='utf-8', newline='') as f:
        for dss in data:
            for ds in dss:
                f.write(str(len(dss)))
                for d in ds:
                    f.write(',')
                    f.write(str(d))
            f.write('\n')
        print('Train data to path:'+outputPath)
    return data

def Normalization(inputPath, outputPath):
    data = pd.read_csv(inputPath,header = None)
    v = preprocessing.scale(data.values)
    np.savetxt(outputPath,v,delimiter=',',fmt='%f')

def CleanTrip(inputPath, outputPath):
    with open(outputPath,'w', encoding='utf-8',newline='')as f:
        ff = csv.writer(f)
        trips = csv.reader(open(inputPath,'r',encoding='utf-8'))
        for trip in trips:
            if(trip[3][0] != 'N'):
                ff.writerow(trip)
                
                
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Cluster analysis
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''   


def Cluster(inputPath,clustersCnt):
    data = pd.read_csv(inputPath,header = None)
    ac = AgglomerativeClustering(n_clusters=clustersCnt,linkage='ward').fit(data.values)
    #ac = KMeans(n_clusters=clustersCnt, random_state=3).fit(data.values)
    return ac.labels_

def ClusterTrips(tripPath, embeddingPath,outputPath,clustersCnt):
    labels = Cluster(embeddingPath,clustersCnt)
    idx = 0
    fid = 0
    with open(outputPath,'w', encoding='utf-8',newline='')as f:
        ff = csv.writer(f)
        trips = csv.reader(open(tripPath,'r',encoding='utf-8'))
        isFirst = True
        foreID = ""
        curID = ""
        for trip in trips:
            if(isFirst):
                foreID = trip[0]
                curID = trip[0]
                ff.writerow([fid,labels[idx],trip[0],trip[1],trip[2],float(trip[3]),float(trip[4]),trip[5]
                ,trip[6],trip[7],trip[8],trip[9],trip[11],trip[12],trip[13]])
                isFirst = False
                fid = fid+1
            else:
                curID = trip[0]
                if(curID != foreID):
                    idx = idx + 1
                    foreID = curID
                ff.writerow([fid,labels[idx],trip[0],trip[1],trip[2],trip[3],trip[4],trip[5]
                ,trip[6],trip[7],trip[8],trip[9],trip[11],trip[12],trip[13]])
                fid = fid+1
                
    CalClusteredPassenger(outputPath,clustersCnt)

def JudgeType(trips):
    '''
    Judge passenger type
    0: can't defind
    1: commuter
    2: actual trip with purpose
    3: others
    '''
    if(len(trips)==1):
        return 0
    elif(len(trips)==2):
        oneOT = StrTimeToFloatTime(trips[0][3])
        oneDT = StrTimeToFloatTime(trips[0][7])
        twoOT = StrTimeToFloatTime(trips[1][3])
        if(oneOT<10 and twoOT>17 and twoOT-oneDT>6 and trips[0][4] == trips[1][8]):
            return 1
        else:
            return 2
    else:
        return 3

def CalClusterFeatures(trips):
    l = 0
    td = 0.0 # Time distance
    sc = 0.0 # stops count
    tc = 0.0 # transfer count
    for trip in trips:
        l += 1
        td = td + float(trip[12])
        sc = sc + float(trip[13])
        tc = tc + float(trip[14])
    return [td/l,sc/l,tc/l]

def CalClusteredPassenger(tripClusterPath,clusterCnt,typeCnt=4):
    '''
    typeCnt: four type of passengers
    '''
    trips = csv.reader(open(tripClusterPath,'r',encoding='utf-8'))
    isFirst = True
    foreID = ""
    curID = ""
    tripsTmp = []
    m = np.zeros((clusterCnt,typeCnt))
    timeDis = np.zeros((clusterCnt))
    stopsCnt = np.zeros((clusterCnt))
    transferCnt = np.zeros((clusterCnt))
    for trip in trips:     
        if(isFirst):
            foreID = trip[2]
            curID = trip[2]
            tripsTmp.append(trip)
            isFirst = False
        else:
            curID = trip[2]
            if(curID == foreID):
                tripsTmp.append(trip)
            else:
                t = JudgeType(tripsTmp)
                c = int(tripsTmp[0][1])
                [td,sc,tc] = CalClusterFeatures(tripsTmp)
                timeDis[c] += td
                stopsCnt[c] += sc
                transferCnt[c] += tc
                m[c][t] += 1
                tripsTmp.clear()
                foreID = curID
                tripsTmp.append(trip)
        # the last passenger
    t = JudgeType(tripsTmp)
    c = int(tripsTmp[0][1])
    m[c][t] += 1
    [td,sc,tc] = CalClusterFeatures(tripsTmp)
    timeDis[c] += td
    stopsCnt[c] += sc
    transferCnt[c] += tc
    tripsTmp.clear()
    rsum = m.sum(axis=1).reshape(clusterCnt,1)
    rrsum = m.sum(axis=1).reshape(1,clusterCnt)
    mm = np.true_divide(m,rsum)
    print(rrsum)
    print(mm)
    print(np.true_divide(timeDis,rrsum))
    print(np.true_divide(stopsCnt,rrsum))
    print(np.true_divide(transferCnt,rrsum))
    return m
     
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Compared experiment
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''  

def CETrip2Vec(trip,embedding):
    st = StrTimeToFloatTime(trip[1])
    slng = float(trip[3])
    slat = float(trip[4])
    dur = float(trip[11])
    stops = float(trip[12])
    transfer = float(trip[13])
    cetrip = [st,slng,slat,dur,stops,transfer,embedding[0],embedding[1],
              embedding[2],embedding[3],embedding[4]]
    return cetrip

def ConCETrip(embeddingPath, tripPath,outputPath):
    trips = csv.reader(open(tripPath,'r',encoding='utf-8'))
    embeddings = pd.read_csv(embeddingPath,header = None)
    CETrips = []
    isFirst = True
    foreID = ""
    curID = ""
    i = 0
    for t in trips:
        if(isFirst):
            foreID = t[0]
            curID = t[0]
            CETrips.append(CETrip2Vec(t,embeddings.iloc[i,:].values))
            isFirst = False
        else:
            curID = t[0]
            if(curID != foreID):
                i += 1
                foreID = curID
            CETrips.append(CETrip2Vec(t,embeddings.iloc[i,:]))
    CETrips = np.array(CETrips)
    CETrips = preprocessing.scale(CETrips)
    #extra = np.zeros((121,11)) #121 means the number of grid
    #CETrips = np.concatenate((extra,CETrips),axis=0)
    np.savetxt(outputPath,CETrips,delimiter=',',encoding='utf-8')
    
def ConCELabel(gridPath,tripPath,outputPath):
    fgs = trip.FrequentGrid()
    fgs.InitFGridsFromFile(gridPath)
    labels = []
    trips = csv.reader(open(tripPath,'r',encoding='utf-8'))
    flen = len(fgs.fGrids)
    for t in trips:
        elng = float(t[7])
        elat = float(t[8])
        i = 0
        label = np.zeros((1,flen))
        for fg in fgs.fGrids:
            lllng = float(fg[0])
            lllat = float(fg[1])
            urlng = float(fg[2])
            urlat = float(fg[3])
            if(elng>lllng and elng<urlng and elat>lllat and elat<urlat):
                label[0][i]=1
                label = label.reshape(flen)
                labels.append(copy.deepcopy(label))
                break
            i += 1
    labels = np.array(labels)
    np.savetxt(outputPath,labels,delimiter=',',encoding='utf-8',fmt='%d')
                
        
       
    
    


