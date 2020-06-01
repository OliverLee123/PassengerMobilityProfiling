import trip
from numpy import *
import time
import csv
import copy

class Apriori():
    def __init__(self, trips):
        self.trips = trips
        self.minSupport = 0.0005
        self.durationDiff = 30

        self.L = []
        self.supportData = {}
    
    def CreateC1(self):
        C1 = []
        for tripPer in self.trips:
            for item in tripPer:
                if not [item] in C1:
                    C1.append([item])
        C1.sort()
        return list(map(frozenset, C1))
 
    def ScanTrips(self, Ck):
        ssCnt = {}
        for tid in self.trips:
            for can in Ck:
                if can.issubset(tid):   
                    if can not in ssCnt:
                        ssCnt[can] = 1
                    else:
                        ssCnt[can] += 1
        numItems = float(len(self.trips))
        retList = []       
        supportData = {}   
        for key in ssCnt:
            support = ssCnt[key] / numItems
            if support >= self.minSupport:
                retList.insert(0, key)
            supportData[key] = support
        return retList, supportData
 
    def AprioriGen(self, Lk, k):
        retList = []
        lenLk = len(Lk)
        for i in range(lenLk): 
            for j in range(i+1, lenLk):
                L1 = list(Lk[i])[:k-2]  
                L2 = list(Lk[j])[:k-2]
                L1.sort(); L2.sort()      
                if L1==L2:
                    retList.append(Lk[i] | Lk[j]) 
        return retList
 
    def Process(self):
        self.trips = list(map(set, self.trips))
        C1 = self.CreateC1()      
        L1, supportData = self.ScanTrips(C1)  
        self.L = [L1]
        k = 2
        while (len(self.L[k-2]) > 0):
            print(k)    
            Ck = self.AprioriGen(self.L[k-2], k) 
            Lk, supK = self.ScanTrips(Ck) 
            self.supportData.update(supK)    
            self.L.append(Lk)
            k += 1  

    def FPToCSV(self, path): # frequent pattern to file
        with open(path,'w',encoding='utf-8',newline='') as f:
            w = csv.writer(f)
            for l in self.L:
                for t in l:
                    w.writerow(t)


    
