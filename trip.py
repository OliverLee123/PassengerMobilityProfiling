import csv
import copy
import tools

class StopsDict():
    def __init__(self):
        self.stopsDict = {}
        self.stopsPosDict = {}

    def CreateStopsDictFromFile(self, path, startStopIndex, endStopIndex):
        '''
        Create the map from stop name to stop id -- stopsDict 
        eg: stopsDict[燕南站]=0, stopsDict[梅景站]=1, ...
        Create the map from stop id to stop position(lng,lat) -- stopsPosDict
        eg: stopsPosDict[0]=[509033.2575,2493390.6817 ], ...                                                        
        '''
        trips = csv.reader(open(path,'r',encoding='utf-8'))
        number = 0
        for trip in trips:
            if(trip[startStopIndex] not in self.stopsDict): # start stop
                self.stopsDict[trip[startStopIndex]] = number
                self.stopsPosDict[number] = [float(trip[3]),float(trip[4])]
                number += 1                
            if(trip[endStopIndex] not in self.stopsDict): # end stop
                self.stopsDict[trip[endStopIndex]] = number
                self.stopsPosDict[number] = [float(trip[7]),float(trip[8])]
                number += 1   
        
class TripsData():
    def __init__(self):
        self.rawODTrips = []
        self.FPODTrips = [] # Frequent pattern OD trips
        self.groupTrips = []

    def CreateRawODTrips(self, path, stopsDict, startStopIndex, endStopIndex):
        '''
        Transfer raw OD to numbers based on StopsDict 
        eg: trip1 : A,B,C -> 1,2,3
        '''
        trips = csv.reader(open(path,'r',encoding='utf-8'))
        isFirst = True
        foreID = ""
        curID = ""
        tripsTmp = []
        for trip in trips:
            if(isFirst):
                foreID = trip[0]
                curID = trip[0]
                tripsTmp.append(stopsDict[trip[startStopIndex]])
                tripsTmp.append(stopsDict[trip[endStopIndex]])
                isFirst = False
            else:
                curID = trip[0]
                if(curID == foreID):
                    tripsTmp.append(stopsDict[trip[startStopIndex]])
                    tripsTmp.append(stopsDict[trip[endStopIndex]])
                else: 
                    self.rawODTrips.append(copy.deepcopy(tripsTmp)) # notice deep copy
                    tripsTmp.clear()
                    foreID = curID
                    tripsTmp.append(stopsDict[trip[startStopIndex]])
                    tripsTmp.append(stopsDict[trip[endStopIndex]])
        # add the last passenger  
        self.rawODTrips.append(copy.deepcopy(tripsTmp)) # notice the deep copy
        tripsTmp.clear()

    def CreateFPODTrips(self, tripPath, FPPath, stopsDict, startStopIndex, endStopIndex):
        totalCnt = 0
        transCnt = 0
        fps = tools.LoadFP(FPPath) # frequent patterns
        trips = csv.reader(open(tripPath,'r',encoding='utf-8'))
        isFirst = True
        foreID = ""
        curID = ""
        tripsTmp = []
        for trip in trips:
            if(isFirst):
                foreID = trip[0]
                curID = trip[0]
                tripsTmp.append(stopsDict[trip[startStopIndex]])
                tripsTmp.append(stopsDict[trip[endStopIndex]])
                isFirst = False
            else:
                curID = trip[0]
                if(curID == foreID):
                    tripsTmp.append(stopsDict[trip[startStopIndex]])
                    tripsTmp.append(stopsDict[trip[endStopIndex]])
                else: 
                    tripsTmp,ifTrans = tools.TransferToFPTrip(tripsTmp, fps)
                    self.FPODTrips.append(copy.deepcopy(tripsTmp)) # notice deep copy
                    tripsTmp.clear()
                    foreID = curID
                    tripsTmp.append(stopsDict[trip[startStopIndex]])
                    tripsTmp.append(stopsDict[trip[endStopIndex]])
                    totalCnt += 1
                    if(ifTrans):
                        transCnt += 1
        # add the last passenger  
        tripsTmp,ifTrans = tools.TransferToFPTrip(tripsTmp, fps)
        self.FPODTrips.append(copy.deepcopy(tripsTmp)) # notice the deep copy
        tripsTmp.clear()
        totalCnt += 1
        if(ifTrans):
            transCnt += 1
        print("Total passengers:"+str(totalCnt)+".Frequnent pattern passangers:"+str(transCnt))

    def IfSimiliarTrip(self, trip, qtrip, posLimit, durLimit):
        if(abs(float(qtrip[3])-float(trip[3])) < posLimit and abs(float(qtrip[4])-float(trip[4])) < posLimit and abs(float(qtrip[7])-float(trip[7])) < posLimit and abs(float(qtrip[8])-float(trip[8])) < posLimit and abs(tools.StrTimeToFloatTime(qtrip[1]) - tools.StrTimeToFloatTime(trip[1])) < durLimit and abs(tools.StrTimeToFloatTime(qtrip[5]) - tools.StrTimeToFloatTime(trip[5])) < durLimit):
            return True
        else:
            return False


    def CreateGroupTrips(self, singlePath, groupPath, stopsDict, posLimit, durLimit):
        '''
        singlePath: single passenger
        groupPath: query the group
        '''
        trips = csv.reader(open(singlePath,'r',encoding='utf-8'))
        isFirst = True
        foreID = ""
        curID = ""
        tripsTmp = []
        for trip in trips:
            if(isFirst):
                foreID = trip[0]
                curID = trip[0]
                qtrips = csv.reader(open(groupPath,'r',encoding='utf-8'))
                for qtrip in qtrips:
                    if(self.IfSimiliarTrip(trip,qtrip,posLimit,durLimit)):
                        tripsTmp.append(stopsDict[qtrip[2]])
                        tripsTmp.append(stopsDict[qtrip[6]])
                isFirst = False
            else:
                curID = trip[0]
                if(curID == foreID):
                    qtrips = csv.reader(open(groupPath,'r',encoding='utf-8'))
                    for qtrip in qtrips:
                        if(self.IfSimiliarTrip(trip,qtrip,posLimit,durLimit)):
                            tripsTmp.append(stopsDict[qtrip[2]])
                            tripsTmp.append(stopsDict[qtrip[6]])
                else: 
                    self.groupTrips.append(copy.deepcopy(tripsTmp)) # notice deep copy
                    tripsTmp.clear()
                    foreID = curID
                    qtrips = csv.reader(open(groupPath,'r',encoding='utf-8'))
                    for qtrip in qtrips:
                        if(self.IfSimiliarTrip(trip,qtrip,posLimit,durLimit)):
                            tripsTmp.append(stopsDict[qtrip[2]])
                            tripsTmp.append(stopsDict[qtrip[6]])
        # add the last passenger  
        self.groupTrips.append(copy.deepcopy(tripsTmp)) # notice the deep copy
        tripsTmp.clear()



class Grid():
    '''
    Original grid class
    '''
    def __init__(self, llLng, llLat, urLng, urLat):
        # lower left's lng and lat, upper right's lng and lat
        self.llLng = llLng
        self.llLat = llLat
        self.urLng = urLng
        self.urLat = urLat
        self.width = self.urLng - self.llLng
        self.height = self.urLat - self.llLat
        # trips occur in this grid. A list of (lng,lat)
        self.tripsOccur = []
        self.tripsCnt = 0

    def InitGridOccurFromFile(self, path):
        trips = csv.reader(open(path,'r',encoding='utf-8'))
        for trip in trips:
            OLng = float(trip[3])
            OLat = float(trip[4])
            DLng = float(trip[7])
            DLat = float(trip[8])
            self.tripsOccur.append(copy.deepcopy([OLng,OLat]))
            self.tripsOccur.append(copy.deepcopy([DLng,DLat]))
        self.tripsCnt = len(self.tripsOccur)
        print('Init '+str(self.tripsCnt)+' trip occurs.')


class FrequentGrid():
    def __init__(self, maxOccur=100000, mGW=2500, mGH=2500):
        self.maxOccur = maxOccur
        self.minGW = mGW # min grid width
        self.minGH = mGH # min grid height

        self.fGrids = [] # result grids :[lllng,lllat,urlng,urlat,tripOccurCnt]

    def CalOccur(self, childGrid, fatherGrid):
        for tripOccur in fatherGrid.tripsOccur:
            if(tripOccur[0]<childGrid.urLng and tripOccur[0]>childGrid.llLng and tripOccur[1]<childGrid.urLat and tripOccur[1]>childGrid.llLat):
                childGrid.tripsOccur.append(tripOccur)
        childGrid.tripsCnt = len(childGrid.tripsOccur)
        return childGrid

    def SplitGrid(self, grid):
        '''
        Split a grid to 4 pieces
        paras:
            grid: The instance of class Grid
        '''
        # center lng and lat of the grid
        cLng = 0.5*(grid.llLng + grid.urLng)
        cLat = 0.5*(grid.llLat + grid.urLat)

        ulGrid = Grid(grid.llLng, cLat, cLng, grid.urLat) # upper left
        urGrid = Grid(cLng, cLat, grid.urLng, grid.urLat) # upper right
        llGrid = Grid(grid.llLng, grid.llLat, cLng, cLat) # lower left
        lrGrid = Grid(cLng, grid.llLat, grid.urLng, cLat) # lower right

        quaGrids = [ulGrid, urGrid, llGrid, lrGrid]

        for quaGrid in quaGrids:
            quaGrid = self.CalOccur(quaGrid,grid)

        return quaGrids

    
    def FilterByQuatree(self, grids):
        '''
        paras:
            grids: The instances of class Grid
        '''
        for grid in grids:
            if(grid.tripsCnt > self.maxOccur and grid.width > self.minGW and grid.height > self.minGH):
                splitedGrids = self.SplitGrid(grid)
                self.FilterByQuatree(splitedGrids)
            else:
                self.fGrids.append(copy.deepcopy([grid.llLng,grid.llLat,grid.urLng,grid.urLat,grid.tripsCnt]))

    def FGridsToFile(self, path):
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            for fgrid in self.fGrids:
                w.writerow(fgrid)
            print('FGrids to path:'+path)

    def InitFGridsFromFile(self, path):
        fgrids = csv.reader(open(path,'r',encoding='utf-8'))
        for fgrid in fgrids:
            fgrid = list(map(eval,fgrid))
            self.fGrids.append(fgrid)
        print('Init frequent grids from:'+path)



