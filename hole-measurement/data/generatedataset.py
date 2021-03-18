import numpy as np
import math
import random

# structure
# [datawidth] columns of height data -100 for null
# hole present (0=no or 1=yes)
# hole size (0 for not present, 1 for one pixel wide hole)


# build examples
dataWidth = 6
minSlope = -3
maxSlope = 3
nullValue = -100

# build varying slopes
def generateSlopes():
    dataset = None
    i=0
    slope = np.empty(maxSlope-minSlope+1)
    for currentSlope in range (minSlope, maxSlope+1):
        slope[i] = currentSlope
        lastValue = 0
        if (currentSlope == 0):
            newdata = np.zeros(dataWidth)
        else:
            newdata = np.arange(0, dataWidth*currentSlope, currentSlope)
        if (dataset is None):
            dataset = newdata
        else:
            dataset = np.vstack((dataset, newdata))
        i+=1
    return dataset    

def knockOutHoles(dataset):
    ds1 = dataset.copy()
    ds2 = dataset.copy()
    ds3 = dataset.copy()
    for row in ds1:
        row[2] = nullValue
    for row in ds2:
        row[3] = nullValue
    for row in ds3:
        row[2:4] = nullValue
    return np.concatenate((ds1, ds2, ds3))

def trimLeftEdge(dataset):
    for row in dataset:
        row[0] = nullValue
    return dataset

def trimRightEdge(dataset):
    for row in dataset:
        row[-1] = nullValue
    return dataset

def addNoise(dataset):
    for row in dataset:
        for val in row:
            if val != nullValue:
                val += random.random() - 0.5 

    return dataset

def holeData(row):
    """Finds hole data from an input row.
    
    Takes in an array of height values
    Returns holePresent, holeSizeProjected, holeSize, holePosition"""
    lastValue = nullValue
    steppedDown = False
    holeSizeProjected = 0
    holeSize = 0
    holePosition = 0
    leftEdgeHeight = 0
    leftEdgePosition = 0
    for value in row:
        # check if we stepped down to null
        if (steppedDown):
            holeSizeProjected = holeSizeProjected+1

            # check for step up after having stepped down
            if (lastValue == nullValue and value != nullValue):
                # calculate holesize
                holeSize = math.sqrt((leftEdgeHeight-value)**2 + holeSizeProjected**2)
                return True, holeSizeProjected, holeSize, 0
                steppedDown = False
        else:
            # check for step down to null
            if (lastValue != nullValue and value == nullValue):
                steppedDown = True
                leftEdgeHeight = lastValue

        lastValue = value

    # no hole found
    return False, 0, 0, 0

def buildLables(unlabled):
    nRows = np.shape(unlabled)[0]
    lables = np.zeros((nRows,4))
    i = 0
    for row in unlabled:
        h = holeData(row)
        lables[i][0] = h[0]
        lables[i][1] = h[1]
        lables[i][2] = h[2]
        lables[i][3] = h[3]
        i = i+1
    return lables

def buildNulls(unlabled):
    nRows = np.shape(unlabled)[0]
    lables = np.zeros((nRows,4))
    i = 0
    for row in unlabled:
        h = holeData(row)
        lables[i][0] = h[0]
        lables[i][1] = h[1]
        lables[i][2] = h[2]
        lables[i][3] = h[3]
        i = i+1
    return lables

ds = generateSlopes()
ds = np.concatenate((ds, knockOutHoles(ds.copy())))
ds = np.concatenate((ds, trimLeftEdge(ds.copy())))
ds = np.concatenate((ds, trimRightEdge(ds.copy())))
ds = np.concatenate((ds, addNoise(ds.copy())))

np.shape(ds)

lables = buildLables(ds)
lds = np.hstack((ds, lables))
# format output
headerStr = ""
for i in range(dataWidth):
    headerStr = headerStr + "v" + str(i) + ", "
headerStr = headerStr + " holePresent, holeWidthProj, holeWidth, holePos"
print("HeaderStr: "+headerStr)
np.savetxt("holes6.csv", lds, delimiter=",", header=headerStr)
print(lds)
