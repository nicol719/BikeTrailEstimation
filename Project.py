#----------------------------
# Bike Trail Estimation
# Created by Grant Nicol
# November 2020
#----------------------------


#----------------------------
# Imports
#----------------------------
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
import math as math
import csv as csv
from threading import Thread
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.neighbors import KDTree



#=============================================================================
#                                 Functions
#=============================================================================

#------------------------------------------
# extractActivity()
# Removes a single activity from the data
# set and does a bit of pre-processing
# manipulation.
# 
#  Velocities are caluculated for Kalman
# filter purposes which are currently
# not applied in this code.
#------------------------------------------
def extractActivity(dataFile, activityName):
    activityZero = dataFile[dataFile['file_name'] == activityName]


    x_values = np.zeros(activityZero.shape[0])
    vx_values = np.zeros(activityZero.shape[0])
    y_values = np.zeros(activityZero.shape[0])
    vy_values = np.zeros(activityZero.shape[0])
    z_values = np.zeros(activityZero.shape[0])
    vz_values = np.zeros(activityZero.shape[0])
    timePoints = np.zeros(activityZero.shape[0])
    
    trim = 0 #Trim data if points are too far apart
    for i in range(activityZero.shape[0]):
        if trim == 0:
            activityPoint = activityZero.iloc[i]
            if i != 0: #Calculate velocities
                activityPointPrev = activityZero.iloc[i-1]
                deltaT = activityPoint.loc['time']/1000 \
                    - activityPointPrev.loc['time']/1000
                deltaX = activityPoint.loc['X'] - activityPointPrev.loc['X']
                deltaY = activityPoint.loc['Y'] - activityPointPrev.loc['Y']
                deltaZ = activityPoint.loc['Z'] - activityPointPrev.loc['Z']
            else:
                timeMin = activityPoint.loc['time']/1000
        
            if int(activityPoint.loc['time']/1000 - timeMin) \
                    - timePoints[i -1] > 60:
                print("Activity trimmed")
                trim = 1
            else:
                
                timePoints[i] = int(activityPoint.loc['time']/1000 - timeMin)
                timeMax = activityPoint.loc['time']/1000
                x_values[i] = activityPoint.loc['X'] - 473940
                y_values[i] = activityPoint.loc['Y'] - 5481556
                z_values[i] = activityPoint.loc['Z']
                if i != 0:   
                    vx_values[i] = deltaX/deltaT
                    vy_values[i] = deltaY/deltaT
                    vz_values[i] = deltaZ/deltaT + 0.0001
    
    x_values = np.trim_zeros(x_values,'b')
    y_values = np.trim_zeros(y_values, 'b')
    z_values = np.trim_zeros(z_values, 'b')
    vx_values = np.trim_zeros(vx_values, 'b')
    vy_values = np.trim_zeros(vy_values, 'b')
    vz_values = np.trim_zeros(vz_values, 'b')
    timePoints = np.trim_zeros(timePoints, 'b')
    Xm = np.c_[x_values, vx_values, y_values, vy_values, z_values, vz_values]
    timeRange = timeMax - timeMin
    return Xm, timeRange, timePoints


def threadMeanSquareError(base, track, results, i):
    error = meanSquareError(base, track)
    x_offset, y_offset, newerror = allignPoints(base, 
                                                track, 
                                                error, 
                                                0, 0, 0.1)
    results[i] = [x_offset, y_offset]
#------------------------------------------
# meanSquareError()
# Calculates the mean square error between
# two tracks.
#------------------------------------------
def meanSquareError(baseTrack, allignTrack, baseTrackTree):
    error = 0
    #print(np.asarray(allignTrack))
    
    for i in range(len(allignTrack)):
        point = allignTrack[i]
        #print(findClosestPoint3(point, Tree)[0][0])
        dist, ind = findClosestPoint3(point, baseTrackTree)
        #closestPoint = baseTrack[index]
        #print(closestPoint)
        error = error + dist**2
    error = error/len(allignTrack)
    #print(error)
    return error

#------------------------------------------
# allignPoints()
# Performs a recursive coordinte search in order to
# find the offset that minimizes the mean
# square error between two tracks.
#------------------------------------------    
def allignPoints(baseTrackTree,
                 baseTrack, 
                 allignTrack, 
                 error, 
                 x_allignment = 0, 
                 y_allignment = 0, 
                 stepSize = 1):
        
    XplusTrack = offset_Track(allignTrack,
                              x_allignment + stepSize,
                              y_allignment)
    errorXplus = meanSquareError(baseTrack, XplusTrack, baseTrackTree)    
    if errorXplus < error:
        final_x_allign, \
        final_y_allign, \
        newError = allignPoints(baseTrackTree,
                                baseTrack,
                                allignTrack,
                                errorXplus,
                                x_allignment + stepSize,
                                y_allignment,
                                stepSize)
    else:
        XminusTrack = offset_Track(allignTrack,
                                   x_allignment - stepSize,
                                   y_allignment)
        errorXminus = meanSquareError(baseTrack, XminusTrack, baseTrackTree)
        if errorXminus < error:
            final_x_allign, \
            final_y_allign, \
            newError = allignPoints(baseTrackTree,
                                    baseTrack,
                                    allignTrack,
                                    errorXminus,
                                    x_allignment - stepSize,
                                    y_allignment,
                                    stepSize)
        else:
            YplusTrack = offset_Track(allignTrack,
                                      x_allignment,
                                      y_allignment + stepSize)
            errorYplus = meanSquareError(baseTrack, YplusTrack, baseTrackTree)
            if errorYplus < error:
                final_x_allign, \
                final_y_allign, \
                newError = allignPoints(baseTrackTree,
                                        baseTrack,
                                        allignTrack,
                                        errorYplus,
                                        x_allignment,
                                        y_allignment + stepSize,
                                        stepSize)
            else:
                YminusTrack = offset_Track(allignTrack,
                                           x_allignment,
                                           y_allignment - stepSize)
                errorYminus = meanSquareError(baseTrack, YminusTrack, baseTrackTree)
                if errorYminus < error:
                    final_x_allign, \
                    final_y_allign, \
                    newError = allignPoints(baseTrackTree,
                                            baseTrack,
                                            allignTrack,
                                            errorYminus,
                                            x_allignment,
                                            y_allignment - stepSize,
                                            stepSize)
                else:
                    if stepSize > 0.2:
                        final_x_allign, \
                        final_y_allign, \
                        newError = allignPoints(baseTrackTree,
                                                baseTrack,
                                                allignTrack,
                                                error,
                                                x_allignment,
                                                y_allignment,
                                                0.1)
                    else:
                        final_x_allign = x_allignment
                        final_y_allign = y_allignment
                        newError = error  
    
    return final_x_allign, final_y_allign, newError

#------------------------------------------
# offset_Track()
# Offsets a given track by a given x and y
#------------------------------------------
def offset_Track(Track, x_offset, y_offset):
    newTrack = []
    for i in Track:
        newTrack.append([i[0] + x_offset, i[1] + y_offset])
    return newTrack

#------------------------------------------
# addPoints()
# Adds additional inferred points to a
# track.
#------------------------------------------
def addPoints(Xm, timeRange, timePoints):
    result = []
    i = 0
    timeRange = timeRange * 10
    timePoints = timePoints * 10
    for timeStep in range(int(timeRange)):
        dataExists = np.where(timePoints == timeStep)
        if dataExists[0].size == 1:
            result.append([Xm[i,0], Xm[i,2]])
            i = i + 1
            
        elif(timeStep != timeRange):
                
                timeDistance = timePoints[i] - timeStep
                
                newX = -(result[-1][0] - Xm[i,0])/(timeDistance) \
                       + result[-1][0]
                newY = -(result[-1][1] - Xm[i,2])/(timeDistance) \
                       + result[-1][1]
                
                result.append([newX,newY])

    return result

#------------------------------------------
# findClosestPoint()
# Takes a given point and searches a given
# track for its closest point.
#------------------------------------------
def findClosestPoint(point, Track):
    distance = (point[0] - Track[0][0])**2 + (point[1] - Track[0][1])**2
    for i in range(len(Track)):
        trackPoint = Track[i]
        if (((point[0] - trackPoint[0])**2 \
             + (point[1] - trackPoint[1])**2 <= distance)):
            newPoint = trackPoint
            distance = (point[0] - trackPoint[0])**2 \
            + (point[1] - trackPoint[1])**2
    return newPoint

#------------------------------------------
# findClosestPoint2()
# Uses numpy to optimize
# Takes a given point and searches a given
# track for its closest point.
#------------------------------------------
def findClosestPoint2(point, Track):
    Track = np.asarray(Track)
    dist_2 = np.sum((Track - point)**2, axis=1)
    return np.argmin(dist_2)

#------------------------------------------
# findClosestPoint3()
# Uses KD tree
# Takes a given point and searches a given
# track for its closest point.
#------------------------------------------
def findClosestPoint3(point, Tree):
    dist, ind = Tree.query(point, k=1)
    #print(dist)
    return dist[0][0], ind[0][0]

#------------------------------------------
# trailAverage()
# Averages a given track with a given
# running weighted average.
#------------------------------------------
def trailAverage(average, weight, nextTrack, x_offset, y_offset):
    if weight == 0:
        return nextTrack, weight + 1
    result = []
    Tree = KDTree(np.asarray(nextTrack), metric = 'euclidean')
    for i in range(len(average)):
        point = average[i]
        dist, ind = findClosestPoint3(point, Tree)
        
        #closestPoint = findClosestPoint(point,
        #                                offset_Track(nextTrack, 
        #                                             x_offset, 
        #                                             y_offset))
        #newX = (point[0]*weight + closestPoint[0] - x_offset)/(weight + 1)
       # newY = (point[1]*weight + closestPoint[1] - y_offset)/(weight + 1)
        newX = (point[0]*weight + nextTrack[ind][0] - x_offset)/(weight + 1)
        newY = (point[1]*weight + nextTrack[ind][1] - y_offset)/(weight + 1)
        result.append([newX,newY])
    return result, weight + 1

#------------------------------------------
# distributePoints()
# Distributes points on a given track
# such that there are no large gaps or
# clusters
#------------------------------------------
def distributePoints(Track):
    newTrack = []
    pointsAdded = 0
    pointsRemoved = 0
    for i in range(len(Track)):
        if i == 0:
            newTrack.append(Track[i])
        else:
            distance = pointDistance(Track[i], newTrack[-1])
            if distance < 0.1:
                averagePoint = [(Track[i][0] + newTrack[-1][0])/2, 
                                (Track[i][1] + newTrack[-1][1])/2]
                del newTrack[-1]
                pointsRemoved = pointsRemoved +1
                newTrack.append(averagePoint)
            else:
                numNewPoints = math.floor((distance/1)) #-1
                newTrack.append(Track[i])
                if numNewPoints != 0:
                    #print("-------------")
                    #print(Track[i][0], newTrack[-2][0])
                    for j in range(numNewPoints):
                        x = (((Track[i][0] - newTrack[-2-j][0])/numNewPoints)*(j+1)
                             + newTrack[-2-j][0])
                        #print(x)
                        y = (((Track[i][1] - newTrack[-2-j][1])/numNewPoints)*(j+1) 
                             + newTrack[-2-j][1])
                        newTrack.append([x,y])
                        #print(x)
                        pointsAdded = 1 + pointsAdded
                
    if pointsAdded >= 1:
        print("Points Added: ", pointsAdded, ", Points Removed:", pointsRemoved)
    return newTrack

#------------------------------------------
# pointDistance()
# Calculates distance between two points.
#------------------------------------------
def pointDistance(Point1, Point2):
    return math.sqrt((Point1[0] - Point2[0])**2 + (Point1[1] - Point2[1])**2)


#=============================================================================
#                                 Script
#=============================================================================


#------------------------------------------
# Load Data
#------------------------------------------
dataFile = pandas.read_csv('LongPoints.csv', sep=',')
activityNames = dataFile.file_name.unique()



# 50 total activities
activityNames = ['activity_4533298430.gpx', 'activity_3743532184.gpx',
                 'activity_4507635950.gpx', 'activity_4340640801.gpx',
                 'activity_3740364897.gpx', 'activity_4370207855.gpx', 
                 'activity_4969337496.gpx', 'activity_3956601084.gpx', 
                 'activity_3901396808.gpx', 'activity_4160161220.gpx', 
                 'activity_3831270241.gpx', 'activity_4035110824.gpx', 
                 'activity_4275836960.gpx', 'activity_3599697933.gpx', 
                 'activity_3593740764.gpx', 'activity_5194165580.gpx', 
                 'activity_3892996697.gpx', 'activity_5105681178.gpx', 
                 'activity_3850901310.gpx', 'activity_3904927438.gpx', 
                 'activity_4907622294.gpx', 'activity_4937512173.gpx', 
                 'activity_3884598808.gpx', 'activity_5182499482.gpx', 
                 'activity_3953245868.gpx', 'activity_3881396776.gpx', 
                 'activity_4176635569.gpx', 'activity_5252479003.gpx', 
                 'activity_3737182817.gpx', 'activity_4952361763.gpx', 
                 'activity_5356651671.gpx', 'activity_4597467771.gpx', 
                 'activity_5314037132.gpx', 'activity_4302070841.gpx', 
                 'activity_4936223553.gpx', 'activity_3756407852.gpx', 
                 'activity_5115106216.gpx', 'activity_5197086831.gpx', 
                 'activity_4665868447.gpx', 'activity_4397184048.gpx', 
                 'activity_4422305760.gpx', 'activity_4511693677.gpx', 
                 'activity_3717894439.gpx', 'activity_4663670569.gpx', 
                 'activity_3763198525.gpx', 'activity_4913132764.gpx',
                 'activity_5302748207.gpx', 'activity_3908204668.gpx',
                 'activity_4983826972.gpx', 'activity_5243255333.gpx',]

#------------------------------------------
# Iterate over all tracks in data
# calaculate a new running average
# with each track
#------------------------------------------

#Define cumulative values
weight = 0 
average = []
offsets = []
j = 0
for i in activityNames:
    
    #define vaalues which reset with each iteration
    x_offset = 0
    y_offset = 0
    error = 0
    newerror = 0
    
    print(i)
    
    #Extrct specific track from data
    Xm, timeRange, timePoints = extractActivity(dataFile,i)
    
    #Infer additional points so there is a point at every 0.1 seconds
    track = addPoints(Xm, timeRange, timePoints)
    
    
    if j >= 2: #Don't attempt to allign when there is only one track
        
        averageTree = KDTree(np.asarray(average), metric = 'euclidean')
        #Calculate mean square error between current track and running average
        error = meanSquareError(average, track, averageTree)
        #print(error)
        
        #Find x and y offset to allign current track with running average
        if error > 2:
            stepSize = 1
        else:
            stepSize = 0.1
        
        
        
        x_offset, y_offset, newerror = allignPoints(averageTree,
                                                    average, 
                                                    track, 
                                                    error, 
                                                    0, 0, stepSize) #x,y,step size
        print("Old error: ", round(error,2), 
              " New error: ", round(newerror,2), 
              " Offset: ", round(x_offset,1), round(y_offset,1))
    
    offsets.append([x_offset, y_offset])
    
    #Add current track to the running average
    average, weight = trailAverage(average, weight, track, x_offset, y_offset)
    
    #Redistribute points so that no gaps or clumps form
    average = distributePoints(average)
    
    #Plot current track
    track_x, track_y = zip(*track)
    plt.plot(track_x, track_y, 'b-', 
             label='track, mse = ' + str(round(error,2)))
    
    #Plot alligned track
    offsetTrack = offset_Track(track, x_offset, y_offset) 
    offset_x, offset_y = zip(*offsetTrack)
    plt.plot(offset_x, offset_y, 'g-', 
             label='alligned track, mse = ' + str(round(newerror,2)))
    
    #Plot running average
    average_x, average_y = zip(*average)
    plt.plot(average_x, average_y, 'r-', label='running average')
    
    #Save plot to file
    plt.title(i,fontsize=20)
    plt.xlabel('x coordinate (m)', fontsize=16)
    plt.ylabel('y coordinte (m)', fontsize=16)
    plt.legend(frameon = False)
    plt.savefig(i + '.png', dpi = 400)
    plt.show()

    #Since there are 50 tracks, each iteration is 2% of the totall    
    j = j + 2
    print(j, "% complete")

#Save average to file    
with open("Average_Results.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(average)
     wr.writerow(offsets)

#------------------------------------------
# With the final average calculated,
# find the x and y offsets for each track
# as well as the mse for each track.
#------------------------------------------
errorSet = []
for i in range(len(activityNames)):
    Xm, timeRange, timePoints = extractActivity(dataFile,activityNames[i])
    track = addPoints(Xm, timeRange, timePoints)
    error = meanSquareError(average, track, averageTree)
    print(offsets[i][0], offsets[i][1])
    x_offset, y_offset, newerror = allignPoints(averageTree,
                                                average, 
                                                track, 
                                                error, 
                                                offsets[i][0], offsets[i][1], 0.1)
    errorSet.append([x_offset, y_offset, error, newerror])
    print("Old error: ", round(error,2), 
              " New error: ", round(newerror,2), 
              " Offset: ", round(x_offset,1), round(y_offset,1))
    print(i*2,"percent complete")
 
with open("Errors_Set.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(errorSet)

#------------------------------------------
# Attempt to redistribute points along average
#------------------------------------------
distAverage = average
for i in range(len(activityNames)):
    Xm, timeRange, timePoints = extractActivity(dataFile,activityNames[i])
    track = addPoints(Xm, timeRange, timePoints)
    error = meanSquareError(average, track, averageTree)
    print(offsets[i][0], offsets[i][1])
    x_offset, y_offset, newerror = allignPoints(averageTree,
                                                average, 
                                                track, 
                                                error, 
                                                offsets[i][0], offsets[i][1], 0.1)
    distAverage, weight = trailAverage(distAverage, weight, track, x_offset, y_offset)
    distAverage = distributePoints(distAverage)
    
    distAverage_x, distAverage_y = zip(*distAverage)
    plt.plot(distAverage_x, distAverage_y, 'r-', label='running average')
    
    #Save plot to file
    plt.title(i,fontsize=20)
    plt.xlabel('x coordinate (m)', fontsize=16)
    plt.ylabel('y coordinte (m)', fontsize=16)
    plt.legend(frameon = False)
    #plt.savefig(i + '.png', dpi = 400)
    plt.show()
 
with open("Dist_Average.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(distAverage)

print('Complete')