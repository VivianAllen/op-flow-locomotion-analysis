import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import csv

#### USER INPUTS ####

# filepath to folder with vid files to process NB make sure you leave the 'r' start of the string in their,
# that changes it from an interpreted string (in which '\' is an escape character) to a raw string, which can be anything.
# Code will rummage through all subfolders in this folder looking for video files (with file extension listed in vidType
# parameter) that are in folders containing the keyword in the subFolderName parameter
filepath_list = [r'G:\DDNC03\STRAIGHT RUNWAY WITH FP\2017_10_04']

# what types of video are they?
vidType = '.avi'

# how far away in m are calibration pnts? i.e. if its the forceplate, whats the width or length or
# whatever (dependent on plate orientation obvs)
ptRealDist = 0.2

# set frame rate (Hz)
frameRate = 60

# set movement threshold in m/s
minFlow = 0.02

## OUTER LOOP ##
for filepath in filepath_list:


    # get calibration folders (NB presumed your calibration files are .csv with two columns and three rows, first row
    # is X Y header, second row is pt1 X Y, third is pt2 X Y. Pts themselves are closest and furthest end of forceplate
    # relative to 'default' direction of travel
    calibFolders = [os.path.join(root, name)
                 for root, dirs, files in os.walk(filepath)
                 for name in dirs
                 if 'optical' in root.lower() and 'calibration' in root.lower() \
                    and 'flow' in root.lower()]

    # parse into subfolder names
    subFolderNames = [x.split("\\")[-1] for x in calibFolders]

    # get calibration pnts as dictionary of lists
    calibPnts1 = {}
    calibPnts2 = {}
    for calibFolder in calibFolders:
        calibFile = [item for item in os.listdir(calibFolder) if item.endswith(".csv")]
        calibFilePath = calibFolder + "\\" + str(calibFile[0])
        with open(calibFilePath) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            pnts = []
            rowCount = 1
            for row in readCSV:
                if rowCount == 2:
                    calibPnts1[calibFolder.split("\\")[-1]] = [int(round(float(row[-2]))), int(round(float(row[-1])))]
                elif rowCount == 3:
                    calibPnts2[calibFolder.split("\\")[-1]] = [int(round(float(row[-2]))), int(round(float(row[-1])))]
                rowCount += 1

    # main loop
    for subFolderName in subFolderNames:

        # get calibration points
        pt1 = calibPnts1[subFolderName]
        pt2 = calibPnts2[subFolderName]

        ## get rotation matrix to level image stuff ##

        # convert those points into vector
        ptVec = np.array(pt2) - np.array(pt1)

        # get vector magnitude
        ptVecMag = np.linalg.norm(ptVec)

        # then unit vector
        ptVecN = ptVec / ptVecMag

        # then 2D rotation matrix
        ptRotMat = np.transpose(np.array([ptVecN, [ptVecN[1]*-1, ptVecN[0]]]))

        ## get length calibration stuff ##

        # set length calibration value (m/px)
        lenCalib =  ptRealDist / ptVecMag

        #### FIND ALL VID FILES IN APPROPRIATE FOLDERS ####

        vidList = [os.path.join(root, name)
                     for root, dirs, files in os.walk(filepath)
                     for name in files
                     if name.endswith(vidType) and subFolderName.lower() in root.lower()]


        # loop through those files
        for vidFile in vidList:


            print("CURRENTLY PROCESSING " + vidFile)

            #### MAIN ROUTINE ####

            # open video object
            cap = cv2.VideoCapture(vidFile)

            # read in first frame
            ret, frame1 = cap.read()

            # set previous frame as first frame
            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

            # get number of frames
            nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # initialise average horiztonal velocity storage
            meanHorzVPos = np.zeros([nFrames, 1])
            meanHorzVNeg = np.zeros([nFrames, 1])

            # main loop - nb, GoPro are cheap fuckers and have spoofed 60hz as 30hz with duplicated frames
            # so we want to skip over every other frame...
            for i1 in range(0,nFrames-1):

                # if even number (we'll include zero in that list...) then do, if not take previous and duplicate
                # like GoPro do. NO we are ALREADY on frame 0, so loop i1 = 0 will actually process frame 1, which will
                # be a duplicate of frame 0. Hence, skip to ODD number frames only

                # read in next frame
                ret, frame2 = cap.read()

                if ret == True: # proceed if there are still frames to go

                    if not i1 % 2 == 0: # if odd-numbered frame, go

                        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                        # calculate flow for each pixel between two frames, output is numpy array
                        # of dimensions imagePixelHeight, imagePixelWidth, 2 that contains the
                        # pixel shift in x and y between the two frames (hence third dimension is 2)
                        # we want the average of that.
                        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                        # get flow as x y columns, invert y to make it cartesian not image coordinates then rotate
                        # so that point vector clicked on from imageJ is X axis

                        # unroll flow into 2D coordinate array. First swap axes 1 and 3 so that it becomes
                        # more like a 2D array
                        flowReord = np.swapaxes(flow,1,2)

                        # prep storage
                        flowCoords = np.zeros([flow.shape[0] * flow.shape[1], 2])

                        # prep write indices
                        writeIndex = [0, flowReord.shape[0]]

                        # loop and populate as XY pairs
                        for i2 in range(flowReord.shape[2]):

                            # populate
                            flowCoords[writeIndex[0]:writeIndex[1],:] = flowReord[...,i2]

                            # update write index
                            writeIndex[0] = writeIndex[1]
                            writeIndex[1] = writeIndex[1] + flowReord.shape[0]

                        # rotate so that chose vector is horizontal
                        flowHorzVert = np.matmul(flowCoords, ptRotMat)

                        # discard vertical flow
                        flowHorz = flowHorzVert[:,0]

                        # convert by multiplying by
                        # metres per pixel / seconds per frame, but as seconds per frame = 1/frames per second
                        # can MATHS it into metres per pixel * frames per second
                        flowHorzms = flowHorz * lenCalib * frameRate

                        # threshold positive
                        flowHorzmsThreshPos = flowHorzms[flowHorzms > minFlow]

                        # threshold negative
                        flowHorzmsThreshNeg = flowHorzms[flowHorzms < minFlow * -1]

                        # if any values above threshold, append to mean horizontal flow, otherwise subsitute zero
                        if flowHorzmsThreshPos.any():
                            meanHorzVPos[i1+1] = np.mean(flowHorzmsThreshPos)
                        else:
                                meanHorzVPos[i1+1] = 0

                        if flowHorzmsThreshNeg.any():
                            meanHorzVNeg[i1+1] = np.mean(flowHorzmsThreshNeg)
                        else:
                                meanHorzVNeg[i1+1] = 0

                        # set previous frame to current frame
                        prvs = next

                        # report progress
                        #print(i1)

                    else: # if even numbereed frame, duplicate previous

                        meanHorzVPos[i1+1] = meanHorzVPos[i1]
                        meanHorzVNeg[i1+1]= meanHorzVNeg[i1]

                else:

                    break



            # define get moving average function
            def getMovingAv(inData, window):
                movAvData = np.zeros([len(inData),1])
                halfWin = round(window / 2 )
                for i1 in range(halfWin, len(inData) - halfWin):
                    movAvData[i1] = np.mean(inData[i1-halfWin:i1+halfWin])

                 # pad back and front of moving average
                movAvData[:halfWin] = movAvData[halfWin]
                movAvData[-halfWin:] = movAvData[-halfWin-1]

                return movAvData



            # get moving averages, use framerate (i.e. 1 second) window
            horzVsMovAvPos = getMovingAv(meanHorzVPos, frameRate)
            horzVsMovAvNeg = getMovingAv(meanHorzVNeg, frameRate)

            # save csvs
            np.savetxt(re.sub(vidFile[-4:], 'HorzVsPos.csv', vidFile), meanHorzVPos, delimiter = ',')
            np.savetxt(re.sub(vidFile[-4:], 'HorzVsMovAvPos.csv', vidFile), horzVsMovAvPos, delimiter = ',')
            np.savetxt(re.sub(vidFile[-4:], 'HorzVsNeg.csv', vidFile), meanHorzVNeg, delimiter = ',')
            np.savetxt(re.sub(vidFile[-4:], 'HorzVsMovAvNeg.csv', vidFile), horzVsMovAvNeg, delimiter = ',')


            # plot
            plt.figure(figsize = (16, 9))
            plt.plot(meanHorzVPos, 'k--', label = 'Positive V')
            plt.plot(horzVsMovAvPos, 'k', label = 'Moving Average Positive V (Win = 1s)')
            plt.plot(meanHorzVNeg, 'r--', label = 'Negative V')
            plt.plot(horzVsMovAvNeg, 'r', label = 'Moving Average Negative V (Win = 1s)')
            plt.legend(loc='upper left')
            plt.xlabel('Frame Number')
            yLabelString = 'Mean Horizontal Speed in m/s, threshold = ' + str(minFlow)
            plt.ylabel(yLabelString)

            # tick every ten seconds
            plt.xticks(range(0, nFrames, 10),rotation='vertical')

            # set ten second grid
            plt.grid(axis = 'x')

            # get figure name and save, substitute mp4 file extension (whatever case it's in) with .pdf
            figName = re.sub(vidFile[-4:], 'HorzVs.pdf', vidFile)
            plt.savefig(figName)

            # show fig (optional)
            #plt.show()

            # close fig
            plt.close()

            cap.release()
            cv2.destroyAllWindows()

    # report progress
    print("ALL VIDEOS PROCESSED")
