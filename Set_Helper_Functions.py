# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:01:09 2021

@author: Peter Reynolds
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

from skimage.transform import rotate, resize
from skimage.util import crop
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import convex_hull_image, binary_erosion

import matplotlib.pyplot as plt
from numpy import amax
from scipy import ndimage as ndi

import tensorflow
from tensorflow.keras.models import load_model


import logging
tensorflow.get_logger().setLevel(logging.ERROR)

import math

#--- DEFINE GLOBAL VARIABLES ACCESSIBLE TO ALL FUNCTIONS ---

autoCrop = True # whether to crop out border space from the image of cards on the table

autoDimensions = True # whether to let computer determine how many cards are on the table
# otherwise, computer will use these set dimensions for cards on the table
overrideNumRows = 4
overrideNumColumns = 3

cannySigma = 2.1 # used to find contours of cards within image.  2.1 seems to work well

# Do not change! for machine learning model conventions; otherwise, model will not work properly
xDimensionsofPhotos = 135
yDimensionsofPhotos = 180
colorDepth = 3 # for RGB photos

# threshold values for determining transition indices for autoCrop
cardNumberThresholdX = 0.3
cardNumberThresholdY = 0.2

downsizeConstant = 5  # factor to downsize input image only if autoDimensions is True

numAttributes = 4 # input number of card attributes (four because: color, shape, shade, number)

cardConfidences = [] # where indices correspond to card #
arrayofRectanglePoints = [] # for storing corner coordinates for each card for drawing bounding boxes

# load attribute predictors
print('loading machine learning models...')
colorModel = load_model('<path to CNN color model>.h5')
shadeModel = load_model('<path to CNN shade model>.h5')
shapeModel = load_model('<path to CNN shape model>.h5')
numberModel = load_model('<path to CNN number model>.h5')

def set_res(cap, x, y):
    '''sets resolution of cv2 camera'''
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))

def prepareImage(img):
    '''ensures that image conforms to standard float32 representation in range [0,1]'''
    if np.average(img)>=1:
        img = img/255
    if img.dtype!='float32':
        img = img.astype('float32')
    return img

def analyzePhoto(myFrame):
    '''
    1. predicts attributes of all the cards in the `myFrame` input image using pre-trained CNN models
    2. searches the card predictions for sets
    3. produces a heads-up display showing the found sets
    '''
    myImage = cv2.cvtColor(myFrame, cv2.COLOR_BGR2RGB) # convert image to rgb because cv2 reads in image as bgr (channels reversed)
    
    prepareImage(myImage) # ensure that image is in the correct data format
    
    grayScale = rgb2gray(myImage)

    fillCards = ndi.binary_fill_holes(canny(grayScale, sigma=cannySigma))  # find canny contours of cards and fill closed card shapes with white pixels

    label_objects, nb_labels = ndi.label(fillCards)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 30
    mask_sizes[0] = 0
    card_candidates = mask_sizes[label_objects] # locical-based indexing to sort out card candidates over the size threshold
          
    eroded = binary_erosion(card_candidates) # erode image to eliminate small white spots that might confuse the convex hull command below
    convexHull = convex_hull_image(eroded) # enclose all white pixels in one white, convex blob
    

    # downsize the eroded image to reduce loop times in the future
    erodedResized = resize(eroded, ((math.floor((convexHull.shape[0])/downsizeConstant)), (math.floor((convexHull.shape[1])/downsizeConstant))), anti_aliasing=False)
    
    if autoDimensions == True:

        #### finding number of columns... ####
        lastPixelAverage = 0
        crossedThresholdGoingDownX = 0
        crossedThresholdGoingUpX = 0
        
        for p in range (1, erodedResized.shape[0]):
            # assuming empty table around the perimeter of the card array,
            currentPixelAverage = np.mean(erodedResized[p]) # get average pixel value of current image column
        
            if lastPixelAverage < cardNumberThresholdX and currentPixelAverage >= cardNumberThresholdX:
                crossedThresholdGoingDownX += 1 # transition from white to black (signifies end of card column)
            
            if currentPixelAverage > cardNumberThresholdX and currentPixelAverage <= cardNumberThresholdX:
                crossedThresholdGoingUpX += 1 # transition from black to white (signifies beginning of new card column)
    
            lastPixelAverage = currentPixelAverage # store previous column pixel averages for comparison in next loop iteration
            
        numColumns = crossedThresholdGoingDownX + crossedThresholdGoingUpX
        print('number of Columns:', numColumns)
        if numColumns<1:
            sys.exit('WARNING: Program terminated because no cards are visible')

        # rotate image for easier access to y-axis pixels
        Rotated = rotate(erodedResized, 90, resize=True, preserve_range=False)


        #### finding number of rows... ####
        crossedThresholdGoingDownY = 0
        crossedThresholdGoingUpY = 0
        lastPixelAverage = 0
        currentPixelAverage = 0
        
        for q in range (1, Rotated.shape[0]):
            # assuming empty table around the perimeter of the card array,
            currentPixelAverage = np.mean(Rotated[q])
            
            if lastPixelAverage < cardNumberThresholdY and currentPixelAverage >= cardNumberThresholdY:
                crossedThresholdGoingDownY += 1 # transition from white to black (signifies end of card row)
            
            if currentPixelAverage > cardNumberThresholdY and currentPixelAverage <= cardNumberThresholdY:
                crossedThresholdGoingUpY += 1 # transition from black to white (signifies beginning of new card row)
    
            lastPixelAverage = currentPixelAverage # store previous column pixel averages for comparison in next loop iteration
                    
        numRows = crossedThresholdGoingDownY + crossedThresholdGoingUpY
        print('Number of Rows:',numRows)
        numCards = numColumns*numRows
    else:
        # if autoDimensions is False, set override dimensions
        numRows = overrideNumRows
        numColumns = overrideNumColumns
        numCards = overrideNumColumns * overrideNumRows

    # prepare plot frame to plot individual card images    
    print('setting up matplotlib...')
    plt.figure()
    plt.axis('off')
    f, axarr = plt.subplots(numColumns,numRows)   
    print('Done.')
    
    
    ###############################
    
    if autoCrop == True:
        # then crop raw image of cards on table to exclude edges of image that do not contain cards

        # define lists to store pixel indices of cards found within the image
        allWhitesx = []
        allWhitesy = []

        leastWhitex = 0
        greatestWhitex = 0

        leastWhitey = 0
        greatestWhitey = 0

        for x in range(0, convexHull.shape[0]):  # for each x column
            if True in convexHull[x]:   # if one pixel is white (included in convex hull)...
                allWhitesx.append(x) # record that white pixel

        leastWhitex = allWhitesx[0]  # grab left-most white x value
        greatestWhitex = allWhitesx[(len(allWhitesx)-1)]  # grab right-most white x value
        
        rotatedHull = rotate(convexHull, 90, resize=True, preserve_range=False) # rotate image 90 degrees for y-axis access


        for m in range(0, rotatedHull.shape[0]):  # for each x column of rotated image
            if True in rotatedHull[m]:   # if one pixel is white...
                allWhitesy.append(m) # record that pixel
    
        leastWhitey = allWhitesy[0]  # grab least white y value
        greatestWhitey = allWhitesy[(len(allWhitesy)-1)]  # grab greatest white y value

        # crop image using previously found pixel indices as boundaries
        myImageCropped = crop(myImage, ( (leastWhitex, ((convexHull.shape[0])-greatestWhitex)), (((rotatedHull.shape[0])-greatestWhitey),leastWhitey), (0,0)), copy=False)

        # store dimensions of cropped image in variables     
        xDimensionswholeShaped = myImageCropped.shape[0]
        yDimensionswholeShaped = myImageCropped.shape[1]   
        
    else:  # if autoCrop is False,
        myImageCropped = myImage
        xDimensionswholeShaped = myImage.shape[0]
        yDimensionswholeShaped = myImage.shape[1]

        
    currentCardPrintIndex= 0 # assign an index number to the card being analyzed currently
    
    Complete = []
        
    # predict each card
    for q in range(1, (numColumns+1)):
        for y in range(1, (numRows+1)):        
            # for each card,
            if autoCrop == True:

                if np.average(myImageCropped) == 0:
                    print('Hmm...myImageCropped is totally black...')
                
                # define dimensions of grid cells from which to extract individual images of each card
                # grid is divided kinda like an empty tic-tac-toe board
                xDivByCol = int(xDimensionswholeShaped/numColumns)
                yDivByRow = int(yDimensionswholeShaped/numRows)
                
                # crop out the current card from the auto-cropped image of cards on the table
                segmentedImage = crop(myImageCropped, (((xDimensionswholeShaped-((xDivByCol) + ((numColumns-q)*(xDivByCol)))) , ((numColumns-q)*(xDivByCol))), ((yDimensionswholeShaped-((yDivByRow) + ((numRows-y)*(yDivByRow)))) , ((numRows-y)*(yDivByRow))), (0,0)), copy=True)              
                
                # get corner coordinates for bounding box for current card (used later)
                x1 = (xDimensionswholeShaped-((xDivByCol) + ((numColumns-q)*(xDivByCol))))
                x2 = ((numColumns-q)*(xDivByCol))
                x2 = xDimensionswholeShaped-x2
                
                y1 = (yDimensionswholeShaped-((yDivByRow) + ((numRows-y)*(yDivByRow))))
                y2 = ((numRows-y)*(yDivByRow))
                y2 = yDimensionswholeShaped-y2

                arrayofRectanglePoints.append([(y1, x1), (y2, x2)]) # using row, col form
                
                
                
            if autoCrop == False:
                segmentedImage = crop(myImage, (((xDimensionswholeShaped-((xDimensionswholeShaped/numColumns) + ((numColumns-q)*(xDimensionswholeShaped/numColumns)))) , ((numColumns-q)*(xDimensionswholeShaped/numColumns))), ((yDimensionswholeShaped-((yDimensionswholeShaped/numRows) + ((numRows-y)*(yDimensionswholeShaped/numRows)))) , ((numRows-y)*(yDimensionswholeShaped/numRows))), (0,0)), copy=True)
            
            resizedImage = resize(segmentedImage,(xDimensionsofPhotos,yDimensionsofPhotos, colorDepth), anti_aliasing=True, preserve_range=False)   # preserve_range must be False to avoid '1010' syndrome in the predictions!!

                
            #resizedImage = resize(isolatedCard,(xDimensionsofPhotos,yDimensionsofPhotos, colorDepth), anti_aliasing=True, preserve_range=False)  
            
            if resizedImage.shape[0] < resizedImage.shape[1]:  # rotate image so that x side is longer than y side
                # print('rotating card for machine learning model')
                resizedImage = rotate(resizedImage, 90, resize=True, mode='constant', preserve_range=False)
                #isolatedCard = rotate(isolatedCard, 90)
                # print('new card shape:', resizedImage.shape)
            
            # plot image of current card
            axarr[q-1][y-1].set_title(str(currentCardPrintIndex))
            axarr[q-1][y-1].imshow(resizedImage)
            
            # make sure that cropped image is in the right format
            if resizedImage.dtype=='float32' and np.average(resizedImage)<=1 and resizedImage.shape==(180,135,3):
                print('card is correctly formatted as float32 decimal of 180x135x3')
            else:
                # fix the image format
                prepareImage(resizedImage)
            
                    
            # What if probabilities of all three class options are below 0.5 and get rounded down? This is the solution...
            def classifyFromArray(rawArray):
                max_value = amax(rawArray)
                
                if max_value < 0.5:
                    print('Hmm...This card is difficult to classify.')

                for i in range(0, len(rawArray[0])): # rawArray[0] because CNN prediction matrix is nested like [[x,x,x]]
                    if rawArray[0][i] < max_value:
                        rawArray[0][i] = 0 # if this class prediction is not the greatest in the set, round down to zero
                    if rawArray[0][i] == max_value:
                        rawArray[0][i] = 1 # if this class prediction is the greatest in the set, round up to one (top score)
                return rawArray
  
            
           
            
            imgAsArray = np.asarray(resizedImage)
            
            image = np.expand_dims(imgAsArray, axis=0) # nest image array like [[<img data>]]
            
            # predict the color of this card
            colorPrediction = colorModel.predict(image)            
            
            # predict the shade of this card
            shadePrediction = shadeModel.predict(image)
            
            # predict the shape of this card
            shapePrediction = shapeModel.predict(image)
            
            # predict the number of this card
            numberPrediction = numberModel.predict(image)
        
            # keep track of prediction confidences
            confidenceTotals = amax(numberPrediction) * amax(shadePrediction) * amax(colorPrediction) * amax(shapePrediction)
            cardConfidences.append(confidenceTotals)
            
            # analyze prediction matrices to round up the most confident prediction and round down the other predictions
            roundedColorPrediction = classifyFromArray(colorPrediction)
            
            roundedShadePrediction = classifyFromArray(shadePrediction)

            roundedShapePrediction = classifyFromArray(shapePrediction)
        
            roundedNumberPrediction = classifyFromArray(numberPrediction)
            
            # interpret rounded prediction matrices as meaningful predictions

            # COLOR
            if np.array_equal(roundedColorPrediction, [[0,0,1]]) == True:
                colorPredictionAsText = 'red'
                colorPredictionAsInteger = 0
            if np.array_equal(roundedColorPrediction, [[0,1,0]]) == True:
                colorPredictionAsText = 'purple'
                colorPredictionAsInteger = 2
            if np.array_equal(roundedColorPrediction, [[1,0,0]]) == True:
                colorPredictionAsText = 'green'
                colorPredictionAsInteger = 1
                
            # SHADE
            if np.array_equal(roundedShadePrediction, [[0,0,1]]) == True:
                shadePredictionAsText = 'partial'
                shadePredictionAsInteger = 1
            if np.array_equal(roundedShadePrediction, [[0,1,0]]) == True:
                shadePredictionAsText = 'full'
                shadePredictionAsInteger = 2
            if np.array_equal(roundedShadePrediction, [[1,0,0]]) == True:
                shadePredictionAsText = 'empty'
                shadePredictionAsInteger = 0
                
            # SHAPE
            if np.array_equal(roundedShapePrediction, [[0,0,1]]) == True:
                shapePredictionAsText = 'squiggle'
                shapePredictionAsInteger = 2
            if np.array_equal(roundedShapePrediction, [[0,1,0]]) == True:
                shapePredictionAsText = 'oval'
                shapePredictionAsInteger = 0
            if np.array_equal(roundedShapePrediction, [[1,0,0]]) == True:
                shapePredictionAsText = 'diamond'
                shapePredictionAsInteger = 1
                
            # NUMBER
            if np.array_equal(roundedNumberPrediction, [[0,0,1]]) == True:
                numberPredictionAsText = 'two'
                numberPredictionAsInteger = 1
            if np.array_equal(roundedNumberPrediction, [[0,1,0]]) == True:
                numberPredictionAsText = 'three'
                numberPredictionAsInteger = 2
            if np.array_equal(roundedNumberPrediction, [[1,0,0]]) == True:
                numberPredictionAsText = 'one'
                numberPredictionAsInteger = 0

    
            print('card number', currentCardPrintIndex, 'predicted as:', numberPredictionAsText, shadePredictionAsText, colorPredictionAsText, shapePredictionAsText) 
            
            # compile card attribute predictions as a list of integer-encoded labels
            currentPredictions = [numberPredictionAsInteger, shadePredictionAsInteger, colorPredictionAsInteger, shapePredictionAsInteger]

            # store current predictions in a master list of integer-encoded labels
            Complete.append(currentPredictions)
        
            currentCardPrintIndex+=1 # increment for the nest loop interation

    # now that all card classifications are completed,
    print(Complete)
    print('cardConfidences:', cardConfidences)
    found = [] # list to store found sets
 
    # loop through combinations of different cards to find sets
    for i in range(0,(numCards-2)):
        for j in range((i+1),(numCards-1)):
            for k in range((j+1),numCards):
                sameAttributes = 0
                for a in range(0,numAttributes): 
                    if (Complete[i][a] == Complete[j][a] and Complete[j][a] == Complete[k][a]) or ((Complete[i][a] != Complete[j][a] and Complete[j][a] != Complete[k][a])) and Complete[i][a] != Complete[k][a]:
                        sameAttributes+=1
                        if sameAttributes == numAttributes:
                            found.append([i,j,k])
                        

    print('\nFound Sets:', found)

    # display the results in an easy-to-read format using a heads-up display (HUD)
    ResultsHUD(img=myImageCropped, setsFound = found, confidenceRatings=cardConfidences, rawRectanglePoints=arrayofRectanglePoints, numSets2Display=5, boxThickness=3, displayShrinkFactor=1.7)
    # ^ end of analyzePhoto function



def ResultsHUD(img, setsFound, confidenceRatings, rawRectanglePoints, numSets2Display, boxThickness, displayShrinkFactor):
    '''
    displays the found sets in an easy-to-read format using a heads-up display (HUD)
    INPUT SYNTAX:
        img should be an rgb image containing all the cards in view
        setsFound should be a nested array like [[x,x,x], [z,z,z]]
        confidenceRatings should be an array like [1,2,1] where each member is product of the probabilities per card
        rawRectanglePoints should be a nested array like [[(x,y), (x,y)], [(x,y), (x,y)]] where indices of two coordinate pairs is the card number
        numSets2Display should be an integer > 1 or 'all'; sets are displayed in order by greatest probability of being a set
        boxThickness should be an integer > 1; is minimum border thickness of drawn boxes
        displayShrinkFactor should be an integer > 1; downsizes high resolution image to fit on a smaller screen
    '''
    
    if numSets2Display == 'all':
        numSets2Display = len(setsFound)
    cardConfidences = confidenceRatings # these are the raw confidences that have not been rounded down or up
    
    # rescale rawRectanglePoints to appropriate size by dividing by displayShrinkFactor
    shrunkenArrayofRectanglePoints = np.asarray(rawRectanglePoints)/displayShrinkFactor
    shrunkenArrayofRectanglePoints = shrunkenArrayofRectanglePoints.tolist()
    print('shrunken array of points', shrunkenArrayofRectanglePoints)
    
    confidenceEvaluation = []
    perSetConfidence = 1 # identity for multiplication
    for i in range(0, len(setsFound)): # i refers to set index
        perSetConfidence = 1
        for j in range(0, len(setsFound[i])): # j refers to card index
            # for each card in a set, calculate the overall confidence of the set by multiplication of probabilities
            perSetConfidence = perSetConfidence * cardConfidences[setsFound[i][j]]
        confidenceEvaluation.append(perSetConfidence)  # record the set confidence
        
    # define some pretty colors for drawing the bounding boxes
    colors = [(0, 242, 166), (128, 128, 0), (138, 0, 166), (0, 252, 0), (0, 0, 252), (255, 0, 0)]
   
    # downsize image to display on smaller screens
    resizedImageCropped = resize(img, (int(img.shape[0]/displayShrinkFactor), int(img.shape[1]/displayShrinkFactor), 3))
    
    boundedImage = []

    # extract the indices of the top `numSets2Display` sets
    indicesOfTopN = sorted(range(len(confidenceEvaluation)), key = lambda sub: confidenceEvaluation[sub])[-numSets2Display:]

    extendedThickness = boxThickness*numSets2Display # at first, set thickness of box borders to be very thick
    
    for n in range(0, len(indicesOfTopN)):
        # for each set,
        for i in range(0, len(setsFound[n])):
            # for each card in set, extract corner points to draw bounding boxes later
            corner1 = tuple([int(x) for x in shrunkenArrayofRectanglePoints[setsFound[indicesOfTopN[n]][i]][0]])
            corner2 = tuple([int(x) for x in shrunkenArrayofRectanglePoints[setsFound[indicesOfTopN[n]][i]][1]])
            
            # color parameter loops through pre-defined colors sequentially and draws a bounding box around cards in a set
            boundedImage = cv2.rectangle(resizedImageCropped, corner1, corner2, colors[n % (len(colors))], extendedThickness)
        
        if extendedThickness > boxThickness:
            extendedThickness-= boxThickness # make next box drawn have thinner borders
        else:
            extendedThickness = extendedThickness

    boundedImage = boundedImage[:,:,::-1] # convert from RGB to BGR for cv2 displaying
    while True:
        # show HUD image in display window
        cv2.imshow("HUD (press 'q' to close)", boundedImage)
        
        # display HUD image until user presses `q`
        k = cv2.waitKey(50) # check every 50 ms to see if user pressed the `q` key
        if k==ord('q'): # if the user presses `q`
            cv2.destroyWindow("HUD (press 'q' to close)")
            break # break out of HUD displaying loop



def undistort(inputImage):
    '''undistorts a fisheyed input image'''
    undistorted_img = [] # allocate space for de-fisheyed image
    h,w = inputImage.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(inputImage, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img



def printRuntime(start, end):
    '''prints the runtime in a readable format'''
    print('\n')
    if (end-start) < 60:
        print("Execution time: ", end - start, ' seconds')
    else:
        # then runtime is over one minute, so display time in `mm and ss` format
        print('\nExecution time is...', (math.floor((end-start)/60)), 'minute(s) and', ((end-start)-((math.floor((end-start)/60))*60)), 'seconds')