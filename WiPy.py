
import cv2
import os
import glob
import numpy as np
import scipy
from scipy import misc,interpolate,stats,signal
import statsmodels.stats.multitest as mt
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors
import time
import shutil

__version__ = '0.45'

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# MISCELANOUS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_theta_array(szY,szX,mode='deg'):
    #Cesar Echavarria 11/2016


    x = np.linspace(-1, 1, szX)
    y = np.linspace(-1, 1, szY)
    xv, yv = np.meshgrid(x, y)

    [radius,theta]=cart2pol(xv,yv)
    if mode=='deg':
        theta = np.rad2deg(theta)

    return theta

def array2cmap(X):
    N = X.shape[0]

    r = np.linspace(0., 1., N+1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in xrange(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in xrange(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in xrange(N)])

    rd = tuple([(r[i], rd[i], rd[i]) for i in xrange(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in xrange(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in xrange(2 * N)])


    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return colors.LinearSegmentedColormap('my_colormap', cdict, N)
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# GENERAL PURPOSE WiPy FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #


def normalize_stack(frameStack):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("No Arguments Passed!")

    frameStack=np.true_divide((frameStack-np.min(frameStack)),(np.max(frameStack)-np.min(frameStack)))
    return frameStack


def get_frame_times(planFolder):
    #Cesar Echavarria 11/2016

    # READ IN FRAME TIMES FILE
    planFile=open(planFolder+'frameTimes.txt')

    #READ HEADERS AND FIND RELEVANT COLUMNS
    headers=planFile.readline()
    headers=headers.split()

    count = 0
    while count < len(headers):
        if headers[count]=='frameCond':
            condInd=count
            break
        count = count + 1

    count = 0
    while count < len(headers):
        if headers[count]=='frameT':
            timeInd=count
            break
        count = count + 1
    planFile.close()

    # READ IN FRAME TIMES FILE ONCE AGAIN
    planFile=open(planFolder+'frameTimes.txt')
    frameTimes=[]
    frameCond=[]
    # GET DESIRED DATA
    for line in planFile:
        x=line.split()
        frameCond.append(x[condInd])
        frameTimes.append(x[timeInd])
    planFile.close()

    frameTimes.pop(0)#take out header
    frameCond.pop(0)#take out header
    frameTimes=np.array(map(float,frameTimes))
    frameCond=np.array(map(int,frameCond))
    frameCount=len(frameTimes)
    return frameTimes,frameCond,frameCount

def get_block_paradigm(planFolder):
    #Cesar Echavarria 10/2016
    
    # READ IN BLOCK PARADIGM
    planFile=open(planFolder+'blockParadigm.txt')
    blockCond=[]
    blockStartT=[]

    for line in planFile:
        x=line.split()
        blockCond.append(x[1])
        blockStartT.append(x[2])

    blockStartT.pop(0)#take out header
    blockCond.pop(0)#take out header
    blockStartT=np.array(map(float,blockStartT))
    blockCond=np.array(map(int,blockCond))
    nBlocks=len(blockCond)
    return blockCond,blockStartT,nBlocks

def contrast_dictionary_to_labels(targetRoot,nCond):
    #Cesar Echavarria 11/2016
    
    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=targetRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']
    
    labelList=[]
    for c in range(nCond):
        #MAKE LABEL
        endInd=contrastDic[c]['name'].index('_')
        labelList.append(contrastDic[c]['name'][0:endInd])
        
    return labelList

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# STRUCTURAL INFO FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_surface(sourceRoot,targetRoot,sessID):
    #Cesar Echavarria - 10/2016
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
        
    #DEFINE DIRECTORY
    sourceDir=glob.glob(sourceRoot+sessID+'_surface*')
    surfDir=sourceDir[0]+'/Surface/'
    outDir=targetRoot+'/Sessions/'+sessID+'/Surface/';
    if not os.path.exists(outDir):
        os.makedirs(outDir) 
    picList=glob.glob(surfDir+'*.tiff')
    nPics=len(picList)


    # READ IN FRAMES
    imFile=surfDir+'frame0.tiff'
    im0=cv2.imread(imFile,-1)
    sz=im0.shape

    allFrames=np.zeros(sz+(nPics,))
    allFrames[:,:,0]=im0
    for pic in range(1,nPics):
        imFile=surfDir+'frame'+str(pic)+'.tiff'
        im0=cv2.imread(imFile,-1)
        allFrames[:,:,pic]=im0

    #AVERAGE OVER IMAGES IN FOLDER
    imAvg=np.mean(allFrames,2)

    # #SAVE IMAGE

    outFile=outDir+'frame0.tiff'
    cv2.imwrite(outFile,np.uint16(imAvg))#THIS FILE MUST BE OPENED WITH CV2 MODULE

    outFile=outDir+'16bitSurf.tiff'
    imAvg=np.true_divide(imAvg,2**12)*(2**16)
    cv2.imwrite(outFile,np.uint16(imAvg))#THIS FILE MUST BE OPENED WITH CV2 MODULE

def register_surface(sourceRoot,targetRoot,sessID,runList):
    #Cesar Echavarria 1/2017

    #DEFINE DIRECTORIES
    anatSource=targetRoot+'Sessions/'+sessID+'/Surface/'
    outDir=anatSource


    #LOAD IMAGES FOR REGISTRATION
    #READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.squeeze(imSurf)

    imRef=get_reference_frame(sourceRoot,sessID,runList[0])

    #REGISTRATION
    imSurf=np.expand_dims(imSurf,2)
    warpMatrices,motionMag=motion_registration(imRef,imSurf)

    imSurf_new=apply_motion_correction(imSurf,warpMatrices)
    imSurf_new=np.squeeze(imSurf_new)

    outFile=outDir+'frame0_registered.tiff'
    cv2.imwrite(outFile,np.uint16(imSurf_new))#THIS FILE MUST BE OPENED WITH CV2 MODULE

def get_reference_frame(sourceRoot,sessID,refRun=1):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot is not defined!")
    if sessID is None:
        raise TypeError("sessID is not defined!")
    
    runFolder=glob.glob(sourceRoot+sessID+'_run'+str(refRun)+'_*')
    frameFolder=runFolder[0]+"/frames/"
        
    imFile=frameFolder+'frame0.tiff'
    imRef=misc.imread(imFile)
    return imRef

 # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# QUALITY CONTROL FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_quality_control_figure_path(qualityControlRoot,\
    motionCorrection=False, smoothing_fwhm=False):
    #Cesar Echavarria 11/2016

    imgOperationDir=''
    #DEFINE DIRECTORIES
    if motionCorrection:
        imgOperationDir=imgOperationDir+'motionCorrection_'

    if smoothing_fwhm is not False:
        imgOperationDir=imgOperationDir+'smoothing_fwhm'+str(smoothing_fwhm)
    else:
        imgOperationDir=imgOperationDir+'noSmoothing'

    QCtargetFolder=qualityControlRoot+'/'+imgOperationDir+'/Figures/'

    return QCtargetFolder


def get_first_frame_correlation(sourceRoot,sessID,runList):
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if runList is None:
        raise TypeError("runList not specified!")

    for (runCount,run) in enumerate(runList):

        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')
        frameFolder=runFolder[0]+"/frames/"

        # READ IN FRAMES
        imFile=frameFolder+'frame0.tiff'
        im0=misc.imread(imFile)
        sz=im0.shape

        #STORE PIXEL VALUES OF FIRST FRAME
        if runCount==0:
            frame1PixMat=np.zeros((sz[0]*sz[1],len(runList)))
        frame1PixMat[:,runCount]=np.reshape(im0,sz[0]*sz[1])
    R=np.corrcoef(np.transpose(frame1PixMat))   
    
    return R

def quick_quality_control(sourceRoot, targetRoot, sessID, runList):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if runList is None:
        raise TypeError("runList not specified!")

        
    # DEFINE DIRECTORIES

    qualityControlRoot=targetRoot+'Sessions/'+sessID+'/QualControl'
    QCtargetFolder=qualityControlRoot+'/quick/'



    #GO THROUGH RUNS AND GET SOME FRAME VALUES
    for (runCount,run) in enumerate(runList):

        print('run='+str(run))

        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')

        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"

        # READ IN FRAME TIMES FILE
        frameTimes,frameCond,frameCount=get_frame_times(planFolder)
        frameTimes=frameTimes[0::100]


        
        print('Loading frames...')

        #GET REFFERNCE FRAME
        imRef=get_reference_frame(sourceRoot,sessID,runList[0])
        szY,szX=imRef.shape

        # READ IN FRAMES
        frameArray=np.zeros((szY,szX,frameCount))
        for f in range (0,frameCount,100):
            imFile=frameFolder+'frame'+str(f)+'.tiff'
            im0=misc.imread(imFile)
            frameArray[:,:,f]=np.copy(im0)
        frameArray=frameArray[:,:,0::100]


        frameArray=np.reshape(frameArray,(szY*szX,np.shape(frameArray)[2]))

        if not os.path.exists(QCtargetFolder):
            os.makedirs(QCtargetFolder)

        meanF=np.squeeze(np.mean(frameArray,0))#average over pixels

        fig=plt.figure()
        plt.plot(frameTimes,meanF)

        fig.suptitle('Mean Pixel Value Over Time', fontsize=20)
        plt.xlabel('Time (secs)',fontsize=16)
        plt.ylabel('Mean Pixel Value',fontsize=16)
        plt.savefig(QCtargetFolder+sessID+'_run'+str(run)+'_meanPixelValue.png')
        plt.close()

        randPix=np.random.randint(0,szY*szX)

        fig=plt.figure()
        plt.plot(frameTimes,frameArray[randPix,:])

        fig.suptitle('Pixel '+str(randPix)+' Value Over Time', fontsize=20)
        plt.xlabel('Time (secs)',fontsize=16)
        plt.ylabel('Mean Pixel Value',fontsize=16)
        plt.savefig(QCtargetFolder+sessID+'_run'+str(run)+'_randomPixelValue.png')
        plt.close()

    R=get_first_frame_correlation(sourceRoot,sessID,runList)
    fig=plt.figure()
    plt.imshow(R,interpolation='none')
    plt.colorbar()
    plt.savefig(QCtargetFolder+sessID+'_firstFrame_CorrelationMatrix.png')
    plt.close()
        


# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# MOTION CORRECTION FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

def motion_registration(imRef,frameStack):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if imRef is None:
        raise TypeError("imRef (reference image) is not defined!")
    if frameStack is None:
        raise TypeError("no frame stack defined!")


    frameCount=np.shape(frameStack)[2]

    imRef_gray=np.uint8(np.true_divide(imRef,np.max(imRef))*255)
    imRef_smooth=cv2.GaussianBlur(imRef_gray, (11,11), .9, .9)
    imRef_forReg = get_gradient(imRef_smooth)
    
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Specify the number of iterations.
    number_of_iterations = 10;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    warpMatrices=np.zeros((2,3,frameCount))
    motionMag=np.zeros((frameCount))
    
    for f in range (0,frameCount):
        if f%1000==0:
            print('Motion Registration at frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        im0_gray=np.uint8(np.true_divide(im0,np.max(im0))*255)
        im0_smooth=cv2.GaussianBlur(im0_gray, (11,11), .9, .9)
        im0_forReg = get_gradient(im0_smooth)
        

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)


        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (imRef_forReg,im0_forReg,warp_matrix, warp_mode, criteria)
        warpMatrices[:,:,f]=warp_matrix
        motionMag[f]=np.sum(np.square(np.eye(2, 3, dtype=np.float32)-warp_matrix))
    print(np.argmax(motionMag)) 
    
    return warpMatrices,motionMag

def apply_motion_correction(frameStack=None,warpMatrices=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack defined!")
    if frameStack is None:
        raise TypeError("no warp matrices provided!")
        
  
        
    #APPLY TRANSFORMATION AND TRIM
    szY,szX,frameCount=np.shape(frameStack)
    newFrameStack=np.zeros((szY,szX,frameCount))
    
    for f in range (0,frameCount):
        if f%1000==0:
            print('Motion Correction at frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        warpMatrix=warpMatrices[:,:,f]
        im1 = cv2.warpAffine(im0, warpMatrix, (szX,szY), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        newFrameStack[:,:,f]=im1
    return newFrameStack



def get_boundaries(img=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if img is None:
        raise TypeError("no image array given!")
    
    szY,szX = np.shape(img)
    theta = get_theta_array(szY,szX)
    
    
    halfAngle=42.5
    physicalUp=np.logical_and(theta>-90-halfAngle, theta <-90+halfAngle)
    physicalRight=np.logical_and(theta>-halfAngle, theta <halfAngle)
    physicalDown=np.logical_and(theta>90-halfAngle, theta <90+halfAngle)
    physicalLeft=np.abs(theta)>180-halfAngle
    

    zeroUp=np.where(np.logical_and(img<10,physicalUp))[0]
    zeroDown=np.where(np.logical_and(img<10,physicalDown))[0]
    zeroLeft=np.where(np.logical_and(img<10,physicalLeft))[1]
    zeroRight=np.where(np.logical_and(img<10,physicalRight))[1]

    if np.size(zeroUp)==0:
        edgeUp=2
    else:
        edgeUp=np.max(zeroUp)+3

    if np.size(zeroDown)==0:
        edgeDown=szY-2
    else:
        edgeDown=np.min(zeroDown)-3

    if np.size(zeroLeft)==0:
        edgeLeft=2
    else:
        edgeLeft=np.max(zeroLeft)+3

    if np.size(zeroRight)==0:
        edgeRight=szX-2
    else:
        edgeRight=np.min(zeroRight)-3
        
    return edgeUp,edgeDown,edgeLeft,edgeRight

def get_motion_corrected_boundaries(frameStack=None,warpMatrices=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack given!")
    if warpMatrices is None:
        raise TypeError("no warp matrix stack given!")

    print('Getting motion corrected image boundaries...')
    szY,szX,frameCount=np.shape(frameStack) 
    
    edgeUpList=np.zeros((frameCount))
    edgeDownList=np.zeros((frameCount))
    edgeLeftList=np.zeros((frameCount))
    edgeRightList=np.zeros((frameCount))
    for f in range (0,frameCount):
        im0=np.copy(frameStack[:,:,f])
        
        edgeUp,edgeDown,edgeLeft,edgeRight = get_boundaries(im0)

        edgeUpList[f]=edgeUp
        edgeDownList[f]=edgeDown
        edgeLeftList[f]=edgeLeft
        edgeRightList[f]=edgeRight
    
    return edgeUpList,edgeDownList,edgeLeftList,edgeRightList
    

def apply_motion_correction_boundaries(frameStack=None,boundaries=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack given!")
    if boundaries is None:
        raise TypeError("no boundaries given!")
    
    print('Applying motion corrected image boundaries...')
    frameCount=np.shape(frameStack)[2]
    
    edgeUp=boundaries[0]
    edgeDown=boundaries[1]
    edgeLeft=boundaries[2]
    edgeRight=boundaries[3]
    
    newSzY=edgeDown-edgeUp
    newSzX=edgeRight-edgeLeft
    
    newFrameStack=np.zeros((newSzY,newSzX,frameCount))
    for f in range (0,frameCount):
        if f%1000==0:
            print('frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        newFrameStack[:,:,f]=im0[edgeUp:edgeDown,edgeLeft:edgeRight]
        
    return newFrameStack

def perform_motion_registration(sourceRoot,targetRoot,sessID,runList,refRun=1,saveFrames=True,makeMovies=False,frameRate=None):
    #Cesar Echavarria 11/2016
    
     #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if makeMovies and frameRate is None:
        raise TypeError("user input to make movies but frame rate not specified!")
        
    #DEFINE AND MAKE DIRECTORIES
    motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
    if not os.path.exists(motionDir):
                os.makedirs(motionDir)

    motionFileDir=motionDir+'Registration/'
    if not os.path.exists(motionFileDir):
                os.makedirs(motionFileDir)

    motionFigDir=motionDir+'Figures/'
    if not os.path.exists(motionFigDir):
                os.makedirs(motionFigDir)
            
    motionMovieDir=motionDir+'Movies/'
    if makeMovies:
        if not os.path.exists(motionMovieDir):
            os.makedirs(motionMovieDir)

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
	outFile=motionFileDir+'analysis_version_info.txt'
	versionTextFile = open(outFile, 'w+')
	versionTextFile.write('WiPy version '+__version__+'\n')
	versionTextFile.close()

    #GET REFFERNCE FRAME
    imRef=get_reference_frame(sourceRoot,sessID,refRun)
    szY,szX=imRef.shape


    #PERFORM REGISTRATION FOR ALL RUNS
    for (runCount,run) in enumerate(runList):
        print('Performing image registration for run '+str(run))

        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')
        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"

        frameTimes,frameCond,frameCount=get_frame_times(planFolder)

        #READ IN FRAMES
        frameArray=np.zeros((szY,szX,frameCount))
        print('Loading frames....')
        for f in range (0,frameCount):
            imFile=frameFolder+'frame'+str(f)+'.tiff'
            im0=misc.imread(imFile)
            frameArray[:,:,f]=im0[:,:]
            
        if makeMovies:
            #GENRATE RAW DATA MOVIE
            frameArrayNorm=normalize_stack(frameArray)
            outFile=motionMovieDir+sessID+'_run'+str(run)+'_raw_stack.mp4'
            make_movie_from_stack(motionMovieDir,frameArrayNorm,frameRate,outFile)
    

        #MOTION REGISTRATION
        warpMatrices,motionMag=motion_registration(imRef,frameArray)

        #-> plot motion magnitude (squared error from identity matrix) and save figure
        fig=plt.figure()
        plt.plot(frameTimes,motionMag)
        fig.suptitle('Motion Over Time', fontsize=20)
        plt.xlabel('Time (secs)',fontsize=16)
        plt.ylabel('Motion Magnitude (AU)',fontsize=16)
        plt.savefig(motionFigDir+sessID+'_run'+str(run)+'_motionMagnitude.png')
        plt.close()

        #-> save warp matrices
        outFile=motionFileDir+sessID+'_run'+str(run)+'_motionRegistration'
        np.savez(outFile,warpMatrices=warpMatrices)

        #APPLY MOTION CORRECTION AND SAVE
        correctedFrameArray=apply_motion_correction(frameArray,warpMatrices)
        if saveFrames:
            outFile=motionFileDir+sessID+'_run'+str(run)+'_correctedFrames'
            np.savez(outFile,correctedFrameArray=correctedFrameArray)

        dummyArray=np.ones(frameArray.shape)*np.mean(frameArray[:])
        correctedDummyArray=apply_motion_correction(dummyArray,warpMatrices)

        #GET MOTION CORRECTED BOUNDARIES
        edgeUpList,edgeDownList,edgeLeftList,edgeRightList = get_motion_corrected_boundaries(correctedDummyArray,warpMatrices)

        if runCount==0:
            if runList[0] == refRun:
                edgeUp=np.max(edgeUpList)
                edgeDown=np.min(edgeDownList)
                edgeLeft=np.max(edgeLeftList)
                edgeRight=np.min(edgeRightList)
            else:
                #LOAD BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries_intermediate.npz'
                f=np.load(inFile)
                boundaries_tmp=f['boundaries']

                edgeUp=boundaries_tmp[0]
                edgeDown=boundaries_tmp[1]
                edgeLeft=boundaries_tmp[2]
                edgeRight=boundaries_tmp[3]

                edgeUp=np.max([edgeUpLast,edpeUpLast])
                edgeDown=np.min([edgeDownLast,np.min(edgeDownList)])
                edgeLeft=np.max([edgeLeftLast,np.max(edgeLeftList)])
                edgeRight=np.min([edgeRightLast,np.min(edgeRightList)])
        else:
            edgeUp=np.max([edgeUp,np.max(edgeUpList)])
            edgeDown=np.min([edgeDown,np.min(edgeDownList)])
            edgeLeft=np.max([edgeLeft,np.max(edgeLeftList)])
            edgeRight=np.min([edgeRight,np.min(edgeRightList)])

        boundaries=(edgeUp,edgeDown,edgeLeft,edgeRight)  
        #->save boundaries
        outFile=motionFileDir+sessID+'_motionCorrectedBoundaries_intermediate'
        np.savez(outFile,boundaries=boundaries)

        if makeMovies:
            #APPLY BOUNDARIES
            tmp_boundaries=(edgeUp,edgeDown,edgeLeft,edgeRight)  
            trimCorrectedFrameArray = apply_motion_correction_boundaries(correctedFrameArray,tmp_boundaries)

            #GENRATE MOTION CORRECTED MOVIE
            frameArrayNorm=normalize_stack(trimCorrectedFrameArray)
            outFile=motionMovieDir+sessID+'_run'+str(run)+'_MC_trimmed_stack.mp4'
            make_movie_from_stack(motionMovieDir,frameArrayNorm,frameRate,outFile)

    boundaries=(edgeUp,edgeDown,edgeLeft,edgeRight)  
    #->save boundaries
    outFile=motionFileDir+sessID+'_motionCorrectedBoundaries'
    np.savez(outFile,boundaries=boundaries)
    


# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# STAGE 1 -- DATA ANALYSIS FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_analysis_path(analysisRoot,interp=False, removeRollingMean=False, \
    motionCorrection=False, smoothing_fwhm=False, \
    timecourseAnalysis=False, baseline_startT=-2, baseline_endT=0, timecourse_startT=0, timecourse_endT=8,\
     timeWindowAnalysis=False,tWindow1_startT=-2, tWindow1_endT=0, tWindow2_startT=0, tWindow2_endT=5):
	#Cesar Echavarria 11/2016

    motionString=''
    smoothString=''
    interpString=''
    averageString=''
    removeRollingString=''
    #DEFINE DIRECTORIES
    if motionCorrection:
        motionString='motionCorrection'
    else:
        motionString='noMotionCorrection'

    if smoothing_fwhm is not None:
        smoothString='_smoothing_fwhm'+str(smoothing_fwhm)

    imgOperationDir=motionString+smoothString


    procedureDir=''
    if interp:
        interpString='interpolate_'


    if removeRollingMean:
        removeRollingString='minusRollingMean'

    procedureDir=interpString+removeRollingString

    tCourseDir=None
    tWindowDir=None
    if timeWindowAnalysis:
        analysisDir='%stimeWindow/%s/%s/'%(analysisRoot,imgOperationDir,interpString+averageString+removeRollingString)
        tWindowDir='%stWindow1_%s_%s_tWindow2_%s_%s/'%(analysisDir,tWindow1_startT,tWindow1_endT,tWindow2_startT,tWindow2_endT)
    if timecourseAnalysis:
        analysisDir='%stimeCourse/%s/%s/'%(analysisRoot,imgOperationDir,interpString+averageString+removeRollingString)
        tCourseDir='%sbaseline_%s_%s_response_%s_%s/'%(analysisDir,baseline_startT,baseline_endT,timecourse_startT,timecourse_endT)
    

    return(tWindowDir,tCourseDir)
  
def analyze_blocked_data(sourceRoot, targetRoot, sessID, runList, frameRate,\
    interp=False, removeRollingMean=False, \
    motionCorrection=False, smoothing_fwhm=False, mask=None, \
    timecourseAnalysis=False, baseline_startT=-2, baseline_endT=0, timecourse_startT=0, timecourse_endT=8,\
     timeWindowAnalysis=False,tWindow1_startT=-2, tWindow1_endT=0, tWindow2_startT=0, tWindow2_endT=5,\
     loadCorrectedFrames=True,qualityControl=True):
    #Cesar Echavarria 11/2016


    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if frameRate is None:
        raise TypeError("frameRate not specified!")

    # DEFINE DIRECTORIES
    analysisRoot=targetRoot+'/Sessions/'+sessID+'/Analyses/';

    tWindowOutDir,tCourseOutDir=get_analysis_path(analysisRoot,interp, removeRollingMean, \
        motionCorrection, smoothing_fwhm, \
        timecourseAnalysis, baseline_startT, baseline_endT, timecourse_startT, timecourse_endT,\
        timeWindowAnalysis,tWindow1_startT, tWindow1_endT, tWindow2_startT, tWindow2_endT)

    qualityControlRoot=targetRoot+'Sessions/'+sessID+'/QualControl'
    QCtargetFolder=get_quality_control_figure_path(qualityControlRoot, motionCorrection, smoothing_fwhm)

    motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'


    #MAKE NECESSARY DIRECTORIES
    if timeWindowAnalysis:
        tWindowOutDir=tWindowOutDir+'MidProcOutput/'
        if not os.path.exists(tWindowOutDir):
            os.makedirs(tWindowOutDir)

        #OUTPUT TEXT FILE WITH PACKAGE VERSION
        outFile=tWindowOutDir+'analysis_version_info.txt'
        versionTextFile = open(outFile, 'w+')
        versionTextFile.write('WiPy version '+__version__+'\n')
        versionTextFile.close() 


    if timecourseAnalysis:
        tCourseOutDir=tCourseOutDir+'MidProcOutput/'
        if not os.path.exists(tCourseOutDir):
            os.makedirs(tCourseOutDir) 

        #OUTPUT TEXT FILE WITH PACKAGE VERSION
        outFile=tCourseOutDir+'analysis_version_info.txt'
        versionTextFile = open(outFile, 'w+')
        versionTextFile.write('WiPy version '+__version__+'\n')
        versionTextFile.close() 


    #BEGIN DATA PROCESSING
    for (runCount,run) in enumerate(runList):

        print('run='+str(run))

        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')

        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"

        # READ IN FRAME TIMES FILE
        frameTimes,frameCond,frameCount=get_frame_times(planFolder)
        # READ IN BLOCK PARADIGM
        blockCond,blockStartT,nBlocks=get_block_paradigm(planFolder)



        print('Loading frames...')
        if motionCorrection:

            if loadCorrectedFrames:
                #LOAD MOTION CORRECTED FRAMES
                inFile=outFile=motionFileDir+sessID+'_run'+str(run)+'_correctedFrames.npz'
                f=np.load(inFile)
                frameArray=f['correctedFrameArray']
            else:
                #GET REFFERNCE FRAME
                imRef=get_reference_frame(sourceRoot,sessID,runList[0])
                szY,szX=imRef.shape

                # READ IN FRAMES
                frameArray=np.zeros((szY,szX,frameCount))
                for f in range (0,frameCount):
                    imFile=frameFolder+'frame'+str(f)+'.tiff'
                    im0=misc.imread(imFile)
                    frameArray[:,:,f]=np.copy(im0)

                #-> load warp matrices
                inFile=motionFileDir+sessID+'_run'+str(run)+'_motionRegistration.npz'
                f=np.load(inFile)
                warpMatrices=f['warpMatrices']

                #APPLY MOTION CORRECTION
                frameArray=apply_motion_correction(frameArray,warpMatrices)

            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']

            #APPLY BOUNDARIES
            frameArray = apply_motion_correction_boundaries(frameArray,boundaries)

            szY,szX = np.shape(frameArray[:,:,0])


        else:
            #GET REFFERNCE FRAME
            imRef=get_reference_frame(sourceRoot,sessID,runList[0])
            szY,szX=imRef.shape

            # READ IN FRAMES
            frameArray=np.zeros((szY,szX,frameCount))
            for f in range (0,frameCount):
                imFile=frameFolder+'frame'+str(f)+'.tiff'
                im0=misc.imread(imFile)
                frameArray[:,:,f]=np.copy(im0)

        #APPLY SMOOTHING
        for f in range(0,frameCount):
            im0=np.copy(frameArray[:,:,f])
            if smoothing_fwhm is not False:
                if smoothing_fwhm == 3:
                    frameArray[:,:,f]=cv2.GaussianBlur(im0, (11,11), .9, .9)
                elif smoothing_fwhm == 5:
                    frameArray[:,:,f]=cv2.GaussianBlur(im0, (21,21), 1.7, 1.7)
                elif smoothing_fwhm == 7:
                    frameArray[:,:,f]=cv2.GaussianBlur(im0, (27,27), 2.6, 2.6)
                elif smoothing_fwhm == 9:
                    frameArray[:,:,f]=cv2.GaussianBlur(im0, (31,31), 3.4, 3.4)
                elif smoothing_fwhm == 11:
                    frameArray[:,:,f]=cv2.GaussianBlur(im0, (37,37), 4.3, 4.3)
                else:
                    raise TypeError("KERNEL NOT IMPLEMENTED")
        #RESHAPE ARRAY TO PIX VS TIME DIMENSIONS
        frameArray=np.reshape(frameArray,(szY*szX,frameCount))

        #GET MASK IF NECESSARY
        if mask is not None:
            maskFile=targetRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
            f=np.load(maskFile)
            maskM=f['maskM']
            tmp=np.where(maskM==1)
            maskInd=np.ravel_multi_index(tmp, (szY,szX))

        if qualityControl:
            if not os.path.exists(QCtargetFolder):
                os.makedirs(QCtargetFolder)

            stimBlockStartT=blockStartT[np.not_equal(blockCond,0)]
            stimBlockEndT=stimBlockStartT+tWindow2_endT
            plotPix=np.ravel_multi_index([szY/2,szX/2], (szY,szX))

            fig=plt.figure()
            plt.plot(frameTimes,frameArray[plotPix,:])
            fig.suptitle('Pixel '+str(plotPix)+' Value Over Time', fontsize=20)
            plt.xlabel('Time (secs)',fontsize=16)
            plt.ylabel('Mean Pixel Value',fontsize=16)
            ax=plt.gca()
            for i in range(len(stimBlockStartT)):
                ax.axvspan(stimBlockStartT[i], stimBlockEndT[i], alpha=0.20, color='red')
            plt.savefig(QCtargetFolder+sessID+'_run'+str(run)+'_pixelValue_beforeProcessing.png')
            plt.close()

        #APPLY MASK, IF SPECIFIED, TO SPEED UP PROCESSING
        if mask is not None:
            frameArray=frameArray[maskInd,:]
            
        # INTERPOLATE FOR CONSTANT FRAME RATE
        if interp:
            print('Interpolating....')
            interpF = interpolate.interp1d(frameTimes, frameArray,1)
            newTimes=np.arange(frameTimes[0],frameTimes[-1],1.0/frameRate)
            frameArray=interpF(newTimes)   # use interpolation function returned by `interp1d`
            frameTimes=newTimes

        # REMOVE ROLLING AVERAGE
        if removeRollingMean:
            print('Removing rolling mean....')
            detrendedFrameArray=np.zeros(np.shape(frameArray))
            rollingWindowSz=int(np.ceil(((tWindow2_endT-tWindow2_startT)*2)*frameRate))

            for pix in range(0,np.shape(frameArray)[0]):
                meanVal=np.mean(frameArray[pix,:],0)

                tmp0=frameArray[pix,:];
                tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                rollingAvg=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                rollingAvg=rollingAvg[rollingWindowSz:-rollingWindowSz]


                detrendedFrameArray[pix,:]=np.subtract(tmp0,rollingAvg)+meanVal;
            frameArray=detrendedFrameArray
            del detrendedFrameArray

        if mask is not None:
            tmp=np.zeros((szX*szY,np.shape(frameArray)[1]))
            tmp[maskInd,:]=frameArray
            frameArray=tmp
            del tmp

        if qualityControl:
            fig=plt.figure()
            plt.plot(frameTimes,frameArray[plotPix,:])

            fig.suptitle('Pixel '+str(plotPix)+' Value Over Time', fontsize=20)
            plt.xlabel('Time (secs)',fontsize=16)
            plt.ylabel('Mean Pixel Value',fontsize=16)
            ax=plt.gca()
            for i in range(len(stimBlockStartT)):
                ax.axvspan(stimBlockStartT[i], stimBlockEndT[i], alpha=0.20, color='red')
            plt.savefig(QCtargetFolder+sessID+'_run'+str(run)+'_pixelValue_afterProcessing.png')
            plt.close()



        # GET BASELINE AND STIM RESPONSE TIME COURSE
        print('aggregating responses across trials....')
        stimBlockStartT=blockStartT[np.not_equal(blockCond,0)]
        stimBlockCond=blockCond[np.not_equal(blockCond,0)]
        nStimBlocks=len(stimBlockStartT)
        nPtsBase=int(np.ceil((tWindow1_endT-tWindow1_startT)*frameRate))
        nPtsStim=int(np.ceil((tWindow2_endT-tWindow2_startT)*frameRate))

        baseResp=np.zeros((szY*szX,nStimBlocks,nPtsBase))
        stimResp=np.zeros((szY*szX,nStimBlocks,nPtsStim))
        if timeWindowAnalysis:
            for stimBlock in range(0,nStimBlocks):
                window1_startInd=np.where(frameTimes>stimBlockStartT[stimBlock]+tWindow1_startT)[0][0]
                window1_endInd=window1_startInd+nPtsBase
                window2_startInd=np.where(frameTimes>stimBlockStartT[stimBlock]+tWindow2_startT)[0][0]
                window2_endInd=window2_startInd+nPtsStim

                baseResp[:,stimBlock,:]=frameArray[:,int(window1_startInd):int(window1_endInd)]
                stimResp[:,stimBlock,:]=frameArray[:,int(window2_startInd):int(window2_endInd)]

            outFile=tWindowOutDir+sessID+'_run'+str(run)
            try:
                np.savez(outFile,stimBlockCond=stimBlockCond,baseResp=baseResp,stimResp=stimResp,frameRate=frameRate,tWindow1_startT=tWindow1_startT,\
                    tWindow1_endT=tWindow1_endT,tWindow2_startT=tWindow2_startT,tWindow2_endT=tWindow2_endT)
            except IOError as e:
                #if file is too big,save in parts
                outFile='%s_run%s_miscellaneous'(tWindowOutDir+sessID,str(run))
                np.savez(outFile,stimBlockCond=stimBlockCond,frameRate=frameRate,tWindow1_startT=tWindow1_startT,\
                        tWindow1_endT=tWindow1_endT,tWindow2_startT=tWindow2_startT,tWindow2_endT=tWindow2_endT)

                nParts=4
                blocksPerPart=np.ceil(np.true_divide(nStimBlocks,nParts))
                for p in range(nParts):
                    startBlock=p*blocksPerPart
                    if p == nParts-1:
                        endBlock=nStimBlocks
                    else:
                        endBlock=startBlock+(blocksPerPart)

                    outFile='%s_run%s_part%s_of%s'%\
                        (tWindowOutDir+sessID,str(run),str(p+1),str(nParts))
                    np.savez(outFile,baseResp=baseResp[:,startBlock:endBlock:],stimResp=stimResp[:,startBlock:endBlock,:],\
                        startBlock=startBlock,endBlock=endBlock,nParts=nParts)
        if timecourseAnalysis:
            nPtsBase=int(np.ceil((baseline_endT-baseline_startT)*frameRate))
            nPtsResponse=int(np.ceil((timecourse_endT-timecourse_startT)*frameRate))

            baseline=np.zeros((szY*szX,nStimBlocks,nPtsBase))
            response=np.zeros((szY*szX,nStimBlocks,nPtsResponse))

            for stimBlock in range(0,nStimBlocks):
                baseline_startInd=np.where(frameTimes>stimBlockStartT[stimBlock]+baseline_startT)[0][0]
                baseline_endInd=baseline_startInd+nPtsBase
                timecourse_startInd=np.where(frameTimes>stimBlockStartT[stimBlock]+timecourse_startT)[0][0]
                timecourse_endInd=timecourse_startInd+nPtsResponse

                baseline[:,stimBlock,:]=frameArray[:,int(baseline_startInd):int(baseline_endInd)]
                response[:,stimBlock,:]=frameArray[:,int(timecourse_startInd):int(timecourse_endInd)]

            
            
            outFile=tCourseOutDir+sessID+'_run'+str(run)+'_miscellaneous'
            np.savez(outFile,stimBlockCond=stimBlockCond,frameRate=frameRate,baseline_startT=baseline_startT,baseline_endT=baseline_endT,timecourse_startT=timecourse_startT,timecourse_endT=timecourse_endT)
            try:
                outFile=tCourseOutDir+sessID+'_run'+str(run)+'_baseline'
                np.savez(outFile,baseline=baseline)
            except IOError as e:
                #if file is too big,save in parts
                nParts=4
                blocksPerPart=np.ceil(np.true_divide(nStimBlocks,nParts))
                for p in range(nParts):
                    startBlock=p*blocksPerPart
                    if p == nParts-1:
                        endBlock=nStimBlocks
                    else:
                        endBlock=startBlock+(blocksPerPart)

                    outFile='%s_run%s_baseline_part%s_of%s'%\
                        (tWindowOutDir+sessID,str(run),str(p+1),str(nParts))
                    np.savez(outFile,baseline=baseline[:,startBlock:endBlock:])
            try:
                outFile=tCourseOutDir+sessID+'_run'+str(run)+'_response'
                np.savez(outFile,response=response)
            except IOError as e:
                #if file is too big,save in parts
                nParts=4
                blocksPerPart=np.ceil(np.true_divide(nStimBlocks,nParts))
                for p in range(nParts):
                    startBlock=p*blocksPerPart
                    if p == nParts-1:
                        endBlock=nStimBlocks
                    else:
                        endBlock=startBlock+(blocksPerPart)

                    outFile='%s_run%s_response_part%s_of%s'%\
                        (tWindowOutDir+sessID,str(run),str(p+1),str(nParts))
                    np.savez(outFile,response=response[:,startBlock:endBlock:],startBlock=startBlock,endBlock=endBlock,nParts=nParts)



        
def average_trials_timecourse(sourceRoot, targetRoot, analysisDir, sessID, nCond, runList, avgFolder,\
    motionCorrection=False, percentSignalChange=True, SDmaps=False):
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if nCond is None:
        raise TypeError("nCond (number of conditions) not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")


    # DEFINE DIRECTORIES
    inDir=analysisDir+'/MidProcOutput/'
    outDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';
    if not os.path.exists(outDir):
        os.makedirs(outDir) 

    if SDmaps:
        figOutDir=analysisDir+'/Figures/'+avgFolder+'/SDmaps/'
        if not os.path.exists(figOutDir):
            os.makedirs(figOutDir) 

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=outDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()

    #OUTPUT TEXT FILE WITH RUN LIST
    outFile=outDir+'list_of_runs_used.txt'
    runTextFile = open(outFile, 'w+')
    runTextFile.write('LIST OF RUNS\n')
    runTextFile.write('\n')
    for run in runList:
        runTextFile.write(str(run)+' ')
    runTextFile.close()

    if motionCorrection:
        #GET MOTION CORRECTED BOUNDARIES
        motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
        motionFileDir=motionDir+'Registration/'
        inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
        f=np.load(inFile)
        boundaries=f['boundaries']
        edgeUp=boundaries[0]
        edgeDown=boundaries[1]
        edgeLeft=boundaries[2]
        edgeRight=boundaries[3]
        #GET MOTION CORRECTED FRAME SIZE
        szY=edgeDown-edgeUp
        szX=edgeRight-edgeLeft
    else:
        #GET REFFERNCE FRAME
        imRef=get_reference_frame(sourceRoot,sessID,runList[0])
        szY,szX=imRef.shape



    for cond in range(nCond):
        print("condition = "+str(cond+1))
        # RETRIEVE DATA
        for (runCount,run) in enumerate(runList):
            print("run = "+str(run))

            inFile=inDir+sessID+'_run'+str(run)+'_baseline.npz'
            f=np.load(inFile)
            baseResp=f['baseline']

            inFile=inDir+sessID+'_run'+str(run)+'_response.npz'
            f=np.load(inFile)
            stimResp=f['response']

            inFile=inDir+sessID+'_run'+str(run)+'_miscellaneous.npz'
            f=np.load(inFile)
            stimBlockCond=f['stimBlockCond']
            
            
            
            if runCount == 0:
                frameRate=f['frameRate']
                timecourse_startT=f['timecourse_startT']
                timecourse_endT=f['timecourse_endT']
                baseline_startT=f['baseline_startT']
                baseline_endT=f['baseline_endT']



            nPix,nTrials,basePts=np.shape(baseResp)
            nPix,nTrials,respPts=np.shape(stimResp)

            condInd=np.where(stimBlockCond==cond+1)[0]

            if runCount==0:

                stimRespAll=np.zeros((nPix,len(condInd)*len(runList),respPts))
                baseRespAll=np.zeros((nPix,len(condInd)*len(runList),basePts))

            startInd=(runCount)*len(condInd)
            endInd=startInd+len(condInd)

            stimRespAll[:,startInd:endInd,:]=stimResp[:,condInd,:]
            baseRespAll[:,startInd:endInd,:]=baseResp[:,condInd,:]

        #WRITE NUMBER OF TRIALS PER CONDITION TO TEXT FILE
        if cond == 0:
            outFile=outDir+'total_trials_per_condition.txt'
            trialTextFile = open(outFile, 'w+')
            trialTextFile.write('NUMBER OF TRIALS\n')
            trialTextFile.write('\n')

        
        trialTextFile.write('condition '+str(cond+1)+' '+str(np.shape(stimRespAll)[1])+' trials\n')
        


        if percentSignalChange:#output percent signal change avged over trials
            if cond==0:
                responseAll=np.zeros((nPix,nCond,np.shape(stimRespAll)[1],respPts))
            baseRespTimeMean=np.mean(baseRespAll,2)#avg over time

            for pt in range(0,respPts):
                responseAll[:,cond,:,pt]=np.true_divide(stimRespAll[:,:,pt]-baseRespTimeMean,baseRespTimeMean)*100

        else: #output pixel values avged over trials
            if cond==0:
                responseAll=np.zeros((nPix,nCond,np.shape(stimRespAll)[1],respPts))
            responseAll[:,cond,:,:]=stimRespAll

        outFile=outDir+sessID+'_'+str(cond)+'_timecourse_allTrials'
        try:
            np.savez(outFile,responseAll=responseAll[:,cond,:,:],nCond=nCond)
        except IOError as e:
            #if file is too big,save in parts
            nParts=4
            trialsPerPart=np.ceil(np.true_divide(nTrials,nParts))
            for p in range(nParts):
                startTrial=p*trialsPerPart
                if p == nParts-1:
                    endTrial=nTrials
                else:
                    endTrial=startTrial+(trialsPerPart)

                outFile='%s_%s_timecourse_allTrials_part%s_of%s'%\
                    (outDir+sessID,str(cond),str(p+1),str(nParts))
                np.savez(outFile,responseAll=responseAll[:,cond,startTrial:endTrial,:],nCond=nCond,\
                    startTrial=startTrial,endTrial=endTrial,nParts=nParts)

    stimRespMean=np.mean(responseAll,2)#avg over trials


    trialTextFile.close()
    

    outFile=outDir+sessID+'_timecourse_trialAvg'
    np.savez(outFile,stimRespMean=stimRespMean)

    outFile=outDir+sessID+'_timecourse_miscelleanous'
    np.savez(outFile,frameRate=frameRate,percentSignalChange=percentSignalChange,\
        timecourse_startT=timecourse_startT,timecourse_endT=timecourse_endT,\
        baseline_startT=baseline_startT,baseline_endT=baseline_endT,\
        nPix=nPix,nCond=nCond,nTrials=np.shape(stimRespAll)[1],respPts=respPts)
    

    #OUTPUT AND SAVE STANDARD DEV
    if SDmaps:

        pixSD=np.std(stimRespMean,2)
        mapSD=np.reshape(pixSD,(szY,szX,nCond))

        #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
        inFile=targetRoot+'contrastDic.npz'
        f=np.load(inFile)
        contrastDic=f['contrastDic']


        for c in range(0,nCond):
            fig=plt.figure()
            plt.imshow(mapSD[:,:,c])
            plt.colorbar()
            endInd=contrastDic[c]['name'].index('_')
            plt.savefig(figOutDir+sessID+'_'+contrastDic[c]['name'][0:endInd]+'_SDmap.png')
            plt.close()

        outFile=outDir+sessID+'_mapSD'
        np.savez(outFile,mapSD=mapSD)
        

def average_trials_tWindow(sourceRoot, targetRoot, sessID, nCond, runList, analysisDir, avgFolder,\
                        motionCorrection=False,retinoAnalysis=False,\
                        percentSignalChange=False, parametricStat=False,bootstrapStat=False):
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if nCond is None:
        raise TypeError("nCond (number of conditions) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")


    # DEFINE DIRECTORIES
    inDir=analysisDir+'/MidProcOutput/'
    outDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';
    if not os.path.exists(outDir):
        os.makedirs(outDir) 

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=outDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()

    #OUTPUT TEXT FILE WITH RUN LIST
    outFile=outDir+'list_of_runs_used.txt'
    runTextFile = open(outFile, 'w+')
    runTextFile.write('LIST OF RUNS\n')
    runTextFile.write('\n')
    for run in runList:
        runTextFile.write(str(run)+' ')
    runTextFile.close()


    if motionCorrection:
        #GET MOTION CORRECTED BOUNDARIES
        motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
        motionFileDir=motionDir+'Registration/'
        inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
        f=np.load(inFile)
        boundaries=f['boundaries']
        edgeUp=boundaries[0]
        edgeDown=boundaries[1]
        edgeLeft=boundaries[2]
        edgeRight=boundaries[3]
        #GET MOTION CORRECTED FRAME SIZE
        szY=edgeDown-edgeUp
        szX=edgeRight-edgeLeft

    else:
        #GET REFFERNCE FRAME
        imRef=get_reference_frame(sourceRoot,sessID,runList[0])
        szY,szX=imRef.shape


    for cond in range(nCond):
        print('cond = '+str(cond+1))
        # RETRIEVE DATA

        for (runCount,run) in enumerate(runList):
            print('run = '+str(run))

            inFile=inDir+sessID+'_run'+str(run)+'.npz'
            if os.path.isfile(inFile):
                f=np.load(inFile)
                stimBlockCond=f['stimBlockCond']
                baseResp=f['baseResp']
                stimResp=f['stimResp']
                frameRate=f['frameRate']
            else:
                inFile=inDir+sessID+'_run'+str(run)+'_miscellaneous.npz'
                f=np.load(inFile)
                stimBlockCond=f['stimBlockCond']
                nTrials=np.size(stimBlockCond)
                frameRate=f['frameRate']
                #LOAD BY PARTS
                inFile=glob.glob(inDir+sessID+'_run'+str(run)+'_part1*')[0]
                f=np.load(inFile)
                baseRespTmp=f['baseResp']
                stimRespTmp=f['stimResp']
                startBlock=f['startTrial']
                endBlock=f['endTrial']
                nParts=f['nParts']

                nPix,d,stimPts=stimRespTmp.shape
                nPix,d,basePts=baseRespTmp.shape
                baseResp=np.zeros(nPix,nTrials,basePts)
                stimResp=np.zeros(nPix,nTrials,stimPts)

                baseResp[:,startTrial:endTrial,:]=baseRespTmp
                stimResp[:,startTrial:endTrial,:]=stimRespTmp

                for p in range(1,nParts):
                    inFile=glob.glob(inDir+sessID+'_run'+str(run)+'_part'+str(p+1)+'*')[0]
                    f=np.load(inFile)
                    baseRespTmp=f['baseResp']
                    stimRespTmp=f['stimResp']
                    startBlock=f['startTrial']
                    endBlock=f['endTrial']
                    baseResp[:,startTrial:endTrial,:]=baseRespTmp
                    stimResp[:,startTrial:endTrial,:]=stimRespTmp

            nPix,nTrials,stimPts=np.shape(stimResp)
            nPix,nTrials,basePts=np.shape(baseResp)

            condInd=np.where(stimBlockCond==cond+1)[0]

            if runCount==0:
                trialsPerCond=(len(runList)*nTrials)/nCond#assume all runs have same number of trials per condition

                stimRespAll=np.zeros((nPix,trialsPerCond,stimPts))
                baseRespAll=np.zeros((nPix,trialsPerCond,basePts))
                startInd=0

            endInd=startInd+len(condInd)

            stimRespAll[:,startInd:endInd,:]=stimResp[:,condInd,:]
            baseRespAll[:,startInd:endInd,:]=baseResp[:,condInd,:]
            startInd=endInd
        #WRITE NUMBER OF TRIALS PER CONDITION TO TEXT FILE
        if cond == 0:
            outFile=outDir+'total_trials_per_condition.txt'
            trialTextFile = open(outFile, 'w+')
            trialTextFile.write('NUMBER OF TRIALS\n')
            trialTextFile.write('\n')

        trialTextFile.write('condition '+str(cond+1)+' '+str(np.shape(stimRespAll)[1])+' trials\n')

        # AVERAGE OVER TIME
        if cond==0:

            baseRespTimeMean=np.zeros((nPix,trialsPerCond,nCond))
            stimRespTimeMean=np.zeros((nPix,trialsPerCond,nCond))
        baseRespTimeMean[:,:,cond]=np.mean(baseRespAll,2)#avg over time
        stimRespTimeMean[:,:,cond]=np.mean(stimRespAll,2)#avg over time

        if retinoAnalysis:
            #Z-SCORE RESPONSE RELATIVE TO BASELINE
            baseRespMean=np.mean(baseRespAll,1)#avg over trials
            stimRespMean=np.mean(stimRespAll,1)#avg over trials

            baseMean=np.mean(baseRespMean,1)#avg over time
            baseMean=np.tile(np.expand_dims(baseMean,1),(1,stimPts))
            baseSD=np.std(baseRespMean,1)#avg over time
            baseSD=np.tile(np.expand_dims(baseSD,1),(1,stimPts))


            zScoredStimResp=np.true_divide(stimRespMean-baseMean,baseSD)#z-score 
            observedResp=np.mean(zScoredStimResp,1)#avg over time

            respMap=np.reshape(observedResp,(szY,szX))
            outFile=outDir+sessID+'_condition'+str(cond)+'_zScoreMap'
            np.savez(outFile,respMap=respMap);

    if not retinoAnalysis:
        #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
        inFile=targetRoot+'contrastDic.npz'
        f=np.load(inFile)
        contrastDic=f['contrastDic']

        #Get Percent Signal Change for All Trials
        PSC_AllTrials=np.true_divide(np.subtract(stimRespTimeMean,baseRespTimeMean),baseRespTimeMean)
        outFile=outDir+sessID+'_PSC_AllTrials'
        np.savez(outFile,PSC_AllTrials=PSC_AllTrials,nPix=nPix,nTrials=trialsPerCond,nCond=nCond)

        if percentSignalChange:
            # Percent Signal Change
            PSC=np.mean(PSC_AllTrials*100,1)#average over trials
            # ->contrast conditions
            for c in range(len(contrastDic)):
                if contrastDic[c]['minus']==[0]:
                    PSCpix=np.mean(PSC[:,np.subtract(contrastDic[c]['plus'],1)],1)
                    positiveBaselineMask=None
                else:
                    PSC_plus=np.mean(PSC[:,np.subtract(contrastDic[c]['plus'],1)],1)
                    PSC_minus=np.mean(PSC[:,np.subtract(contrastDic[c]['minus'],1)],1)
                    positiveBaselineMask=np.logical_or(PSC_plus>=0,PSC_minus>=0).reshape((szY,szX))

                    PSCpix=PSC_plus-PSC_minus
                PSCmap=np.reshape(PSCpix,(szY,szX))
                outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_PSCmap'
                np.savez(outFile,PSCmap=PSCmap,positiveBaselineMask=positiveBaselineMask);

        if parametricStat:
            # SPM
            for c in range(len(contrastDic)):
                if contrastDic[c]['minus']==[0]:
                    plusMat=np.mean(PSC_AllTrials[:,:,np.subtract(contrastDic[c]['plus'],1)],2)
                    minusMat=np.zeros(np.shape(plusMat))
                    positiveBaselineMask=None

                else:
                    PSC=np.mean(PSC_AllTrials*100,1)#average over trials
    
                    PSC_plus=np.mean(PSC[:,np.subtract(contrastDic[c]['plus'],1)],1)
                    plusMat=np.mean(PSC_AllTrials[:,:,np.subtract(contrastDic[c]['plus'],1)],2)

                    PSC_minus=np.mean(PSC[:,np.subtract(contrastDic[c]['minus'],1)],1)
                    minusMat=np.mean(PSC_AllTrials[:,:,np.subtract(contrastDic[c]['minus'],1)],2)
                    positiveBaselineMask=np.logical_or(PSC_plus>=0,PSC_minus>=0).reshape((szY,szX))

                tStatPix,pValPix=stats.ttest_rel(plusMat,minusMat,1)
                tStatMap=np.reshape(tStatPix,(szY,szX))
                pValMap=np.reshape(pValPix,(szY,szX))
                outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_SPMmap'
                np.savez(outFile,tStatMap=tStatMap,pValMap=pValMap,positiveBaselineMask=positiveBaselineMask);

        if bootstrapStat:
            #Non-parametric stat
            for c in range(len(contrastDic)):
                if contrastDic[c]['minus']==[0]:
                    plusMat=np.mean(PSC_AllTrials[:,:,np.subtract(contrastDic[c]['plus'],1)],2)
                    minusMat=np.zeros(np.shape(plusMat))

                else:
                    PSC=np.mean(PSC_AllTrials*100,1)#average over trials
    
                    PSC_plus=np.mean(PSC[:,np.subtract(contrastDic[c]['plus'],1)],1)
                    plusMat=np.mean(PSC_AllTrials[:,:,np.subtract(contrastDic[c]['plus'],1)],2)
                    plusMat[PSC_plus<0,:]=0
                    
                    PSC_minus=np.mean(PSC[:,np.subtract(contrastDic[c]['minus'],1)],1)
                    minusMat=np.mean(PSC_AllTrials[:,:,np.subtract(contrastDic[c]['minus'],1)],2)
                    minusMat[PSC_minus<0,:]=0

                #GET OBSERVED MEAN DIFFERENCE
                empDiff=np.mean(plusMat,1)-np.mean(minusMat,1);
                signPix=np.sign(empDiff)

                nPix,nTrialsPlus=np.shape(plusMat)
                nPix,nTrialsMinus=np.shape(minusMat)

                pValPix=np.zeros(nPix)
                nReps=1
                nSimsPerRep=10**5
                totalSims=nReps*nSimsPerRep
                for pix in range(nPix):
                    respValues=np.concatenate((plusMat[pix,:],minusMat[pix,:]),0)
                    aboveMeanCount=0
                    for rep in range(nReps):
                        #SAMPLE WITH REPLACEMENT
                        permIndPlus=np.random.randint(0,nTrialsPlus+nTrialsMinus,(nSimsPerRep,nTrialsPlus))
                        permIndMinus=np.random.randint(0,nTrialsPlus+nTrialsMinus,(nSimsPerRep,nTrialsMinus))

                        simPlusValues=respValues[permIndPlus]
                        simMinusValues=respValues[permIndMinus]

                        simDiff=np.mean(simPlusValues,1)-np.mean(simMinusValues,1)

                        aboveMeanCount=aboveMeanCount+sum(np.absolute(simDiff)>np.absolute(empDiff[pix]))#double-sided test

                    pValPix[pix]=np.true_divide(aboveMeanCount,totalSims)

                pValPix[pValPix==0]=np.true_divide(1,totalSims)
                pValMap=np.reshape(pValPix,(szY,szX))
                signMap=np.reshape(signPix,(szY,szX))
                outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_bootstrapMap'
                np.savez(outFile,signMap=signMap,pValMap=pValMap)
                
    trialTextFile.close()
                    

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ROI DEFINITION FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #


def make_ROI_single_pixel(ROIdir, dataDir, sessID, nPix=100):

    #Cesar Echavarria 11/2016
    

    #LOAD IN STANDARD DEV MAP
    inFile=dataDir+sessID+'_mapSD.npz'
    f=np.load(inFile)
    mapSD=f['mapSD']

    #GET PIXELS WITH MOST VARIANCE (MAYBE BEST SIGNAL)
    szY,szX,nCond=np.shape(mapSD)
    mapSD=np.sum(mapSD,2)
    pixSD=np.reshape(mapSD,(szY*szX))

    pixList=np.unique(np.argsort(pixSD).argsort()[::-1][:nPix]%(szY*szX))

    
    locY,locX=np.unravel_index(pixList, (szY,szX))

    #SAVE ROI INDICES
    for (pixCount,pix) in enumerate(pixList):
        outFile=ROIdir+'/pixel'+str(pixCount)+'_ROI'

        np.savez(outFile,ROIind=pix,locX=locX[pixCount],locY=locY[pixCount])

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# VISUALIZATION FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #


        
def plot_ROI_average_timecourse(figOutDir, sessID, ROIlabel, stimRespROI, frameTimes,\
                                condLabels, stimStartT, stimEndT, percentSignalChange=True):
    #Cesar Echavarria 11/2016
    
    #GET PARAMETERS FROM DATA STRUCTURE
    nCond=np.shape(stimRespROI)[0]
    
    #MAKE PLOTS
    colorList='bgrmkc'
    legHand=[]
    fig=plt.figure()
    for c in range(nCond):
        plt.plot(frameTimes,stimRespROI[c,:],colorList[c],linewidth=2)

        
        legHand.append(mlines.Line2D([], [], color=colorList[c], marker='_',
                              markersize=15, label=condLabels[c]))
    if nCond>3:
        plt.legend(handles=legHand,fontsize=8,loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3)
    else:
        plt.legend(handles=legHand)

    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    if nCond>3:
        axes.set_ylim(ymin,ymax+(.2*ymax))
        ymin, ymax = axes.get_ylim()
    plt.axvline(x=stimStartT, ymin=ymin, ymax = ymax, linewidth=1, color='k')
    plt.axvline(x=stimEndT, ymin=ymin, ymax = ymax, linewidth=1, color='r')
    if ymin<=0:
        xmin, xmax = axes.get_xlim()
        plt.axhline(xmin=xmin, xmax=xmax , y= 0, linewidth=1, color='k')

    fig.suptitle(ROIlabel+' Response', fontsize=16)
    plt.xlabel('Time (secs)',fontsize=16)
    if percentSignalChange:
        plt.ylabel('PSC',fontsize=16)
    else:
        plt.ylabel('Pixel Value',fontsize=16)

    plt.savefig(figOutDir+sessID+'_'+ROIlabel+'_timecourse_trialAvg.png')

    plt.close()
    
def plot_ROI_trial_timecourse(figFile, ROIlabel, stimRespROI, frameTimes,\
                                condLabels, stimStartT, stimEndT, percentSignalChange=True):
    #Cesar Echavarria 11/2016
    
    #GET PARAMETERS FROM DATA STRUCTURE
    nCond,nTrials,respPts=np.shape(stimRespROI)
    
    permList=np.random.permutation(nCond*nTrials)
    
    #MAKE PLOTS
    condInLegend=[]
    colorList='bgrmkc'
    legHand=[]
    fig=plt.figure()
    for permInd in permList:
        cond=int(np.floor(np.true_divide(permInd,nTrials)))
        trial=int(permInd%nTrials)
        plt.plot(frameTimes,stimRespROI[cond,trial,:],colorList[cond],linewidth=1)
        if cond not in condInLegend:
            legHand.append(mlines.Line2D([], [], color=colorList[cond], marker='_',
                                  markersize=15, label=condLabels[cond]))
            condInLegend.append(cond)
    if nCond>3:
        plt.legend(handles=legHand,fontsize=8,loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3)
    else:
        plt.legend(handles=legHand)

    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    if nCond>3:
        axes.set_ylim(ymin,ymax+(.2*ymax))
        ymin, ymax = axes.get_ylim()
    plt.axvline(x=stimStartT, ymin=ymin, ymax = ymax, linewidth=1, color='k')
    plt.axvline(x=stimEndT, ymin=ymin, ymax = ymax, linewidth=1, color='r')
    if ymin<=0:
        xmin, xmax = axes.get_xlim()
        plt.axhline(xmin=xmin, xmax=xmax , y= 0, linewidth=1, color='k')

    fig.suptitle(ROIlabel+' Response', fontsize=16)
    plt.xlabel('Time (secs)',fontsize=16)
    if percentSignalChange:
        plt.ylabel('PSC',fontsize=16)
    else:
        plt.ylabel('Pixel Value',fontsize=16)
    plt.savefig(figFile)
    plt.close()
 
    
def plot_single_pixel_timecourse(targetRoot, sessID, analysisDir, avgFolder,\
                                 makeROIs=False, stimDur=5, nPix=100,plotMode='Average',subsetSize=10,\
                                average_conditions=False,average_name=None,newCond=None,condLabels=None):
    
    #Cesar Echavarria 11/2016
    
    #DEFINE SOME DIRECTORIES
    inDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';
    
    ROIdir=analysisDir+'/ROIs/singlePixel'
    if not os.path.exists(ROIdir):
        os.makedirs(ROIdir) 
    if plotMode=='Average':
        figOutDir=analysisDir+'/Figures/'+avgFolder+'/timecourse_average_singlePixel/'
    elif plotMode=='All':
        figOutDir=analysisDir+'/Figures/'+avgFolder+'/timecourse_allTrials_singlePixel/'
    elif plotMode=='Subset':
        figOutDir=analysisDir+'/Figures/'+avgFolder+'/timecourse_trialSubset_singlePixel/'
    if average_conditions:
        figOutDir=figOutDir+average_name+'/'
        
    if not os.path.exists(figOutDir):
        os.makedirs(figOutDir)
    if makeROIs:
        print('Making single pixel ROIs....')
        #GENERATE SINGLE PIXEL ROIs
        make_ROI_single_pixel(ROIdir, inDir, sessID, nPix)
    
    #LOAD MISCELlANEOUS
    inFile=inDir+sessID+'_timecourse_miscelleanous.npz'
    f=np.load(inFile)
    startT=f['timecourse_startT']
    stimStartT=f['baseline_endT']
    frameRate=f['frameRate']
    percentSignalChange=f['percentSignalChange']
    nCond=f['nCond']
    nPix=f['nPix']
    nTrials=f['nTrials']
    respPts=f['respPts']
    
    sampRate=np.true_divide(1,frameRate)
    
    if plotMode=='Average':
        #LOAD IN TIME COURSE DATA
        inFile=inDir+sessID+'_timecourse_trialAvg.npz'
        f=np.load(inFile)
        stimRespMean=f['stimRespMean']
        

        #GET SOME INFO ABOUT DATA
        nPix,nCond,respPts=np.shape(stimRespMean)
        frameTimes=np.linspace(startT,startT+(respPts)*sampRate,respPts)
        if average_conditions:
            allNewCond=np.sort(np.unique(newCond))
            nCond=np.size(allNewCond)
            stimRespMeanTmp=np.zeros((nPix,nCond,respPts))
            for newC in allNewCond:                
                condInd=np.where(newCond==newC)[0]
                if np.size(condInd)>1:
                    stimRespMeanTmp[:,newC-1,:]=np.mean(stimRespMean[:,condInd,:],1)
                else:
                    stimRespMeanTmp[:,newC-1,:]=stimRespMean[:,condInd,:]
            stimRespMean=stimRespMeanTmp
            del stimRespMeanTmp
        else:
            condLabels=contrast_dictionary_to_labels(targetRoot,nCond)
        
    elif plotMode=='All' or plotMode=='Subset':
        #LOAD IN TIME COURSE DATA
        responseAll=np.zeros((nPix,nCond,nTrials,respPts))
        for cond in range(nCond):
            inFile=inDir+sessID+'_'+str(cond)+'_timecourse_allTrials.npz'
            if os.path.isfile(inFile):
                f=np.load(inFile)
                responseAll[:,cond,:,:]=f['responseAll']
            else:
                #LOAD BY PARTS
                inFile=glob.glob(inDir+sessID+'_'+str(cond)+'_timecourse_allTrials_part1*')[0]
                f=np.load(inFile)
                tmp=f['responseAll']
                startTrial=f['startTrial']
                endTrial=f['endTrial']
                nParts=f['nParts']

                responseAll[:,cond,startTrial:endTrial,:]=tmp

                for p in range(1,nParts):
                    inFile=glob.glob(inDir+sessID+'_'+str(cond)+'_timecourse_allTrials_part'+str(p+1)+'*')[0]
                    f=np.load(inFile)
                    tmp=f['responseAll']
                    startTrial=f['startTrial']
                    endTrial=f['endTrial']
                    responseAll[:,cond,startTrial:endTrial,:]=tmp

        #GET SOME INFO ABOUT DATA
        frameTimes=np.linspace(startT,startT+(respPts)*sampRate,respPts)
        if average_conditions:
            allNewCond=np.sort(np.unique(newCond))
            nCond=np.size(allNewCond)
            responseAllTmp=np.zeros((nPix,nCond,nTrials,respPts))
            for newC in allNewCond:                
                condInd=np.where(newCond==newC)[0]
                if np.size(condInd)>1:
                    responseAllTmp[:,newC-1,:,:]=np.mean(responseAll[:,condInd,:,:],1)
                else:
                    responseAllTmp[:,newC-1,:,:]=stimRespMean[:,condInd,:,:]
            responseAll=responseAllTmp
            del responseAllTmp
            
        else:
            condLabels=contrast_dictionary_to_labels(targetRoot,nCond)

    #GET ROI LIST
    listROI=glob.glob(ROIdir+'/*.npz')

    #LOAD ROI INDICES AND DETAILS
    if plotMode=='Average':
            print('Plotting average timecourse within single pixel ROIs....')
    elif plotMode=='All':
         print('Plotting timecourse for all trials within single pixel ROIs....')
    elif plotMode=='Subset':
         print('Plotting timecourse for a subset of trials within single pixel ROIs....')
    for roi in listROI:
        # load file and variables
        f=np.load(roi)
        ROI_ind=f['ROIind']
        locX=f['locX']
        locY=f['locY']

        ROI_label='Pixel_X_'+str(locX)+'_Y_'+str(locY)
        if plotMode=='Average':
            #AVERAGE ACROSS PIXELS IN ROI
            if np.size(ROI_ind)>1:
                stimRespROI=np.mean(stimRespMean[ROI_ind,:,:],0)
            else:
                stimRespROI=stimRespMean[ROI_ind,:,:]

            #PLOT ROI
            plot_ROI_average_timecourse(figOutDir, sessID, ROI_label, stimRespROI, frameTimes,\
                                    condLabels, stimStartT, stimStartT+stimDur, percentSignalChange)
        elif plotMode=='All':
            #AVERAGE ACROSS PIXELS IN ROI
            if np.size(ROI_ind)>1:
                stimRespROI=np.mean(responseAll[ROI_ind,:,:,:],0)
            else:
                stimRespROI=responseAll[ROI_ind,:,:,:]

            #PLOT ROI
            figFile=figOutDir+sessID+'_'+ROI_label+'_timecourse_allTrials.png'
            plot_ROI_trial_timecourse(figFile, ROI_label, stimRespROI, frameTimes,\
                                    condLabels, stimStartT, stimStartT+stimDur, percentSignalChange)
        elif plotMode=='Subset':
            #AVERAGE ACROSS PIXELS IN ROI
            if np.size(ROI_ind)>1:
                stimRespROI=np.mean(responseAll[ROI_ind,:,:,:],0)
            else:
                stimRespROI=responseAll[ROI_ind,:,:,:]
            
            #CHOOSE SUBSET OF TRIALS
            stimRespROI_subset=np.zeros((nCond,subsetSize,respPts))
            for cond in range(nCond):
                permList=np.random.permutation(int(nTrials))
                stimRespROI_subset[cond,:,:]=stimRespROI[cond,permList[0:subsetSize],:]
            

            #PLOT ROI
            figFile=figOutDir+sessID+'_'+ROI_label+'_timecourse_'+str(subsetSize)+'Trials.png'
            plot_ROI_trial_timecourse(figFile, ROI_label, stimRespROI_subset, frameTimes,\
                                    condLabels, stimStartT, stimStartT+stimDur, percentSignalChange)
        
        
def make_movie_from_stack(rootDir,frameStack,frameRate=24,movFile='test.mp4'):
    #Cesar Echavarria 10/2016
    
    #CHECK INPUTS
    if frameStack is None:
        raise TypeError("no frame stack provided!")
    if np.amax(frameStack) > 1:
        raise TypeError("frame stack values must be in the range 0-1")
    
    
    #GET STACK INFO
    szY,szX,nFrames=np.shape(frameStack)



    #MAKE TEMP FOLDER
    tmpDir=rootDir+'/tmp/'
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)


    #WRITE TO TEMP FOLDER
    for i in range(0,nFrames):
        outFile=tmpDir+str(i)+'.png'
        frame0=frameStack[:,:,i]
        if szY%2 == 1:
            frame0=np.delete(frame0,0,0)
        if szX%2 == 1:
            frame0=np.delete(frame0,0,1)

        frame=np.uint8(frame0*255)
        misc.imsave(outFile,frame)

    #WRITE VIDEO
    cmd='ffmpeg -y -r '+'%.3f'%frameRate+' -i '+tmpDir+'%d.png -vcodec libx264 -f mp4 -pix_fmt yuv420p '+movFile
    os.system(cmd)


    #GET RID OF TEMP FOLDER
    shutil.rmtree(tmpDir)
    
                                    
def make_movie_from_stack_mark_stim(rootDir,frameStack,frameRate=24,onFrame=0,offFrame=None,movFile='test.mpr'):
    #Cesar Echavarria 10/2016
    
    #CHECK INPUTS
    if frameStack is None:
        raise TypeError("no frame stack provided!")
    if np.amax(frameStack) > 1:
        raise TypeError("frame stack values must be in the range 0-1")
    
    
    #GET STACK INFO
    szY,szX,nFrames=np.shape(frameStack)

  
    #MAKE TEMP FOLDER
    tmpDir=rootDir+'/tmp/'
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)


    #WRITE TO TEMP FOLDER
    for i in range(0,nFrames):
        outFile=tmpDir+str(i)+'.png'
        frame=np.copy(frameStack[:,:,i])
        if offFrame is None:
            if i >= onFrame:
                frame=np.pad(frame,(6,), 'constant', constant_values=(1,))
            else:
                frame=np.pad(frame,(6,), 'constant', constant_values=(0,))
        else:
            if i >= onFrame and i<offFrame:
                frame=np.pad(frame,(6,), 'constant', constant_values=(1,))
            else:
                frame=np.pad(frame,(6,), 'constant', constant_values=(0,))

        if frame.shape[0]%2 == 1:
            frame=np.delete(frame,0,0)
        if frame.shape[1]%2 == 1:
            frame=np.delete(frame,0,1)
        frame=np.uint8(frame*255)
        misc.imsave(outFile,frame)

    #WRITE VIDEO
    cmd='ffmpeg -y -r '+'%.3f'%frameRate+' -i '+tmpDir+'%d.png -vcodec libx264 -pix_fmt yuv420p '+movFile
    os.system(cmd)


    #GET RID OF TEMP FOLDER
    shutil.rmtree(tmpDir)  
    
def make_movie_from_timecourse_average(sourceRoot, targetRoot, sessID, frameRate, analysisDir, avgFolder,\
    motionCorrection=False, stimDur=5):
    #Cesar Echavarria 10/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if frameRate is None:
        raise TypeError("frameRate not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
        
    # DEFINE DIRECTORIES
    inDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';

    movieOutDir=analysisDir+'/Movies/'+avgFolder+'/'
    if not os.path.exists(movieOutDir):
        os.makedirs(movieOutDir) 

    if motionCorrection:#COULD BE MADE INTO A FXN (CALLED MULTIPLE TIMES)
        #GET MOTION CORRECTED BOUNDARIES
        motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
        motionFileDir=motionDir+'Registration/'
        inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
        f=np.load(inFile)
        boundaries=f['boundaries']
        edgeUp=boundaries[0]
        edgeDown=boundaries[1]
        edgeLeft=boundaries[2]
        edgeRight=boundaries[3]
        #GET MOTION CORRECTED FRAME SIZE
        szY=edgeDown-edgeUp
        szX=edgeRight-edgeLeft
    else:
        #GET REFFERNCE FRAME
        imRef=get_reference_frame(sourceRoot,sessID)
        szY,szX=imRef.shape


    #LOAD IN TIME COURSE AND RELATED PARAMETERS
    
    inFile=inDir+sessID+'_timecourse_miscelleanous.npz'
    f=np.load(inFile)
    startT=f['timecourse_startT']
    baseline_endT=f['baseline_endT']
    frameRate=f['frameRate']
    sampRate=np.true_divide(1,frameRate)

    inFile=inDir+sessID+'_timecourse_trialAvg.npz'
    f=np.load(inFile)
    stimRespMean=f['stimRespMean']
    

    nPix,nCond,respPts=np.shape(stimRespMean)

    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=targetRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    #GET SOME INFO ABOUT DATA
    nPix,nCond,respPts=np.shape(stimRespMean)
    frameTimes=np.linspace(startT,startT+((respPts)*sampRate),respPts)

    onFrame=np.where(frameTimes>baseline_endT)[0][0]
    offFrame=np.where(frameTimes>stimDur)[0][0]



    for c in range(0,nCond):
        #NORMALIZE ARRAY
        frameArray=stimRespMean[:,c,:]
        arrayMin=np.amin(frameArray)
        arrayMax=np.amax(frameArray)
        frameArray=np.true_divide((frameArray-arrayMin),(arrayMax-arrayMin))

        #RESHAPE
        frameArray=np.reshape(frameArray,(szY,szX,respPts))

        #MAKE MOVIE
        endInd=contrastDic[c]['name'].index('_')
        outFile=movieOutDir+sessID+'_'+contrastDic[c]['name'][0:endInd]+'_movie.mp4'
        #make_movie_from_stack(frameArray,frameRate=frameRate,movFile=outFile)
        make_movie_from_stack_mark_stim(movieOutDir,frameArray,frameRate=frameRate,onFrame=onFrame,offFrame=offFrame,movFile=outFile)


def generate_PSC_map(sourceRoot,sessID,analysisDir,avgFolder, motionCorrection=False,\
    manualThresh = False, threshMin = 1, threshMax = 3, contrastName=None,mask = None):
    #Cesar Echavarria - 10/2016
    
    #VERIFY WE GOT NECESSARY VALUES
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")

        
    # DEFINE DIRECTORIES
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'

    resultsDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/'
    if manualThresh:
        outDir=analysisDir+'/Figures/'+avgFolder+'/PSCcontrast/manualThresh/'
    else:
        outDir=analysisDir+'/Figures/'+avgFolder+'/PSCcontrast/'

    if not os.path.exists(outDir):
        os.makedirs(outDir) 


    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;



    funcOverlayTmp=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlayTmp)

    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=sourceRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    #GET MASK IF NECESSARY
    if mask is not None:
        maskFile=sourceRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
        f=np.load(maskFile)
        maskM=f['maskM']

    if contrastName is not None:
        c=0
        while c < len(contrastDic):
            if contrastDic[c]['name']==contrastName:
                contrastDic=[contrastDic[c]]
                break
            c=c+1

    for c in range(len(contrastDic)):
        # LOAD IN MAP WITH PSC PER PIXEL
        mapFile=resultsDir+sessID+'_'+contrastDic[c]['name']+'_PSCmap.npz'
        f=np.load(mapFile)
        PSCmap=f['PSCmap']
        positiveBaselineMask=f['positiveBaselineMask']
        PSCmap=np.nan_to_num(PSCmap)
        if positiveBaselineMask is not None:
            PSCmap[positiveBaselineMask==0]=0

        if motionCorrection:
            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']
            padDown=int(boundaries[0])
            padUp=int(szY-boundaries[1])
            padLeft=int(boundaries[2])
            padRight=int(szX-boundaries[3])
            PSCmap=np.pad(PSCmap,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))



        #APPLY MASK, IF INDICATED
        if mask is not None:
            PSCmap=PSCmap*maskM

        funcOverlay=np.copy(funcOverlayTmp)

        #AUTOMATIC THRESHOLD 
        if not manualThresh:
            pMin=.2
            pMax=.8

            threshMin=np.around(pMin*np.max(np.absolute(PSCmap)),2)
            if threshMin>np.max(np.absolute(PSCmap)):
                threshMax=threshMin
            else:
                threshMax=np.around(pMax*np.max(np.absolute(PSCmap)),2)
        threshList=np.linspace(threshMin,threshMax,8);

        #DEFINE COLOR MAP
        colorOverlayNeg=np.array([[0,0,.25],[0,0,.5],[0,0,.75],[0,0,1],[0,.25,1],[0,.5,1],[0,.75,1],[0,1,1]]);
        colorOverlayPos=np.array([[.25,0,0],[.5,0,0],[.75,0,0],[1,0,0],[1,.25,0],[1,.5,0],[1,.75,0],[1,1,0]]);
        cMap=np.vstack((np.flipud(colorOverlayNeg),np.array([.5,.5,.5]),colorOverlayPos))

        imSat=.5#SET SATURATION OF COLORS

        #COOL COLORS
        for t in range(len(threshList)-1):
            colorInd=np.where(np.logical_and(PSCmap>-threshList[t+1],PSCmap<=-threshList[t]))

            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=(np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1))-imTemp
                  #  imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=PSCmap<=-threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
            #imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            funcOverlay[:,:,channel]=imTemp[:,:]

        #WARM COLORS
        for t in range(len(threshList)-1):
            colorInd=np.logical_and(PSCmap<threshList[t+1],PSCmap>=threshList[t])
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                #  imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=PSCmap>=threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            #imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
            funcOverlay[:,:,channel]=imTemp[:,:]

        #MAKE AND SAVE IMAGE WITH COLOR BAR
        mycmap = array2cmap(cMap)
        plt.imshow(np.uint8(funcOverlay),cmap=mycmap)
        plt.axis('off')

        nPos=np.shape(colorOverlayPos)[0]
        nNeg=np.shape(colorOverlayNeg)[0]
        nSteps=nPos+nNeg+1

        t1Loc=np.true_divide(1,nSteps)
        t2Loc=(np.true_divide(1,nSteps)*nPos)-np.true_divide(t1Loc,2)
        t3Loc=t2Loc+(2*np.true_divide(1,nSteps))
        t4Loc=1-np.true_divide(1,nSteps)
        tickLoc=np.array([t1Loc,t2Loc,t3Loc,t4Loc])
        tickLoc=tickLoc*np.max(np.uint8(funcOverlay))

        t1Label=-np.max(threshList)
        t2Label=-np.min(threshList)
        t3Label=np.min(threshList)
        t4Label=np.max(threshList)
        tickLabels=np.around(np.array([t1Label,t2Label,t3Label,t4Label]),4)

        cb = plt.colorbar()
        cb.set_ticks(tickLoc)
        cb.set_ticklabels(tickLabels)

        if mask is None:
            plt.savefig(outDir+sessID+'_'+contrastDic[c]['name']+'_withColorbar.png')
            outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_image.png'
        else:
            plt.savefig(outDir+sessID+'_'+contrastDic[c]['name']+'_mask_'+mask+'_withColorbar.png')
            outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_mask_'+mask+'_image.png'
        misc.imsave(outFile,np.uint8(funcOverlay))

        plt.clf()
        
def generate_SPM(sourceRoot,sessID,analysisDir,avgFolder,motionCorrection=False,\
    manualThresh = False, threshMin = 1.35, threshMax = 3,contrastName=None, mask = None):
    #Cesar Echavarria - 10/2016
    
    #VERIFY WE GOT NECESSARY VALUES
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
        
    # DEFINE DIRECTORIES
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'
    resultsDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/'


    outDir=analysisDir+'/Figures/'+avgFolder+'/SPMmaps/'

    if not os.path.exists(outDir):
        os.makedirs(outDir) 



    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;

    funcOverlayTmp=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlayTmp)



    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=sourceRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    #GET MASK IF NECESSARY
    if mask is not None:
        maskFile=sourceRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
        f=np.load(maskFile)
        maskM=f['maskM']
    
    if contrastName is not None:
        c=0
        while c < len(contrastDic):
            if contrastDic[c]['name']==contrastName:
                contrastDic=[contrastDic[c]]
                break
            c=c+1
    for c in range(len(contrastDic)):
        # LOAD IN MAP WITH PSC PER PIXEL
        mapFile=resultsDir+sessID+'_'+contrastDic[c]['name']+'_SPMmap.npz'
        f=np.load(mapFile)
        pValMap=f['pValMap']
        tStatMap=f['tStatMap']
        positiveBaselineMask=f['positiveBaselineMask']


        SPMmap=(-np.log10(pValMap))*np.sign(tStatMap)
        SPMmap=np.nan_to_num(SPMmap)
        if positiveBaselineMask is not None:
            SPMmap[positiveBaselineMask==0]=0

        if motionCorrection:
            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']
            padDown=int(boundaries[0])
            padUp=int(szY-boundaries[1])
            padLeft=int(boundaries[2])
            padRight=int(szX-boundaries[3])
            SPMmap=np.pad(SPMmap,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

        #APPLY MASK, IF INDICATED
        if mask is not None:
            SPMmap=SPMmap*maskM

        funcOverlay=np.copy(funcOverlayTmp)
        if not manualThresh:
            #AUTOMATIC THRESHOLD 
            pMin=.2
            pMax=.8
            
            threshMin=np.round(pMin*np.max(np.absolute(SPMmap)))
            
            if threshMin<1.35:
                threshMin=1.35
            if threshMin>np.round(pMax*np.max(np.absolute(SPMmap))):
                threshMax=np.ceil(threshMin)
            else:
                threshMax=np.round(pMax*np.max(np.absolute(SPMmap)))
            
        threshList=np.linspace(threshMin,threshMax,8)
        
        #DEFINE COLOR MAP
        colorOverlayNeg=np.array([[0,0,.25],[0,0,.5],[0,0,.75],[0,0,1],[0,.25,1],[0,.5,1],[0,.75,1],[0,1,1]]);
        colorOverlayPos=np.array([[.25,0,0],[.5,0,0],[.75,0,0],[1,0,0],[1,.25,0],[1,.5,0],[1,.75,0],[1,1,0]]);
        cMap=np.vstack((np.flipud(colorOverlayNeg),np.array([.5,.5,.5]),colorOverlayPos))

        imSat=0.5#SET SATURATION OF CLORS
        #COOL COLORS
        for t in range(len(threshList)-1):
            colorInd=np.where(np.logical_and(SPMmap>-threshList[t+1],SPMmap<=-threshList[t]))
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
                  #  imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                funcOverlay[:,:,channel]=np.copy(imTemp)

        colorInd=SPMmap<=-threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
            #imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            funcOverlay[:,:,channel]=imTemp[:,:]

       ## WARM COLORS
        for t in range(len(threshList)-1):
            colorInd=np.logical_and(SPMmap<threshList[t+1],SPMmap>=threshList[t])
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                #  imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=SPMmap>=threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)

            #imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
            funcOverlay[:,:,channel]=imTemp[:,:]    
        #MAKE AND SAVE IMAGE WITH COLOR BAR
        mycmap = array2cmap(cMap)
        plt.imshow(np.uint8(funcOverlay),cmap=mycmap)
        plt.axis('off')

        nPos=np.shape(colorOverlayPos)[0]
        nNeg=np.shape(colorOverlayNeg)[0]
        nSteps=nPos+nNeg+1

        t1Loc=np.true_divide(1,nSteps)
        t2Loc=(np.true_divide(1,nSteps)*nPos)-np.true_divide(t1Loc,2)
        t3Loc=t2Loc+(2*np.true_divide(1,nSteps))
        t4Loc=1-np.true_divide(1,nSteps)
        tickLoc=np.array([t1Loc,t2Loc,t3Loc,t4Loc])
        tickLoc=tickLoc*np.max(np.uint8(funcOverlay))

        #*Priting out P-value directly is troublesome...small p-values not printed out correctly. Opting for printing out -log10(pValue)
        t1Label=-np.max(threshList)
        t2Label=-np.min(threshList)
        t3Label=np.min(threshList)
        t4Label=np.max(threshList)
        tickLabels=np.around(np.array([t1Label,t2Label,t3Label,t4Label]),4)


        cb = plt.colorbar()
        cb.set_ticks(tickLoc)
        cb.set_ticklabels(tickLabels)

        if mask is None:
            plt.savefig('%s_%s_min_%.2f_max_%.2f_withColorbar.png'%\
                        (outDir+sessID,contrastDic[c]['name'],threshMin,threshMax))

            outFile='%s_%s_min_%.2f_max_%.2f_image.png'%\
                        (outDir+sessID,contrastDic[c]['name'],threshMin,threshMax)
        else:
            plt.savefig('%s_%s_min_%.2f_max_%.2f_mask_%s_withColorbar.png'%\
                        (outDir+sessID,contrastDic[c]['name'],threshMin,threshMax,mask))

            outFile='%s_%s_min_%.2f_max_%.2f_mask_%s_image.png'%\
                        (outDir+sessID,contrastDic[c]['name'],threshMin,threshMax,mask)

        misc.imsave(outFile,np.uint8(funcOverlay))

        plt.clf()
        del funcOverlay,SPMmap


def generate_FDR_map(sourceRoot,sessID,analysisDir,avgFolder, motionCorrection=False,\
    mask = None):
    # DEFINE DIRECTORIES

    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'
    resultsDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/'

    fileOutDir=resultsDir
    figOutDir=analysisDir+'/Figures/'+avgFolder+'/qValueMaps/'

    if not os.path.exists(figOutDir):
            os.makedirs(figOutDir) 

    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;

    funcOverlayTmp=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlayTmp)


    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=sourceRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    #GET MASK IF NECESSARY
    if mask is not None:
        maskFile=sourceRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
        f=np.load(maskFile)
        maskM=f['maskM']
    else:
        maskM=None


    for c in range(len(contrastDic)):

        # LOAD IN MAP WITH PSC PER PIXEL
        mapFile=resultsDir+sessID+'_'+contrastDic[c]['name']+'_SPMmap.npz'
        f=np.load(mapFile)
        pValMap=f['pValMap']
        tStatMap=f['tStatMap']
        positiveBaselineMask=f['positiveBaselineMask']


        if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])
                pValMap=np.pad(pValMap,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))



        #ADJUST P-VALUES
        pValPix=pValMap[maskM>0].reshape(np.where(maskM>0)[1].size)
        correctedOutput=mt.multipletests(pValPix,method='fdr_bh')
        qValPix=correctedOutput[1]

        #PUT Q-VALUES INTO MAP
        FDRmap=np.zeros((szY,szX))
        FDRmap[maskM>0]=qValPix

        #SAVE VARIABLES TO FILE
        outFile=fileOutDir+sessID+'_'+contrastDic[c]['name']+'_FDRmap.npz'
        np.savez(outFile,FDRmap=FDRmap,maskM=maskM);

        #VISUALIZE MAP
        FDRmap=(-np.log10(FDRmap))
        FDRmap[FDRmap==np.inf]=0


        funcOverlay=np.copy(funcOverlayTmp)
        #AUTOMATIC THRESHOLD 
        pMin=.2
        pMax=.8

        threshMin=np.round(pMin*np.max(np.absolute(FDRmap)))

        if threshMin<1.35:
            threshMin=1.35
        if threshMin>np.round(pMax*np.max(np.absolute(FDRmap))):
            threshMax=np.ceil(threshMin)
        else:
            threshMax=np.round(pMax*np.max(np.absolute(FDRmap)))

        threshList=np.linspace(threshMin,threshMax,8)

        #DEFINE COLOR MAP
        colorOverlayNeg=np.array([[0,0,.25],[0,0,.5],[0,0,.75],[0,0,1],[0,.25,1],[0,.5,1],[0,.75,1],[0,1,1]]);
        colorOverlayPos=np.array([[.25,0,0],[.5,0,0],[.75,0,0],[1,0,0],[1,.25,0],[1,.5,0],[1,.75,0],[1,1,0]]);
        cMap=np.vstack((np.flipud(colorOverlayNeg),np.array([.5,.5,.5]),colorOverlayPos))

        imSat=0.5#SET SATURATION OF CLORS
        #COOL COLORS
        for t in range(len(threshList)-1):
            colorInd=np.where(np.logical_and(FDRmap>-threshList[t+1],FDRmap<=-threshList[t]))
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
                  #  imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                funcOverlay[:,:,channel]=np.copy(imTemp)

        colorInd=FDRmap<=-threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
            #imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            funcOverlay[:,:,channel]=imTemp[:,:]

        ## WARM COLORS
        for t in range(len(threshList)-1):
            colorInd=np.logical_and(FDRmap<threshList[t+1],FDRmap>=threshList[t])
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                #  imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=FDRmap>=threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)

            #imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
            funcOverlay[:,:,channel]=imTemp[:,:]    

        #MAKE AND SAVE IMAGE WITH COLOR BAR
        mycmap = array2cmap(cMap)
        plt.imshow(np.uint8(funcOverlay),cmap=mycmap)
        plt.axis('off')

        nPos=np.shape(colorOverlayPos)[0]
        nNeg=np.shape(colorOverlayNeg)[0]
        nSteps=nPos+nNeg+1

        t1Loc=np.true_divide(1,nSteps)
        t2Loc=(np.true_divide(1,nSteps)*nPos)-np.true_divide(t1Loc,2)
        t3Loc=t2Loc+(2*np.true_divide(1,nSteps))
        t4Loc=1-np.true_divide(1,nSteps)
        tickLoc=np.array([t1Loc,t2Loc,t3Loc,t4Loc])
        tickLoc=tickLoc*np.max(np.uint8(funcOverlay))

        t1Label=-np.max(threshList)
        t2Label=-np.min(threshList)
        t3Label=np.min(threshList)
        t4Label=np.max(threshList)
        tickLabels=np.around(np.array([t1Label,t2Label,t3Label,t4Label]),4)

        cb = plt.colorbar()
        cb.set_ticks(tickLoc)
        cb.set_ticklabels(tickLabels)

        if mask is None:
            plt.savefig(figOutDir+sessID+'_'+contrastDic[c]['name']+'_withColorbar.png')
            outFile=figOutDir+sessID+'_'+contrastDic[c]['name']+'_image.png'
        else:
            plt.savefig(figOutDir+sessID+'_'+contrastDic[c]['name']+'_mask_'+mask+'_withColorbar.png')
            outFile=figOutDir+sessID+'_'+contrastDic[c]['name']+'_mask_'+mask+'_image.png'
        misc.imsave(outFile,np.uint8(funcOverlay))

        plt.clf()

def generate_FDR_thresh_PSC_map(sourceRoot,sessID,analysisDir,avgFolder, motionCorrection=False,\
    FDRthresh=.05,manualThresh = False, threshMin = 1, threshMax = 3, contrastName=None,mask = None):

    # DEFINE DIRECTORIES

    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'
    resultsDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/'

    fileOutDir=resultsDir
    figOutDir=analysisDir+'/Figures/'+avgFolder+'/PSC_FDR_maps/'

    if not os.path.exists(figOutDir):
            os.makedirs(figOutDir) 

    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;

    funcOverlayTmp=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlayTmp)


    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=sourceRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    #GET MASK IF NECESSARY
    if mask is not None:
        maskFile=sourceRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
        f=np.load(maskFile)
        maskM=f['maskM']
    else:
        maskM=None

    if contrastName is not None:
        c=0
        while c < len(contrastDic):
            if contrastDic[c]['name']==contrastName:
                contrastDic=[contrastDic[c]]
                break
            c=c+1


    for c in range(len(contrastDic)):

        # LOAD IN MAP WITH PSC PER PIXEL
        mapFile=resultsDir+sessID+'_'+contrastDic[c]['name']+'_PSCmap.npz'
        f=np.load(mapFile)
        PSCmap=f['PSCmap']
        positiveBaselineMask=f['positiveBaselineMask']
        PSCmap=np.nan_to_num(PSCmap)
        if positiveBaselineMask is not None:
            PSCmap[positiveBaselineMask==0]=0

        mapFile=resultsDir+sessID+'_'+contrastDic[c]['name']+'_FDRmap.npz'
        f=np.load(mapFile)
        FDRmap=f['FDRmap']


        if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])
                PSCmap=np.pad(PSCmap,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

        #MASK BASED ON FDR THRESHOLD
        FDRmask=FDRmap<FDRthresh
        PSCmap[FDRmask==0]=0

        #APPLY MASK, IF INDICATED
        if mask is not None:
            PSCmap=PSCmap*maskM



        funcOverlay=np.copy(funcOverlayTmp)

        if not manualThresh:
            pMin=.2
            pMax=.8

            threshMin=np.around(pMin*np.max(np.absolute(PSCmap)),2)
            if threshMin>np.max(np.absolute(PSCmap)):
                threshMax=threshMin
            else:
                threshMax=np.around(pMax*np.max(np.absolute(PSCmap)),2)
        threshList=np.linspace(threshMin,threshMax,8);

        #DEFINE COLOR MAP
        colorOverlayNeg=np.array([[0,0,.25],[0,0,.5],[0,0,.75],[0,0,1],[0,.25,1],[0,.5,1],[0,.75,1],[0,1,1]]);
        colorOverlayPos=np.array([[.25,0,0],[.5,0,0],[.75,0,0],[1,0,0],[1,.25,0],[1,.5,0],[1,.75,0],[1,1,0]]);
        cMap=np.vstack((np.flipud(colorOverlayNeg),np.array([.5,.5,.5]),colorOverlayPos))

        imSat=0.5#SET SATURATION OF CLORS
         #COOL COLORS
        for t in range(len(threshList)-1):
            colorInd=np.where(np.logical_and(PSCmap>-threshList[t+1],PSCmap<=-threshList[t]))

            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=(np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1))-imTemp
                  #  imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=PSCmap<-threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
            #imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            funcOverlay[:,:,channel]=imTemp[:,:]

        #WARM COLORS
        for t in range(len(threshList)-1):
            colorInd=np.logical_and(PSCmap<threshList[t+1],PSCmap>=threshList[t])
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                #  imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=PSCmap>threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            #imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
            funcOverlay[:,:,channel]=imTemp[:,:]

        #MAKE AND SAVE IMAGE WITH COLOR BAR
        mycmap = array2cmap(cMap)
        plt.imshow(np.uint8(funcOverlay),cmap=mycmap)
        plt.axis('off')

        nPos=np.shape(colorOverlayPos)[0]
        nNeg=np.shape(colorOverlayNeg)[0]
        nSteps=nPos+nNeg+1

        t1Loc=np.true_divide(1,nSteps)
        t2Loc=(np.true_divide(1,nSteps)*nPos)-np.true_divide(t1Loc,2)
        t3Loc=t2Loc+(2*np.true_divide(1,nSteps))
        t4Loc=1-np.true_divide(1,nSteps)
        tickLoc=np.array([t1Loc,t2Loc,t3Loc,t4Loc])
        tickLoc=tickLoc*np.max(np.uint8(funcOverlay))

        t1Label=-np.max(threshList)
        t2Label=-np.min(threshList)
        t3Label=np.min(threshList)
        t4Label=np.max(threshList)
        tickLabels=np.around(np.array([t1Label,t2Label,t3Label,t4Label]),4)

        cb = plt.colorbar()
        cb.set_ticks(tickLoc)
        cb.set_ticklabels(tickLabels)

        if mask is None:
            outFile='%s_%s_FDR_%s_withColorbar.png'%\
            (figOutDir+sessID,contrastDic[c]['name'],str(FDRthresh))
            plt.savefig(outFile)
            outFile='%s_%s_FDR_%s_image.png'%\
            (figOutDir+sessID,contrastDic[c]['name'],str(FDRthresh))
        else:
            outFile='%s_%s_FDR_%s_mask_%s_withColorbar.png'%\
            (figOutDir+sessID,contrastDic[c]['name'],str(FDRthresh),mask)
            plt.savefig(outFile)
            outFile='%s_%s_FDR_%s_mask_%s_image.png'%\
            (figOutDir+sessID,contrastDic[c]['name'],str(FDRthresh),mask)

        misc.imsave(outFile,np.uint8(funcOverlay))

        plt.clf()

def generate_bootstrapStat_map(sourceRoot,sessID,analysisDir,avgFolder,motionCorrection=False,\
    manualThresh = False, threshMin = 1.35, threshMax = 3,contrastName=None,mask=None):
    #Cesar Echavarria - 11/2016
    
    #VERIFY WE GOT NECESSARY VALUES
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
        
    # DEFINE DIRECTORIES
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'
    resultsDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/'


    outDir=analysisDir+'/Figures/'+avgFolder+'/bootstrapStatMaps/'

    if not os.path.exists(outDir):
        os.makedirs(outDir) 



    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;

    funcOverlayTmp=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlayTmp)



    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=sourceRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    #GET MASK IF NECESSARY
    if mask is not None:
        maskFile=sourceRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
        f=np.load(maskFile)
        maskM=f['maskM']
    
    if contrastName is not None:
        c=0
        while c < len(contrastDic):
            if contrastDic[c]['name']==contrastName:
                contrastDic=[contrastDic[c]]
                break
            c=c+1

    for c in range(len(contrastDic)):
        # LOAD IN MAP WITH PSC PER PIXEL
        mapFile=resultsDir+sessID+'_'+contrastDic[c]['name']+'_bootstrapMap.npz'
        f=np.load(mapFile)
        pValMap=f['pValMap']
        signMap=f['signMap']

        SPMmap=(-np.log10(pValMap))*signMap
        SPMmap=np.nan_to_num(SPMmap)

        if motionCorrection:
            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']
            padDown=int(boundaries[0])
            padUp=int(szY-boundaries[1])
            padLeft=int(boundaries[2])
            padRight=int(szX-boundaries[3])
            SPMmap=np.pad(SPMmap,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

        #APPLY MASK, IF INDICATED
        if mask is not None:
            SPMmap=SPMmap*maskM

        funcOverlay=np.copy(funcOverlayTmp)
        if not manualThresh:
            #AUTOMATIC THRESHOLD 
            pMin=.2
            pMax=.8
            
            threshMin=np.round(pMin*np.max(np.absolute(SPMmap)))
            
            if threshMin<1.35:
                threshMin=1.35
            if threshMin>np.round(pMax*np.max(np.absolute(SPMmap))):
                threshMax=np.ceil(threshMin)
            else:
                threshMax=np.round(pMax*np.max(np.absolute(SPMmap)))
            
        threshList=np.linspace(threshMin,threshMax,8);
        print(threshList)
        
        #DEFINE COLOR MAP
        colorOverlayNeg=np.array([[0,0,.25],[0,0,.5],[0,0,.75],[0,0,1],[0,.25,1],[0,.5,1],[0,.75,1],[0,1,1]]);
        colorOverlayPos=np.array([[.25,0,0],[.5,0,0],[.75,0,0],[1,0,0],[1,.25,0],[1,.5,0],[1,.75,0],[1,1,0]]);
        cMap=np.vstack((np.flipud(colorOverlayNeg),np.array([.5,.5,.5]),colorOverlayPos))

        imSat=.5#SET SATURATION OF CLORS
        #COOL COLORS
        for t in range(len(threshList)-1):
            colorInd=np.where(np.logical_and(SPMmap>-threshList[t+1],SPMmap<=-threshList[t]))

            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=(np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1))-imTemp
                  #  imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=SPMmap<=-threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
            #imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            funcOverlay[:,:,channel]=imTemp[:,:]

        #WARM COLORS
        for t in range(len(threshList)-1):
            colorInd=np.logical_and(SPMmap<threshList[t+1],SPMmap>=threshList[t])
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                #  imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=SPMmap>=threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            #imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
            funcOverlay[:,:,channel]=imTemp[:,:]    
        #MAKE AND SAVE IMAGE WITH COLOR BAR
        mycmap = array2cmap(cMap)
        plt.imshow(np.uint8(funcOverlay),cmap=mycmap)
        plt.axis('off')

        nPos=np.shape(colorOverlayPos)[0]
        nNeg=np.shape(colorOverlayNeg)[0]
        nSteps=nPos+nNeg+1

        t1Loc=np.true_divide(1,nSteps)
        t2Loc=(np.true_divide(1,nSteps)*nPos)-np.true_divide(t1Loc,2)
        t3Loc=t2Loc+(2*np.true_divide(1,nSteps))
        t4Loc=1-np.true_divide(1,nSteps)
        tickLoc=np.array([t1Loc,t2Loc,t3Loc,t4Loc])
        tickLoc=tickLoc*np.max(np.uint8(funcOverlay))

        #*Priting out P-value directly is troublesome...small p-values not printed out correctly. Opting for printing out -log10(pValue)
        t1Label=-np.max(threshList)
        t2Label=-np.min(threshList)
        t3Label=np.min(threshList)
        t4Label=np.max(threshList)
        tickLabels=np.around(np.array([t1Label,t2Label,t3Label,t4Label]),4)



        cb = plt.colorbar()
        cb.set_ticks(tickLoc)
        cb.set_ticklabels(tickLabels)

        if mask is None:
            plt.savefig('%s_%s_min_%.2f_max_%.2f_withColorbar.png'%\
                        (outDir+sessID,contrastDic[c]['name'],threshMin,threshMax))

            outFile='%s_%s_min_%.2f_max_%.2f_image.png'%\
                        (outDir+sessID,contrastDic[c]['name'],threshMin,threshMax)
        else:
            plt.savefig('%s_%s_min_%.2f_max_%.2f_mask_%s_withColorbar.png'%\
                        (outDir+sessID,contrastDic[c]['name'],threshMin,threshMax,mask))

            outFile='%s_%s_min_%.2f_max_%.2f_mask_%s_image.png'%\
                        (outDir+sessID,contrastDic[c]['name'],threshMin,threshMax,mask)
        misc.imsave(outFile,np.uint8(funcOverlay))

        plt.clf()


def generate_zScore_map(sourceRoot,sessID,analysisDir,avgFolder,nCond,motionCorrection=False,\
    thresh=0,mask=None):
    #Cesar Echavarria - 11/2016
    
    #VERIFY WE GOT NECESSARY VALUES
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
        
    # DEFINE DIRECTORIES
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'
    resultsDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/'
    outDir=analysisDir+'/Figures/'+avgFolder+'/zScoreMap/'

    if not os.path.exists(outDir):
        os.makedirs(outDir) 



    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;

    funcOverlayTmp=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlayTmp)

    #GET MASK IF NECESSARY
    if mask is not None:
        maskFile=sourceRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
        f=np.load(maskFile)
        maskM=f['maskM']

    #LOAD IN MAP: RESPONSES Z-SCORED RELATIVE TO BASELINE
    for cond in range(0,nCond):

        mapFile=resultsDir+sessID+'_condition'+str(cond)+'_zScoreMap.npz'  
        f=np.load(mapFile)
        respMap=f['respMap']
        respMap=np.nan_to_num(respMap)
        #threshold based on z-score relative to baseline

        respMap[respMap<=thresh]=0#threshold
        
        szYmap,szXmap=np.shape(respMap)


        if motionCorrection:
            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']
            padDown=int(boundaries[0])
            padUp=int(szY-boundaries[1])
            padLeft=int(boundaries[2])
            padRight=int(szX-boundaries[3])
            respMap=np.pad(respMap,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

        #APPLY MASK, IF INDICATED
        if mask is not None:
            respMap=respMap*maskM
        
        funcOverlay=np.copy(funcOverlayTmp)
        #AUTOMATIC THRESHOLD 
        pMin=.01
        pMax=.8
        
        threshMin=np.ceil(pMin*np.max(np.absolute(respMap)))
        

        threshMax=np.round(pMax*np.max(np.absolute(respMap)))
        
        threshList=np.linspace(threshMin,threshMax,8);
        print(threshList)
        
        #DEFINE COLOR MAP
        colorOverlayNeg=np.array([[0,0,.25],[0,0,.5],[0,0,.75],[0,0,1],[0,.25,1],[0,.5,1],[0,.75,1],[0,1,1]]);
        colorOverlayPos=np.array([[.25,0,0],[.5,0,0],[.75,0,0],[1,0,0],[1,.25,0],[1,.5,0],[1,.75,0],[1,1,0]]);
        cMap=np.vstack((np.flipud(colorOverlayNeg),np.array([.5,.5,.5]),colorOverlayPos))

        imSat=.5#SET SATURATION OF CLORS
        #COOL COLORS
        for t in range(len(threshList)-1):
            colorInd=np.where(np.logical_and(respMap>-threshList[t+1],respMap<=-threshList[t]))

            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=(np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1))-imTemp
                  #  imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=respMap<=-threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
            #imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            funcOverlay[:,:,channel]=imTemp[:,:]

        #WARM COLORS
        for t in range(len(threshList)-1):
            colorInd=np.logical_and(respMap<threshList[t+1],respMap>=threshList[t])
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                #  imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
                funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=respMap>=threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            #imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
            funcOverlay[:,:,channel]=imTemp[:,:]    
        #MAKE AND SAVE IMAGE WITH COLOR BAR
        mycmap = array2cmap(cMap)
        plt.imshow(np.uint8(funcOverlay),cmap=mycmap)
        plt.axis('off')

        nPos=np.shape(colorOverlayPos)[0]
        nNeg=np.shape(colorOverlayNeg)[0]
        nSteps=nPos+nNeg+1

        t1Loc=np.true_divide(1,nSteps)
        t2Loc=(np.true_divide(1,nSteps)*nPos)-np.true_divide(t1Loc,2)
        t3Loc=t2Loc+(2*np.true_divide(1,nSteps))
        t4Loc=1-np.true_divide(1,nSteps)
        tickLoc=np.array([t1Loc,t2Loc,t3Loc,t4Loc])
        tickLoc=tickLoc*np.max(np.uint8(funcOverlay))

        #*Priting out P-value directly is troublesome...small p-values not printed out correctly. Opting for printing out -log10(pValue)
        t1Label=-np.max(threshList)
        t2Label=-np.min(threshList)
        t3Label=np.min(threshList)
        t4Label=np.max(threshList)
        tickLabels=np.around(np.array([t1Label,t2Label,t3Label,t4Label]),4)



        cb = plt.colorbar()
        cb.set_ticks(tickLoc)
        cb.set_ticklabels(tickLabels)

        if mask is None:
            plt.savefig('%s_cond_%s_min_%.2f_max_%.2f_withColorbar.png'%\
                        (outDir+sessID,str(cond+1),threshMin,threshMax))

            outFile='%s_cond_%s_min_%.2f_max_%.2f_image.png'%\
                        (outDir+sessID,str(cond+1),threshMin,threshMax)
        else:
            plt.savefig('%s_cond_%s_min_%.2f_max_%.2f_mask_%s_withColorbar.png'%\
                        (outDir+sessID,str(cond+1),threshMin,threshMax,mask))

            outFile='%s_cond_%s_min_%.2f_max_%.2f_mask_%s_image.png'%\
                        (outDir+sessID,str(cond+1),threshMin,threshMax,mask)

        misc.imsave(outFile,np.uint8(funcOverlay))

        plt.clf()

def generate_preference_map(sourceRoot,sessID,analysisDir,avgFolder,nCond,motionCorrection=False,positiveThresh=True,\
                            thresh=0,useThresh2=True,thresh2=0,mask=None):
    #Cesar Echavarria - 10/2016
    #VERIFY WE GOT NECESSARY VALUES
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
    if nCond is None:
        raise TypeError("nCond (number of conditions) not specified!")

    # DEFINE DIRECTORIES
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'
    inDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';
    outDir=analysisDir+'/Figures/'+avgFolder+'/preferenceMap/'

    if not os.path.exists(outDir):
        os.makedirs(outDir) 


    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;


    funcOverlay=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlay)

    #GET MASK IF NECESSARY
    if mask is not None:
        maskFile=sourceRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
        f=np.load(maskFile)
        maskM=f['maskM']


    #LOAD IN MAP: RESPONSES Z-SCORED RELATIVE TO BASELINE
    
    for cond in range(0,nCond):

        mapFile=inDir+sessID+'_condition'+str(cond)+'_zScoreMap.npz'  
        f=np.load(mapFile)
        respMap=f['respMap']
        respMap=np.nan_to_num(respMap)
        
        szYmap,szXmap=np.shape(respMap)
        if cond ==0:
            allMaps=np.zeros((szYmap,szXmap,nCond))

        if not positiveThresh:#for negative thresholding
            respMap=-respMap

        #threshold based on z-score relative to baseline
        respMap=np.reshape(respMap,szYmap*szXmap)
        respMapThresh=np.copy(respMap)
        respMapThresh[respMap<=thresh]=0#threshold
        respMask=respMapThresh>0

        #Z-SCORE AGAIN
        respMap2=np.zeros(np.shape(respMap))
        if np.std(respMapThresh) != 0:
            respMap2[respMap>thresh]=(respMapThresh[respMap>thresh]-np.mean(respMap))/np.std(respMap)
        #store into array for all conditions
        tmpMap=np.reshape(respMap2,(szYmap,szXmap))
   #     tmpMap=cv2.resize(tmpMap,(szX,szY))
        allMaps[:,:,cond]=tmpMap

    #SET 2ND THRESHOLD
    if useThresh2:
        allMaps[allMaps<thresh2]=0

    if motionCorrection:
        #LOAD MOTION CORRECTED BOUNDARIES
        inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
        f=np.load(inFile)
        boundaries=f['boundaries']
        padDown=int(boundaries[0])
        padUp=int(szY-boundaries[1])
        padLeft=int(boundaries[2])
        padRight=int(szX-boundaries[3])
        allMaps=np.pad(allMaps,((padDown,padUp),(padLeft,padRight),(0,0)),'constant',constant_values=((0, 0),(0,0),(0,0)))


    szYmap=szX
    szXmap=szY
    #FIND CONDITION THAT EVOKE MAX RESPONSE FOR EACH PIXEL
    sumSigMap=np.sum(allMaps,2)
    maxPos=np.argmax(allMaps,2)+1
    maxPos[sumSigMap==0]=0

    #APPLY MASK, IF INDICATED
    if mask is not None:
        maxPos=maxPos*maskM


    if nCond==4:
        #MAKE MAPS
        colorCode=np.array([[255,0,0],[0,0,255],[255,255,0],[0,255,0]])
        constantSat=.4

        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            for pos in range(1,nCond+1):
                diffVal=(np.ones((szY,szX))*colorCode[pos-1,channel])-imTemp

                imTemp[maxPos==pos]=imTemp[maxPos==pos]+np.multiply(diffVal[maxPos==pos],constantSat)
                funcOverlay[:,:,channel]=imTemp[:,:]
        #SAVE MAPS
        if mask is None:
            if useThresh2:
                outFile=outDir+sessID+'_colorCodedPositionMap_thresh_'+str(thresh)+'_thresh2_'+str(thresh2)+'.png'
            else:
                outFile=outDir+sessID+'_colorCodedPositionMap_thresh_'+str(thresh)+'.png'
        else:
            if useThresh2:
                outFile=outDir+sessID+'_colorCodedPositionMap_thresh_'+str(thresh)+'_thresh2_'+str(thresh2)+'_mask_'+mask+'.png'
            else:
                outFile=outDir+sessID+'_colorCodedPositionMap_thresh_'+str(thresh)+'_mask_'+mask+'.png'
        misc.imsave(outFile,np.uint8(funcOverlay))

        #MAKE LEGEND
        nRows=2
        nCols=2
        legend=np.zeros((nRows*100,nCols*100,3))

        count=0
        for y in range(0,nRows):
            Y1=(100*(y))
            Y2=Y1+100
            for x in range(0,nCols):
                X1=(100*(x))
                X2=X1+100
                for dim in range(0,3):
                    legend[Y1:Y2,X1:X2,dim]=colorCode[count,dim];
                count=count+1;

        outFile=outDir+'colorCodedPositionLegend.png'   
        misc.imsave(outFile,np.uint8(legend))
    elif nCond==15:
        #MAKE MAPS
        colorDic=[]
        colorDic.append({'name':'AllPositions',\
                         'values':np.array([[255,0,0],[255,0,127],[127,0,127],[127,0,255],[0,0,255],\
                                            [255,128,0],[255,128,127],[55,55,127],[127,128,255],[0,128,255],\
                                            [255,255,0],[255,255,127],[127,255,127],[127,255,255],[0,255,255]])})
        colorDic.append({'name':'Elevation',\
                         'values':np.array([[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],\
                                            [255,128,0],[255,128,0],[255,128,0],[255,128,0],[255,128,0],\
                                            [0,255,0],[0,255,0],[0,255,0],[0,255,0],[0,255,0]])})
        colorDic.append({'name':'Azimuth',\
                         'values':np.array([[255,0,0],[255,100,127],[127,0,127],[127,0,255],[0,0,255],\
                                            [255,0,0],[255,100,127],[127,0,127],[127,0,255],[0,0,255],\
                                            [255,0,0],[255,100,127],[127,0,127],[127,0,255],[0,0,255]])})
        constantSat=.5
        for cMap in range(0,len(colorDic)):
            funcOverlay=np.dstack((imSurf,imSurf,imSurf))
            colorCode=colorDic[cMap]['values']
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                for pos in range(1,nCond+1):
                    diffVal=(np.ones((szY,szX))*colorCode[pos-1,channel])-imTemp
                    imTemp[maxPos==pos]=imTemp[maxPos==pos]+np.multiply(diffVal[maxPos==pos],constantSat)
                    funcOverlay[:,:,channel]=imTemp[:,:]
            #SAVE MAPS
            if mask is None:
                if useThresh2:
                    outFile=outDir+sessID+'_colorCodedPositionMap'+colorDic[cMap]['name']+'_thresh_'+str(thresh)+'_thresh2_'+str(thresh2)+'.png'
                else:
                    outFile=outDir+sessID+'_colorCodedPositionMap'+colorDic[cMap]['name']+'_thresh_'+str(thresh)+'.png'
            else:
                if useThresh2:
                    outFile=outDir+sessID+'_colorCodedPositionMap'+colorDic[cMap]['name']+'_thresh_'+str(thresh)+'_thresh2_'+str(thresh2)+'_mask_'+mask+'.png'
                else:
                    outFile=outDir+sessID+'_colorCodedPositionMap'+colorDic[cMap]['name']+'_thresh_'+str(thresh)+'_mask_'+mask+'.png'

            misc.imsave(outFile,np.uint8(funcOverlay))

            #MAKE LEGEND
            nRows=3
            nCols=5
            legend=np.zeros((nRows*100,nCols*100,3))

            count=0
            for y in range(0,nRows):
                Y1=(100*(y))
                Y2=Y1+100
                for x in range(0,nCols):
                    X1=(100*(x))
                    X2=X1+100
                    for dim in range(0,3):
                        legend[Y1:Y2,X1:X2,dim]=colorCode[count,dim];
                    count=count+1;

            outFile=outDir+colorDic[cMap]['name']+'_Legend.png'   
            misc.imsave(outFile,np.uint8(legend))


# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ROI ANALYSIS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ROI_remove_vessels(rawRoot, analyzedRoot, sessID, ROIlist):


    #DEFINE DIRECTORIES
    inDir=analyzedRoot+'Sessions/'+sessID+'/ROIs/ROImaker/'
    outDir=inDir

    borderImgDir=outDir+'Figures/'

    if not os.path.exists(borderImgDir):
                os.makedirs(borderImgDir)

    borderFileDir=outDir+'Files/'
    if not os.path.exists(borderFileDir):
                os.makedirs(borderFileDir)

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=borderFileDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()

    #GET SOME INFO
    imRef=get_reference_frame(rawRoot,sessID)
    szY,szX=imRef.shape

    #LOAD VESSEL ROI
    inFile=inDir+'Files/vessels.npz'
    f=np.load(inFile)
    vesselInd=f['ROIind']

    allBorders=np.zeros((szY,szX))
    for roi in ROIlist:
        #LOAD VISUAL AREA ROI 
        inFile=inDir+'Files/'+roi+'.npz'
        f=np.load(inFile)
        ROIind=f['ROIind']
        ROIimg=np.zeros((szY,szX))
        ROIimg[ROIind[0],ROIind[1]]=1

        #REMOVE VESSELS
        ROI_noVessels=np.copy(ROIimg)
        ROI_noVessels[vesselInd[0],vesselInd[1]]=0

        #GET BORDERS OF NEW ROI
        tmp=np.uint8(ROI_noVessels*255)
        tmp=cv2.GaussianBlur(tmp,(11,11),.9,.9)
        border_noVessels = cv2.Canny(tmp,50,200)
        border_noVessels = np.true_divide(border_noVessels,255)


        #SAVE NEW ROI
        outFile=borderFileDir+roi+'_noVessels'
        borderInd=np.where(border_noVessels==1)

        ROIind=np.where(ROI_noVessels==1)
        np.savez(outFile,ROIind=ROIind,borderInd=borderInd)   

        #SAVE IMAGES WITH BORDERS OF NEW ROI
        anatSource=analyzedRoot+'Sessions/'+sessID+'/Surface/'

        imFile=anatSource+'16bitSurf.tiff'
        imSurf=cv2.imread(imFile,-1)
        imSurf[border_noVessels==1]=65535

        outFile=borderImgDir+'16bitSurf_withBorders_'+roi+'_noVessels.tiff'
        cv2.imwrite(outFile,np.uint16(imSurf))#THIS FILE MUST BE OPENED WITH CV2 MODULE

        imRef = np.copy(imRef)
        imRef=np.uint8(np.true_divide(imRef,np.max(imRef))*255)
        imRef[border_noVessels==1]=255
        outFile=borderImgDir+'firstFrame_withBorders_'+roi+'_noVessels.png'
        misc.imsave(outFile,imRef)

        #AGGREGATE BORDERS
        allBorders[borderInd[0],borderInd[1]]=1

    #OUTPUT IMAGES WITH ALL BORDERS
    anatSource=analyzedRoot+'Sessions/'+sessID+'/Surface/'

    imFile=anatSource+'16bitSurf.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf[allBorders==1]=65535

    outFile=borderImgDir+'16bitSurf_withBorders_All_noVessels.tiff'
    cv2.imwrite(outFile,np.uint16(imSurf))#THIS FILE MUST BE OPENED WITH CV2 MODULE

    imRef=np.uint8(np.true_divide(imRef,np.max(imRef))*255)
    imRef[allBorders==1]=255
    outFile=borderImgDir+'firstFrame_withBordersAll_noVessels.png'
    misc.imsave(outFile,imRef)

def register_ROI(rawRoot_source,analyzedRoot_source,rawRoot_target,analyzedRoot_target,\
                  sessID_source, sessID_target,\
                  ROIfolder,ROIlist,sourceRun=1,sourceTarget=1):
    #Cesar Echavarria 1/2017
    
    #DEFINE DIRECTORIES
    inDir=analyzedRoot_source+'Sessions/'+sessID_source+'/ROIs/'+ROIfolder+'/Files/'
    
    outDir=analyzedRoot_target+'Sessions/'+sessID_target+'/ROIs/'+ROIfolder+'/'
    borderImgDir_target=outDir+'Figures/'

    if not os.path.exists(borderImgDir_target):
                os.makedirs(borderImgDir_target)

    borderFileDir_target=outDir+'Files/'
    if not os.path.exists(borderFileDir_target):
                os.makedirs(borderFileDir_target)

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=borderFileDir_target+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()
            
    #LOAD IMAGES FOR REGISTRATION
    sourceRun=1
    targetRun=1
    #USE FIRST FRAME FROM FIRST RUN TO REGISTER DATA ACROSS SESSIONS
    sourceImg=get_reference_frame(rawRoot_source,sessID_source,sourceRun)
    targetImg=get_reference_frame(rawRoot_target,sessID_target,targetRun)

    #REGISTRATION
    sourceImg=np.expand_dims(sourceImg,2)
    warpMatrices,motionMag=motion_registration(targetImg,sourceImg)
    
    count=0        
    for ROIname in ROIlist:
        
        #LOAD ROI DATA
        inFile=inDir+ROIname+'.npz'
        f=np.load(inFile)
        ROIind=f['ROIind']
        borderInd=f['borderInd']


        #TRANSFORM ROI DATA
        sourceImg=np.squeeze(sourceImg)
        ROIimg=np.zeros(np.shape(sourceImg))
        ROIimg[ROIind[0],ROIind[1]]=1
        ROIimg1=np.expand_dims(ROIimg,2)
        ROIimg2=apply_motion_correction(ROIimg1,warpMatrices)
        ROIimg3=ROIimg2>0
        ROIimg3=np.squeeze(ROIimg3)

        sourceImg=np.squeeze(sourceImg)
        borderImg=np.zeros(np.shape(sourceImg))
        borderImg[borderInd[0],borderInd[1]]=1
        borderImg1=np.expand_dims(borderImg,2)
        borderImg2=apply_motion_correction(borderImg1,warpMatrices)
        borderImg3=borderImg2>0
        borderImg3=np.squeeze(borderImg3)
        
        
        

        #SAVE ROI INDICES TO FILE
        outFile=borderFileDir_target+ROIname
        borderInd=np.where(borderImg3==1)
        ROIind=np.where(ROIimg3==1)
        np.savez(outFile,ROIind=ROIind,borderInd=borderInd)        

        #OUTPUT IMAGES
        anatSource=analyzedRoot_target+'Sessions/'+sessID_target+'/Surface/'
        imFile=anatSource+'16bitSurf.tiff'
        imSurf=cv2.imread(imFile,-1)
        imSurf[borderImg3==1]=65535
        outFile=borderImgDir_target+'16bitSurf_'+ROIname+'_Borders.tiff'
        cv2.imwrite(outFile,np.uint16(imSurf))#THIS FILE MUST BE OPENED WITH CV2 MODULE
        
        templateImg=np.copy(targetImg)
        templateImg=np.uint8(np.true_divide(templateImg,np.max(templateImg))*255)
        templateImg[borderImg3==1]=255
        outFile=borderImgDir_target+'firstFrame_'+ROIname+'_Borders.png'
        misc.imsave(outFile,templateImg)
        
        if count == 0:
            allBorders=np.copy(targetImg)
            allBorders=np.uint8(np.true_divide(allBorders,np.max(allBorders))*255)
        allBorders[borderImg3==1]=255
        count=count+1
        
    outFile=borderImgDir_target+'firstFrame_Borders.png'
    misc.imsave(outFile,allBorders)

def draw_ROI_all_borders(rawRoot,analyzedRoot,sessID,ROIlist,ROIfolder,analysisDir,avgFolder,targetRun=1):
    #Cesar Echavarria 1/2017

    #DEFINE DIRECTORIES
    inDir=analyzedRoot+'Sessions/'+sessID+'/ROIs/'+ROIfolder+'/Files/'


    targetImg=get_reference_frame(rawRoot,sessID,targetRun)
    allBorders=np.zeros(np.shape(targetImg))  

    for count,ROIname in enumerate(ROIlist):

        #LOAD ROI DATA
        inFile=inDir+ROIname+'.npz'
        f=np.load(inFile)
        ROIind=f['ROIind']
        borderInd=f['borderInd']


        #PUT BORDER INDICES TO ARRAY
        allBorders[borderInd[0],borderInd[1]]=1
    
    folderList=['SPMmaps','PSCcontrast','bootstrapStatMaps']
    for folder in folderList:
        sourceImgDir=analysisDir+'/Figures/'+avgFolder+'/'+folder+'/'
        
        if os.path.exists(sourceImgDir):
            outImgDir=sourceImgDir+'withBorders/'+ROIfolder+'/'
            if not os.path.exists(outImgDir):
                    os.makedirs(outImgDir)

            #GO TRHOUGH IMAGES IN FOLDER
            imFileList=glob.glob(sourceImgDir+sessID+'*image.png')

            for imFile in imFileList:
                im0=misc.imread(imFile)
                szY,szX,szZ=np.shape(im0)
                #DRAW BORDERS
                for d in range(0,szZ):
                    imDim=im0[:,:,d]
                    imDim[allBorders==1]=255
                    im0[:,:,d]=imDim

                refInd=imFile.rfind('/')+1
                imFileName=imFile[refInd:]
                outImgFile=outImgDir+imFileName
                misc.imsave(outImgFile,np.uint8(im0))



def show_ROI_session_timecourse(figOutRoot, ROI_label, sessIDlist,stimRespROI,dataExists, frameTimes,\
                              condLabels, stimStartT, stimEndT, normData=False):
    
    #GET SESSIONS FOR WHICH DATA EXISTS
    validInd=np.where(dataExists)[0]
    stimRespROI=stimRespROI[validInd,:,:]
    sessIDlist=[sessIDlist[i] for i in validInd]
    
    #GET INFO ABOUT DATA
    nSess,nCond,nPts=np.shape(stimRespROI)
    nReps=np.round(np.true_divide(nPts,nSess))
    
    #NORMALIZE DATA
    if normData:
        tmpResp=np.zeros(np.shape(stimRespROI))
        for s in range(nSess):
            tmp=stimRespROI[s,:,:]
            tmpResp[s,:,:]=np.true_divide(tmp,np.max(tmp))
        stimRespROI=tmpResp
        del tmpResp
    
    #TILE TRIAL TIMECOURSE TO GET A 1:1 RATIO FIGURE
    stimRespROI_figure=np.zeros((nSess*nReps,nCond,nPts))
    for c in range(nCond):
        startInd=0
        for s in range(nSess):
            tmp=stimRespROI[s,c,:]
            tmpTile=np.tile(tmp,(nReps,1))
            stimRespROI_figure[startInd:startInd+nReps,c,:]=tmpTile
            startInd=startInd+nReps

    #MAKE PLOTS
    for c in range(nCond):
        fig, ax = plt.subplots()
        plt.imshow(np.squeeze(stimRespROI_figure[:,c,:]),cmap='jet',vmin=np.amin(stimRespROI),vmax=np.amax(stimRespROI))
        plt.colorbar()
        plt.xlabel('Time(s)',fontsize=14)
        plt.ylabel('Sessions',fontsize=14)
        
        #set x-ticks and labels
        xTickLabels=np.int64(np.arange(np.min(frameTimes),np.max(frameTimes),2))
        xTickLoc=np.arange(len(xTickLabels))
        for tickCount,tick in enumerate(xTickLabels):
            xTickLoc[tickCount]=np.where(frameTimes>tick)[0][0]
        ax.xaxis.set_ticks(xTickLoc)
        ax.set_xticklabels(xTickLabels,fontsize=10)
        
        #set y-ticks and labels
        yTickLabels=sessIDlist
        yTickLoc=np.arange(np.true_divide(nReps,2),nReps*nSess,nReps)-1
        ax.yaxis.set_ticks(yTickLoc)
        ax.set_yticklabels(yTickLabels,fontsize=10)
        
        #set stim onset/offset lines
        plt.axvline(x=np.where(frameTimes>stimStartT)[0][0], ymin=0, ymax = nReps*nSess, linewidth=1, color='k')
        plt.axvline(x=np.where(frameTimes>stimEndT)[0][0], ymin=0, ymax = nReps*nSess, linewidth=1, color='k')
        
        fig.suptitle(ROI_label+' '+condLabels[c]+' Response', fontsize=14)
        if normData:
            figFile=figOutRoot+ROI_label+'_timecourse_'+condLabels[c]+'_allSessions_Norm.png'
        else:
            figFile=figOutRoot+ROI_label+'_timecourse_'+condLabels[c]+'_allSessions.png'
        plt.savefig(figFile)
        plt.close()  
        
def plot_ROI_aggregate_tCourse(stimRespROI, frameTimes, stimStartT, stimEndT, listROI, \
    condLabels, figOutDir, dataExists=None, groupSessions=True, normData=False):
    if dataExists is None:
        dataExists=np.ones(np.shape(stimRespROI))
    
    nSess,nROI,nCond,nPts=np.shape(stimRespROI)
    #NORMALIZE DATA
    if normData:
        tmpResp=np.zeros(np.shape(stimRespROI))
        for s in range(nSess):
            validInd=np.where(dataExists[s,:])[0]
            tmp=stimRespROI[s,validInd,:,:]
            tmpResp[s,validInd,:,:]=np.true_divide(tmp,np.max(tmp))
        stimRespROI=tmpResp
        del tmpResp
        
    #GET MEAN ACROSS SESSIONS
    if np.all(dataExists):
        stimRespROI_mean=np.mean(stimRespROI,0)
    else:
        stimRespROI_mean=np.zeros((nROI,nCond,nPts))
        for r in range(nROI):
            validInd=np.where(dataExists[:,r])[0]
            tmp=stimRespROI[validInd,r,:,:]
            stimRespROI_mean[r,:,:]=np.mean(tmp,0)

    for roiCount,roi in enumerate(listROI):
        #MAKE PLOT, SESSION-BY-SESSION
        colorList='bgrmkc'
        legHand=[]
        fig=plt.figure()
        for s in range(nSess):
            for c in range(nCond):
                if dataExists[s,roiCount]:
                    plt.plot(frameTimes,stimRespROI[s,roiCount,c,:],colorList[c],linewidth=1)
                    if s==0:
                        legHand.append(mlines.Line2D([], [], color=colorList[c], marker='_',
                                              markersize=10, label=condLabels[c]))
        plt.legend(handles=legHand)

        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        if normData:
            ymax=1
        plt.axvline(x=stimStartT, ymin=ymin, ymax = ymax, linewidth=1, color='k')
        plt.axvline(x=stimEndT, ymin=ymin, ymax = ymax, linewidth=1, color='r')
        if ymin<=0:
            xmin, xmax = axes.get_xlim()
            plt.axhline(xmin=xmin, xmax=xmax , y= 0, linewidth=1, color='k')

        fig.suptitle(roi+' Response', fontsize=14)
        plt.xlabel('Time (secs)',fontsize=14)
        if normData:
            plt.ylabel('Response Magnitude',fontsize=14)
        else:    
            plt.ylabel('PSC',fontsize=14)
        if normData:
            figOutFile=figOutDir+roi+'_timecourse_aggregate_normalized.png'
        else:
            figOutFile=figOutDir+roi+'_timecourse_aggregate.png'

        plt.savefig(figOutFile)
        plt.close()

        #MAKE PLOT; MEAN
        colorList='bgrmkc'
        legHand=[]
        fig=plt.figure()

        for c in range(nCond):
            plt.plot(frameTimes,stimRespROI_mean[roiCount,c,:],colorList[c],linewidth=1)

            legHand.append(mlines.Line2D([], [], color=colorList[c], marker='_',
                                  markersize=10, label=condLabels[c]))
        plt.legend(handles=legHand)

        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        if normData:
            ymax=1
        plt.axvline(x=stimStartT, ymin=ymin, ymax = ymax, linewidth=1, color='k')
        plt.axvline(x=stimEndT, ymin=ymin, ymax = ymax, linewidth=1, color='r')
        if ymin<=0:
            xmin, xmax = axes.get_xlim()
            plt.axhline(xmin=xmin, xmax=xmax , y= 0, linewidth=1, color='k')

        fig.suptitle(roi+' Mean Response', fontsize=14)
        plt.xlabel('Time (secs)',fontsize=14)
        if normData:
            plt.ylabel('Response Magnitude',fontsize=14)
        else:    
            plt.ylabel('PSC',fontsize=14)

        if normData:
            figOutFile=figOutDir+roi+'_timecourse_mean_normalized.png'
        else:
            figOutFile=figOutDir+roi+'_timecourse_mean.png'

        plt.savefig(figOutFile)
        plt.close()



def ROI_analysis_aggregate_tCourse(targetRoot, analysisPath, nCond, stimDur, avgFolder, \
                                   aggregateName, sessIDlist, ROIfolder, listROI):

    #Cesar Echavarria 1/2017

    #DEFINE SOME DIRECTORIES
    ROIoutDir=targetRoot+'/ROIanalyses/'+aggregateName+'/'

    figOutDir_plots=ROIoutDir+'Figures/tCourse/plots/'
    if not os.path.exists(figOutDir_plots):
            os.makedirs(figOutDir_plots)
            
    figOutDir_imgs=ROIoutDir+'Figures/tCourse/images/'
    if not os.path.exists(figOutDir_imgs):
            os.makedirs(figOutDir_imgs) 

    fileOutDir=ROIoutDir+'Files/'
    if not os.path.exists(fileOutDir):
            os.makedirs(fileOutDir) 

#     #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=fileOutDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()

    nSess=len(sessIDlist)
    nROI=len(listROI)
    dataExists=np.empty((nSess,nROI))

    #GO THROUGH EACH SESSION
    for sessCount,sessID in enumerate(sessIDlist):


        #DEFINE SOME DIRECTORIES
        analysisDir=targetRoot+'/Sessions/'+sessID+analysisPath
        ROI_analysisFolder=analysisDir+'/ROIanalysis/'+avgFolder+'/'+ROIfolder+'/'
        fileInDir=ROI_analysisFolder+'/timecourse_average/Files/'


        #GET ROI DATA
        for roiCount,roi in enumerate(listROI):
            inFile=fileInDir+roi+'.npz'
            if os.path.isfile(inFile):
                dataExists[sessCount,roiCount]=True
                f=np.load(inFile)
                stimResp=f['stimRespROI']
                frameTimes=f['frameTimes']
                condLabels=f['condLabels']
                stimStartT=f['stimStartT']
                stimEndT=f['stimEndT']
            else:
                dataExists[sessCount,roiCount]=False

            if roiCount==0 and sessCount==0:
                #INTIALIZE EMPTY ARRAY
                nCond,nPts=np.shape(stimResp)
                stimRespAll=np.zeros((nSess,nROI,nCond,nPts))
            if os.path.isfile(inFile):    
                stimRespAll[sessCount,roiCount,:,:]=stimResp

    # # #SAVE DATA
    outFile=fileOutDir+aggregateName+'_timecourse_ROIdata'
    np.savez(outFile,stimResp=stimRespAll,dataExists=dataExists ,frameTimesAll=frameTimes,condLabels=condLabels,\
             stimStartT=stimStartT,stimEndT=stimStartT+stimDur)

    # #WRITE sessID list to file
    outFile=fileOutDir+'sessID_list_timecourse.txt'
    sessIDTextFile = open(outFile, 'w+')
    sessIDTextFile.write('LIST OF SESSIONS\n')
    sessIDTextFile.write('\n')
    for sess in sessIDlist:
        sessIDTextFile.write(sess+'\n')
    sessIDTextFile.close()

    plot_ROI_aggregate_tCourse(stimRespAll, frameTimes, stimStartT, stimEndT, listROI, condLabels,\
        figOutDir_plots, dataExists, normData=False)
    plot_ROI_aggregate_tCourse(stimRespAll, frameTimes, stimStartT, stimEndT, listROI, condLabels, \
        figOutDir_plots, dataExists, normData=True)
    
    for r in range(len(listROI)):
        ROI_label=listROI[r]
        show_ROI_session_timecourse(figOutDir_imgs, ROI_label, sessIDlist, np.squeeze(stimRespAll[:,r,:,:]),dataExists[:,r], frameTimes,\
                              condLabels, stimStartT, stimStartT+stimDur)
        show_ROI_session_timecourse(figOutDir_imgs, ROI_label, sessIDlist, np.squeeze(stimRespAll[:,r,:,:]),dataExists[:,r], frameTimes,\
                              condLabels, stimStartT, stimStartT+stimDur,normData=True)

def show_ROI_trial_timecourse(figOutRoot, ROI_label, stimRespROI, frameTimes,\
                              condLabels, stimStartT, stimEndT):
    
    #GET INFO ABOUT DATA
    nCond,nTrials,nPts=np.shape(stimRespROI)
    nReps=np.round(np.true_divide(nPts,nTrials));
    
    #TILE TRIAL TIMECOURSE TO GET A 1:1 RATIO FIGURE
    stimRespROI_figure=np.zeros((nCond,nTrials*nReps,nPts))
    for c in range(nCond):
        startInd=0
        for t in range(nTrials):
            tmp=stimRespROI[c,t,:]
            tmpTile=np.tile(tmp,(nReps,1))
            stimRespROI_figure[c,startInd:startInd+nReps,:]=tmpTile
            startInd=startInd+nReps

    #MAKE PLOTS
    for c in range(nCond):
        fig, ax = plt.subplots()
        plt.imshow(np.squeeze(stimRespROI_figure[c,:,:]),cmap='jet',vmin=np.amin(stimRespROI),vmax=np.amax(stimRespROI))
        plt.colorbar()
        plt.xlabel('Time(s)',fontsize=14)
        plt.ylabel('Trials',fontsize=14)
        
        #set x-ticks and labels
        xTickLabels=np.int64(np.arange(np.min(frameTimes),np.max(frameTimes),2))
        xTickLoc=np.arange(len(xTickLabels))
        for tickCount,tick in enumerate(xTickLabels):
            xTickLoc[tickCount]=np.where(frameTimes>tick)[0][0]
        ax.xaxis.set_ticks(xTickLoc)
        ax.set_xticklabels(xTickLabels,fontsize=10)
        
        #set y-ticks and labels
        nTrialGap=5
        yTickLabels=np.int64(np.arange(nTrialGap,nTrials+1,nTrialGap))
        yTickLoc=np.int64(np.arange(nReps*nTrialGap,nReps*nTrials,nReps*nTrialGap))-1
        ax.yaxis.set_ticks(yTickLoc)
        ax.set_yticklabels(yTickLabels,fontsize=10)
        
        #set stim onset/offset lines
        plt.axvline(x=np.where(frameTimes>stimStartT)[0][0], ymin=0, ymax = nReps*nTrials, linewidth=1, color='k')
        plt.axvline(x=np.where(frameTimes>stimEndT)[0][0], ymin=0, ymax = nReps*nTrials, linewidth=1, color='k')
        
        fig.suptitle(ROI_label+' '+condLabels[c]+' Response', fontsize=14)
        figFile=figOutRoot+'_'+ROI_label+'_timecourse'+condLabels[c]+'_allTrials.png'
        plt.savefig(figFile)
        plt.close()  

def show_ROI_pixel_timecourse(figOutRoot, ROI_label, stimRespROI, frameTimes,\
                              condLabels, stimStartT, stimEndT):
    
    #GET INFO ABOUT DATA
    nPix,nCond,nPts=np.shape(stimRespROI)
    nReps=np.round(np.true_divide(np.true_divide(nPix,nPts),2))
    
    #CHOOSE A RANDOM SUBSET OF PIXELS GET A 1:1 RATIO FIGURE
    permList=np.random.permutation(int(nPix))
    if nPix>nPts*2:
        stimRespROI_figure=stimRespROI[permList[0:nPts*2],:,:]
    else:
        stimRespROI_figure=stimRespROI[permList,:,:]

    #MAKE PLOTS
    for c in range(nCond):
        fig, ax = plt.subplots()
        plt.imshow(np.squeeze(stimRespROI_figure[:,c,:]),cmap='jet',vmin=np.amin(stimRespROI),vmax=np.amax(stimRespROI))
        plt.colorbar()
        plt.xlabel('Time(s)',fontsize=14)
        plt.ylabel('Pixels',fontsize=14)
        
        #set x-ticks and labels
        xTickLabels=np.int64(np.arange(np.min(frameTimes),np.max(frameTimes),2))
        xTickLoc=np.arange(len(xTickLabels))
        for tickCount,tick in enumerate(xTickLabels):
            xTickLoc[tickCount]=np.where(frameTimes>tick)[0][0]
        ax.xaxis.set_ticks(xTickLoc)
        ax.set_xticklabels(xTickLabels,fontsize=10)
        
        
        #set stim onset/offset lines
        plt.axvline(x=np.where(frameTimes>stimStartT)[0][0], ymin=0, ymax = nPix, linewidth=1, color='k')
        plt.axvline(x=np.where(frameTimes>stimEndT)[0][0], ymin=0, ymax = nPix, linewidth=1, color='k')
        
        fig.suptitle(ROI_label+' '+condLabels[c]+' Response', fontsize=14)
        figFile=figOutRoot+'_'+ROI_label+'_timecourse_'+condLabels[c]+'_randomPixels.png'
        plt.savefig(figFile)
        plt.close()  

def plot_ROI_timecourse(sourceRoot,targetRoot, sessID, motionCorrection, analysisDir, stimDur,avgFolder, \
                                ROIfolder, listROI,plotMode='Average',subsetSize=10,\
                                average_conditions=False,average_name=None,newCond=None,condLabels=None):

    #Cesar Echavarria 1/2017

    #DEFINE SOME DIRECTORIES
    inDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';

    ROIdir=targetRoot+'Sessions/'+sessID+'/ROIs/'+ROIfolder+'/'

    ROI_analysisFolder=analysisDir+'/ROIanalysis/'+avgFolder+'/'+ROIfolder+'/'
    if plotMode=='Average':
        figOutDir=ROI_analysisFolder+'/timecourse_average/Figures/'
    elif plotMode=='All':
        figOutDir=ROI_analysisFolder+'/timecourse_allTrials/Figures/'
    elif plotMode=='Subset':
        figOutDir=ROI_analysisFolder+'/timecourse_trialSusbset/Figures/'
    if average_conditions:
        figOutDir=figOutDir+average_name+'/'
    if not os.path.exists(figOutDir):
        os.makedirs(figOutDir)

    if plotMode=='Average':
        fileOutDir=ROI_analysisFolder+'/timecourse_average/Files/'
    elif plotMode=='All':
        fileOutDir=ROI_analysisFolder+'/timecourse_allTrials/Files/'
    elif plotMode=='Subset':
        fileOutDir=ROI_analysisFolder+'/timecourse_trialSusbset/Files/'
    if average_conditions:
        fileOutDir=fileOutDir+average_name+'/'
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=fileOutDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()



    #LOAD MISCELlANEOUS
    inFile=inDir+sessID+'_timecourse_miscelleanous.npz'
    f=np.load(inFile)
    startT=f['timecourse_startT']
    stimStartT=f['baseline_endT']
    frameRate=f['frameRate']
    percentSignalChange=f['percentSignalChange']
    nCond=f['nCond']
    nPix=f['nPix']
    nTrials=f['nTrials']
    respPts=f['respPts']

    sampRate=np.true_divide(1,frameRate)


    if plotMode=='Average':
        #LOAD IN TIME COURSE DATA
        inFile=inDir+sessID+'_timecourse_trialAvg.npz'
        f=np.load(inFile)
        stimRespMean=f['stimRespMean']


        #GET SOME INFO ABOUT DATA
        nPix,nCond,respPts=np.shape(stimRespMean)
        frameTimes=np.linspace(startT,startT+(respPts)*sampRate,respPts)
        if average_conditions:
            allNewCond=np.sort(np.unique(newCond))
            nCond=np.size(allNewCond)
            stimRespMeanTmp=np.zeros((nPix,nCond,respPts))
            for newC in allNewCond:                
                condInd=np.where(newCond==newC)[0]
                if np.size(condInd)>1:
                    stimRespMeanTmp[:,newC-1,:]=np.mean(stimRespMean[:,condInd,:],1)
                else:
                    stimRespMeanTmp[:,newC-1,:]=stimRespMean[:,condInd,:]
            stimRespMean=stimRespMeanTmp
            del stimRespMeanTmp
        else:
            condLabels=contrast_dictionary_to_labels(targetRoot,nCond)

    elif plotMode=='All' or plotMode=='Subset':
        #LOAD IN TIME COURSE DATA
        responseAll=np.zeros((nPix,nCond,nTrials,respPts))
        for cond in range(nCond):
            inFile=inDir+sessID+'_'+str(cond)+'_timecourse_allTrials.npz'
            if os.path.isfile(inFile):
                f=np.load(inFile)
                responseAll[:,cond,:,:]=f['responseAll']
            else:
                #LOAD BY PARTS
                inFile=glob.glob(inDir+sessID+'_'+str(cond)+'_timecourse_allTrials_part1*')[0]
                f=np.load(inFile)
                tmp=f['responseAll']
                startTrial=f['startTrial']
                endTrial=f['endTrial']
                nParts=f['nParts']

                responseAll[:,cond,startTrial:endTrial,:]=tmp

                for p in range(1,nParts):
                    inFile=glob.glob(inDir+sessID+'_'+str(cond)+'_timecourse_allTrials_part'+str(p+1)+'*')[0]
                    f=np.load(inFile)
                    tmp=f['responseAll']
                    startTrial=f['startTrial']
                    endTrial=f['endTrial']
                    responseAll[:,cond,startTrial:endTrial,:]=tmp



        #GET SOME INFO ABOUT DATA
        frameTimes=np.linspace(startT,startT+(respPts)*sampRate,respPts)
        if average_conditions:
            allNewCond=np.sort(np.unique(newCond))
            nCond=np.size(allNewCond)
            responseAllTmp=np.zeros((nPix,nCond,nTrials,respPts))
            for newC in allNewCond:                
                condInd=np.where(newCond==newC)[0]
                if np.size(condInd)>1:
                    responseAllTmp[:,newC-1,:,:]=np.mean(responseAll[:,condInd,:,:],1)
                else:
                    responseAllTmp[:,newC-1,:,:]=stimRespMean[:,condInd,:,:]
            responseAll=responseAllTmp
            del responseAllTmp

        else:
            condLabels=contrast_dictionary_to_labels(targetRoot,nCond)

    #LOAD ROI INDICES AND DETAILS
    if plotMode=='Average':
            print('Plotting average timecourse within ROIs....')
    elif plotMode=='All':
         print('Plotting timecourse for all trials within ROIs....')
    elif plotMode=='Subset':
         print('Plotting timecourse for a subset of trials within ROIs....')


    for roi in listROI:
        print(roi)
        # load file and variables
        inFile=ROIdir+'/Files/'+roi+'.npz'
        f=np.load(inFile)
        ROI_ind=f['ROIind']

        # account for motion correction boundaries, if necessary
        if motionCorrection:
            #GET MOTION CORRECTED BOUNDARIES
            motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
            motionFileDir=motionDir+'Registration/'
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']
            edgeUp=boundaries[0]
            edgeDown=boundaries[1]
            edgeLeft=boundaries[2]
            edgeRight=boundaries[3]
            #GET MOTION CORRECTED FRAME SIZE
            szY=edgeDown-edgeUp
            szX=edgeRight-edgeLeft

            #GET ROI INDICES WITHIN BOUNDARY
            imRef=get_reference_frame(sourceRoot,sessID)
            ROIimg=np.zeros(imRef.shape)
            ROIimg[ROI_ind[0],ROI_ind[1]]=1
            ROIimg=ROIimg[edgeUp:edgeDown,edgeLeft:edgeRight]
            ROI_ind=np.where(ROIimg==1)
        else:
            imRef=get_reference_frame(sourceRoot,sessID)
            szY,szX=np.shape(imRef)

        #convert to linear indices
        ROI_ind=np.ravel_multi_index(ROI_ind,(szY,szX))    


        ROI_label=roi
        if plotMode=='Average':
            #GET ROI PIXELS
            stimRespPixels=stimRespMean[ROI_ind,:,:]

            #PLOT A SUBSET OF PIXEL TIMECOURSES 
            figOutRoot=figOutDir+sessID
            show_ROI_pixel_timecourse(figOutRoot, ROI_label, stimRespPixels, frameTimes,\
                              condLabels, stimStartT, stimStartT+stimDur)
            #AVERAGE ACROSS PIXELS IN ROI
            if np.size(ROI_ind)>1:
                stimRespROI=np.mean(stimRespPixels,0)
            else:
                stimRespROI=stimRespPixels
            del stimRespPixels

            #PLOT ROI
            plot_ROI_average_timecourse(figOutDir, sessID, ROI_label, stimRespROI, frameTimes,\
                                    condLabels, stimStartT, stimStartT+stimDur, percentSignalChange)

            #SAVE TIMECOURSE TO FILE
            outFile=fileOutDir+ROI_label
            np.savez(outFile,stimRespROI=stimRespROI,frameTimes=frameTimes,condLabels=condLabels,stimStartT=stimStartT,\
                    stimEndT=stimStartT+stimDur)
        elif plotMode=='All':
            #AVERAGE ACROSS PIXELS IN ROI
            if np.size(ROI_ind)>1:
                stimRespROI=np.mean(responseAll[ROI_ind,:,:,:],0)
            else:
                stimRespROI=responseAll[ROI_ind,:,:,:]

            #PLOT ROI
            figFile=figOutDir+sessID+'_'+ROI_label+'_timecourse_allTrials.png'
            plot_ROI_trial_timecourse(figFile, ROI_label, stimRespROI, frameTimes,\
                                    condLabels, stimStartT, stimStartT+stimDur, percentSignalChange)
            #JUXTAPOSE ALL RESPONSES
            
            show_ROI_trial_timecourse(figOutDir+sessID, ROI_label, stimRespROI, frameTimes,\
                                    condLabels, stimStartT, stimStartT+stimDur)
            #SAVE TIMECOURSE TO FILE
            outFile=fileOutDir+ROI_label
            np.savez(outFile,stimRespROI=stimRespROI,frameTimes=frameTimes,condLabels=condLabels,stimStartT=stimStartT,\
                    stimEndT=stimStartT+stimDur)
        elif plotMode=='Subset':
            #AVERAGE ACROSS PIXELS IN ROI
            if np.size(ROI_ind)>1:
                stimRespROI=np.mean(responseAll[ROI_ind,:,:,:],0)
            else:
                stimRespROI=responseAll[ROI_ind,:,:,:]

            #CHOOSE SUBSET OF TRIALS
            stimRespROI_subset=np.zeros((nCond,subsetSize,respPts))
            for cond in range(nCond):
                permList=np.random.permutation(int(nTrials))
                stimRespROI_subset[cond,:,:]=stimRespROI[cond,permList[0:subsetSize],:]


            #PLOT ROI
            figFile=figOutDir+sessID+'_'+ROI_label+'_timecourse_'+str(subsetSize)+'Trials.png'
            plot_ROI_trial_timecourse(figFile, ROI_label, stimRespROI_subset, frameTimes,\
                                    condLabels, stimStartT, stimStartT+stimDur, percentSignalChange)
            #SAVE TIMECOURSE TO FILE
            outFile=fileOutDir+ROI_label
            np.savez(outFile,stimRespROI=stimRespROI_subset,frameTimes=frameTimes,condLabels=condLabels,stimStartT=stimStartT,\
                    stimEndT=stimStartT+stimDur)



def plot_ROI_response_single_subject(figOutDir,sessID,ROIdata,listROI,condLabels,figLabel=None,condOrder=None):
    #Cesar Echavarria 1/2017
    
    #Get data details
    nROI,nTrials,nCond=np.shape(ROIdata)
    ROIdata_mean=np.mean(ROIdata,1)

    if condOrder is None:
        condOrder=np.arange(0,nCond)

    else:
        condOrder=condOrder-1
    

    colorList='bgrmkc'
    #figure out location of data in plot
    condGap=0.2
    ROIgap=0.5
    plotLoc=np.empty((nROI,nCond))
    loc=0
    for r in range(nROI):
        if r > 0:
            loc = loc+ROIgap
        for c in condOrder:
            if c > 0:
                loc = loc+condGap
            plotLoc[r,c]=loc


    fig, ax = plt.subplots()
    legHand=[]
    #draw line at mean response
    for r in range(nROI):
        for c in range(nCond):
            ax.hlines(y=ROIdata_mean[r,c],xmin=plotLoc[r,c]-np.true_divide(condGap,2),xmax=plotLoc[r,c]+np.true_divide(condGap,2),colors='k')
            
    #draw trial-by-trial response
    for r in range(nROI):
        for c in range(nCond):
            lh=plt.scatter(np.ones(nTrials)*plotLoc[r,c],ROIdata[r,:,c],c=colorList[c],marker='o')
            if r == 0:
                legHand.append(lh)
        if r == 0:
            #draw legend
            plt.legend(legHand,condLabels,scatterpoints=1,fontsize=8,loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3)
    #adjust axes labels
    plt.xlabel('ROI',fontsize=16)
    plt.ylabel('Signal Change (%)',fontsize=16)
    plt.xlim([np.min(plotLoc)-condGap,np.max(plotLoc)+condGap])
    ax.xaxis.set_ticks(np.mean(plotLoc,1))
    if nROI <4:
        ax.set_xticklabels(listROI,fontsize=14)
    else:
        ax.set_xticklabels(listROI,fontsize=10)

    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    if ymin<=0:
        xmin, xmax = axes.get_xlim()
        plt.axhline(xmin=xmin, xmax=xmax , y= 0, linewidth=1, color='k')

    fig.suptitle(sessID+' ROI response', fontsize=16)
    if figLabel is None:
        plt.savefig(figOutDir+sessID+'_PSC_All_Trials.png')
    else:
        plt.savefig(figOutDir+sessID+'_PSC_All_Trials_'+figLabel+'.png')
    plt.close()

def plot_ROI_tWindow(sourceRoot,targetRoot, sessID, motionCorrection, analysisDir, avgFolder, \
                     ROIfolder, listROI, \
                     average_conditions=False,average_name=None,newCond=None,condLabels=None,\
                     figLabel=None,condOrder=None):


    #Cesar Echavarria 1/2017

    #DEFINE SOME DIRECTORIES
    inDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';

    ROIdir=targetRoot+'Sessions/'+sessID+'/ROIs/'+ROIfolder+'/'

    ROI_analysisFolder=analysisDir+'/ROIanalysis/'+avgFolder+'/'+ROIfolder+'/'

    figOutDir=ROI_analysisFolder+'/Figures/'
    if average_conditions:
        figOutDir=figOutDir+average_name+'/'
    if not os.path.exists(figOutDir):
        os.makedirs(figOutDir)

    fileOutDir=ROI_analysisFolder+'/Files/'
    if average_conditions:
        fileOutDir=fileOutDir+average_name+'/'
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=fileOutDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()


    #LOAD PERCENT SIGNAL CHANGE DATA AND OTHER VARIABLES
    inFile=inDir+sessID+'_PSC_AllTrials.npz'
    f=np.load(inFile)
    PSC=f['PSC_AllTrials']*100
    nTrials=f['nTrials']
    nPix=f['nPix']
    nCond=f['nCond']

    
    if average_conditions:
        allNewCond=np.sort(np.unique(newCond))
        nCond=np.size(allNewCond)
        PSC_tmp=np.zeros((nPix,nTrials,nCond))

        for newC in allNewCond:                
            condInd=np.where(newCond==newC)[0]
            if np.size(condInd)>1:
                PSC_tmp[:,:,newC-1]=np.mean(PSC[:,:,condInd],2)
            else:
                PSC_tmp[:,:,newC-1]=PSC[:,:,condInd]
        PSC=PSC_tmp
        del PSC_tmp
    else:
        condLabels=contrast_dictionary_to_labels(targetRoot,nCond)

    #INITIALIZE ARRAY FOR DATA STORAGE
    nROI=len(listROI)
    ROIdata=np.zeros((nROI,nTrials,nCond))


    #LOAD ROI INDICES AND DETAILS
    print('Getting ROI response....')

    count=0
    for roi in listROI:
        print(roi)
        # load file and variables
        inFile=ROIdir+'/Files/'+roi+'.npz'
        f=np.load(inFile)
        ROI_ind=f['ROIind']

        # account for motion correction boundaries, if necessary
        if motionCorrection:
            #GET MOTION CORRECTED BOUNDARIES
            motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
            motionFileDir=motionDir+'Registration/'
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']
            edgeUp=boundaries[0]
            edgeDown=boundaries[1]
            edgeLeft=boundaries[2]
            edgeRight=boundaries[3]
            #GET MOTION CORRECTED FRAME SIZE
            szY=edgeDown-edgeUp
            szX=edgeRight-edgeLeft

            #GET ROI INDICES WITHIN BOUNDARY
            imRef=get_reference_frame(sourceRoot,sessID)
            ROIimg=np.zeros(imRef.shape)
            ROIimg[ROI_ind[0],ROI_ind[1]]=1
            ROIimg=ROIimg[edgeUp:edgeDown,edgeLeft:edgeRight]
            ROI_ind=np.where(ROIimg==1)
        else:
            imRef=get_reference_frame(sourceRoot,sessID)
            szY,szX=np.shape(imRef)

        #convert to linear indices
        ROI_ind=np.ravel_multi_index(ROI_ind,(szY,szX))  

        #STORE ROI RESPONSE IN ARRAY
        if np.size(ROI_ind)>1:
            ROIdata[count,:,:]=np.mean(PSC[ROI_ind,:,:],0)
        else:
            ROIdata[count,:,:]=PSC[ROI_ind,:,:]
        count=count+1


    print('Plotting ROI response....')
    plot_ROI_response_single_subject(figOutDir,sessID,ROIdata,listROI,condLabels,figLabel,condOrder)


    #SAVE DATA FOR EACH ROI
    count=0
    ROIdata_mean=np.mean(ROIdata,1)#average across trials
    for roi in listROI:
        outFile=fileOutDir+sessID+'_'+roi+'_PSC'
        savedData=ROIdata_mean[count,:]
        np.savez(outFile,ROIdata=savedData,condLabels=condLabels)
        count=count+1


def plot_ROI_aggregate_tCourse(stimRespROI, frameTimes, stimStartT, stimEndT, listROI, \
    condLabels, figOutDir, dataExists=None, groupSessions=True, normData=False):
    if dataExists is None:
        dataExists=np.ones(np.shape(stimRespROI))
    
    nSess,nROI,nCond,nPts=np.shape(stimRespROI)
    #NORMALIZE DATA
    if normData:
        tmpResp=np.zeros(np.shape(stimRespROI))
        for s in range(nSess):
            validInd=np.where(dataExists[s,:])[0]
            tmp=stimRespROI[s,validInd,:,:]
            tmpResp[s,validInd,:,:]=np.true_divide(tmp,np.max(tmp))
        stimRespROI=tmpResp
        del tmpResp
        
    #GET MEAN ACROSS SESSIONS
    if np.all(dataExists):
        stimRespROI_mean=np.mean(stimRespROI,0)
    else:
        stimRespROI_mean=np.zeros((nROI,nCond,nPts))
        for r in range(nROI):
            validInd=np.where(dataExists[:,r])[0]
            tmp=stimRespROI[validInd,r,:,:]
            stimRespROI_mean[r,:,:]=np.mean(tmp,0)

    for roiCount,roi in enumerate(listROI):
        #MAKE PLOT, SESSION-BY-SESSION
        colorList='bgrmkc'
        legHand=[]
        fig=plt.figure()
        print(roi)
        for s in range(nSess):
            for c in range(nCond):
                if dataExists[s,roiCount]:
                    plt.plot(frameTimes,stimRespROI[s,roiCount,c,:],colorList[c],linewidth=1)
                    if s==0:
                        legHand.append(mlines.Line2D([], [], color=colorList[c], marker='_',
                                              markersize=10, label=condLabels[c]))
        plt.legend(handles=legHand)

        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        if normData:
            ymax=1
        plt.axvline(x=stimStartT, ymin=ymin, ymax = ymax, linewidth=1, color='k')
        plt.axvline(x=stimEndT, ymin=ymin, ymax = ymax, linewidth=1, color='r')
        if ymin<=0:
            xmin, xmax = axes.get_xlim()
            plt.axhline(xmin=xmin, xmax=xmax , y= 0, linewidth=1, color='k')

        fig.suptitle(roi+' Response', fontsize=14)
        plt.xlabel('Time (secs)',fontsize=14)
        if normData:
            plt.ylabel('Response Magnitude',fontsize=14)
        else:    
            plt.ylabel('PSC',fontsize=14)
        if normData:
            figOutFile=figOutDir+roi+'_timecourse_aggregate_normalized.png'
        else:
            figOutFile=figOutDir+roi+'_timecourse_aggregate.png'

        plt.savefig(figOutFile)
        plt.close()

        #MAKE PLOT; MEAN
        colorList='bgrmkc'
        legHand=[]
        fig=plt.figure()

        for c in range(nCond):
            plt.plot(frameTimes,stimRespROI_mean[roiCount,c,:],colorList[c],linewidth=1)

            legHand.append(mlines.Line2D([], [], color=colorList[c], marker='_',
                                  markersize=10, label=condLabels[c]))
        plt.legend(handles=legHand)

        axes = plt.gca()
        if nCond>3:
            plt.legend(handles=legHand,fontsize=8,loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3)
        else:
            plt.legend(handles=legHand)

        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        if normData:
            ymax=1

        if nCond>3:
            axes.set_ylim(ymin,ymax+(.2*ymax))
            ymin, ymax = axes.get_ylim()
        plt.axvline(x=stimStartT, ymin=ymin, ymax = ymax, linewidth=1, color='k')
        plt.axvline(x=stimEndT, ymin=ymin, ymax = ymax, linewidth=1, color='r')
        if ymin<=0:
            xmin, xmax = axes.get_xlim()
            plt.axhline(xmin=xmin, xmax=xmax , y= 0, linewidth=1, color='k')

        fig.suptitle(roi+' Mean Response', fontsize=14)
        plt.xlabel('Time (secs)',fontsize=14)
        if normData:
            plt.ylabel('Response Magnitude',fontsize=14)
        else:    
            plt.ylabel('PSC',fontsize=14)

        if normData:
            figOutFile=figOutDir+roi+'_timecourse_mean_normalized.png'
        else:
            figOutFile=figOutDir+roi+'_timecourse_mean.png'

        plt.savefig(figOutFile)
        plt.close()



def plot_ROI_aggregate_tWindow(ROIdata, figOutFile, listROI, condLabels, dataExists=None, \
                               groupSessions=True, normSess=False, normROI=False,condOrder=None):

    if dataExists is None:
        dataExists=np.ones(np.shape(ROIdata))
    #Get data details
    nSess,nROI,nCond=np.shape(ROIdata)

    if condOrder is None:
        condOrder=np.arange(0,nCond)
    else:
        condOrder=condOrder-1

    #normalize response per session, if necessary
    if normSess:
        for s in range(nSess):
            validInd=np.where(dataExists[s,:])[0]
            tmp=ROIdata[s,validInd,:]
            tmp=np.true_divide(tmp,np.amax(tmp))
            ROIdata[s,validInd,:]=tmp
            
    #normalize response per session and ROI, if necessary
    if normROI:
        for s in range(nSess):
            for r in range(nROI):
                if dataExists[s,r]:
                    tmp=ROIdata[s,r,:]
                    tmp=np.true_divide(tmp,np.amax(tmp))
                    ROIdata[s,r,:]=tmp


    #Get mean response per ROI
    if np.all(dataExists):
        ROIdata_mean=np.mean(ROIdata,0)
    else:
        ROIdata_mean=np.zeros((nROI,nCond))
        for r in range(nROI):
            validInd=np.where(dataExists[:,r])[0]
            tmp=ROIdata[validInd,r,:]
            ROIdata_mean[r,:]=np.mean(tmp,0)

    colorList='bgrmkc'
    #figure out location of data in plot
    condGap=0.2
    ROIgap=0.5
    plotLoc=np.empty((nROI,nCond))
    loc=0
    for r in range(nROI):
        if r > 0:
            loc = loc+ROIgap
        for c in condOrder:
            if c > 0:
                loc = loc+condGap
            plotLoc[r,c]=loc


    fig, ax = plt.subplots()
    legHand=[]
    #draw line at mean response
    for r in range(nROI):
        for c in range(nCond):
            ax.hlines(y=ROIdata_mean[r,c],xmin=plotLoc[r,c]-np.true_divide(condGap,2),xmax=plotLoc[r,c]+np.true_divide(condGap,2),colors='k',linewidths=2)

    if groupSessions:
        #connect lines per session
        for s in range(nSess):
            for r in range(nROI):
                if dataExists[s,r]:
                    plt.plot(plotLoc[r,condOrder],ROIdata[s,r,condOrder],'k')

    #draw session-by-session
    for r in range(nROI):
        for c in range(nCond):
            if np.all(dataExists):
                lh=plt.scatter(np.ones(nSess)*plotLoc[r,c],ROIdata[:,r,c],c=colorList[c],marker='o')
            else:
                validInd=np.where(dataExists[:,r])[0]
                lh=plt.scatter(np.ones(len(validInd))*plotLoc[r,c],ROIdata[validInd,r,c],c=colorList[c],marker='o')
            if r == 0:
                legHand.append(lh)
        if r == 0:
            #draw legend
            plt.legend(legHand,condLabels,scatterpoints=1,fontsize=8,loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3)
    #adjust axes labels
    plt.xlabel('ROI',fontsize=16)
    plt.ylabel('Signal Change (%)',fontsize=16)
    
    plt.xlim([np.min(plotLoc)-condGap,np.max(plotLoc)+condGap])
#     plt.xlim([np.min(plotLoc)-condGap,np.max(plotLoc)+(nCond*condGap)+ROIgap])
    ax.xaxis.set_ticks(np.mean(plotLoc,1))
    if nROI <4:
        ax.set_xticklabels(listROI,fontsize=14)
    else:
        ax.set_xticklabels(listROI,fontsize=10)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    if ymin<=0:
        xmin, xmax = axes.get_xlim()
        plt.axhline(xmin=xmin, xmax=xmax , y= 0, linewidth=1, color='k')
    
    fig.suptitle('ROI response', fontsize=16)
    plt.savefig(figOutFile)
    plt.close()
    
def ROI_analysis_aggregate_tWindow(targetRoot, analysisPath, nCond,avgFolder,\
                                   aggregateName, sessIDlist, ROIfolder, listROI, makeFigures=True,\
                                   condOrder=None):

    #Cesar Echavarria 1/2017


    #DEFINE SOME DIRECTORIES
    ROIoutDir=targetRoot+'/ROIanalyses/'+aggregateName+'/'

    figOutDir=ROIoutDir+'Figures/tWindow/'
    if not os.path.exists(figOutDir):
            os.makedirs(figOutDir) 

    fileOutDir=ROIoutDir+'Files/'
    if not os.path.exists(fileOutDir):
            os.makedirs(fileOutDir) 
            
 #   OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=fileOutDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()

    #INTIALIZE EMPTY ARRAY
    nSess=len(sessIDlist)
    nROI=len(listROI)
    ROIdataAll=np.zeros((nSess,nROI,nCond))
    dataExists=np.empty((nSess,nROI))

    #GO THROUGH EACH SESSION
    for sessCount,sessID in enumerate(sessIDlist):


        #DEFINE SOME DIRECTORIES
        analysisDir=targetRoot+'/Sessions/'+sessID+analysisPath
        ROI_analysisFolder=analysisDir+'/ROIanalysis/'+avgFolder+'/'+ROIfolder+'/'
        fileInDir=ROI_analysisFolder+'/Files/'


        #GET ROI DATA
        for roiCount,roi in enumerate(listROI):
            inFile=fileInDir+sessID+'_'+roi+'_PSC.npz'
            if os.path.isfile(inFile):
                dataExists[sessCount,roiCount]=True
                f=np.load(inFile)
                ROIdata=f['ROIdata']
                ROIdataAll[sessCount,roiCount,:]=ROIdata
            else:
                dataExists[sessCount,roiCount]=False

        condLabels=f['condLabels']
    #SAVE DATA
    outFile=fileOutDir+aggregateName+'_ROIdata'
    np.savez(outFile,ROIdata=ROIdataAll,dataExists=dataExists,condLabels=condLabels,sessIDlist=sessIDlist,listROI=listROI)

    #WRITE sessID list to file
    outFile=fileOutDir+'sessID_list.txt'
    sessIDTextFile = open(outFile, 'w+')
    sessIDTextFile.write('LIST OF SESSIONS\n')
    sessIDTextFile.write('\n')
    for sess in sessIDlist:
        sessIDTextFile.write(sess+'\n')
    sessIDTextFile.close()
    
    #OUT FIGURES
    if makeFigures:
        figOutFile=figOutDir+'PSC_Aggregate.png'
        plot_ROI_aggregate_tWindow(ROIdataAll, figOutFile, listROI, condLabels, dataExists,\
                                   groupSessions=True,condOrder=condOrder)
        figOutFile=figOutDir+'PSC_Aggregate_NormSess.png'
        plot_ROI_aggregate_tWindow(ROIdataAll, figOutFile, listROI, condLabels,\
                                   dataExists, groupSessions=True, normSess=True,condOrder=condOrder)
        figOutFile=figOutDir+'PSC_Aggregate_NormROI.png'
        plot_ROI_aggregate_tWindow(ROIdataAll, figOutFile, listROI, condLabels,\
                                   dataExists, groupSessions=True, normROI=True,condOrder=condOrder)
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# PERIODIC STIM CODE

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_condition_list(sourceRoot,sessID,runList):
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if runList is None:
        raise TypeError("runList not specified!")

    condList=np.zeros(len(runList))
    for idx,run in enumerate(runList):
        #DEFINE DIRECTORIES
        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')
        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"
        frameTimes,frameCond,frameCount = get_frame_times(planFolder)
        condList[idx]=frameCond[0]
    return condList

def exclude_edge_periods(frameArray,frameTimes,stimFreq):
    tStart=np.true_divide(1,stimFreq)
    nCycles=np.round(frameTimes[-1]/np.true_divide(1,stimFreq))
    tEnd=np.true_divide(1,stimFreq)*(nCycles-1)
    
    indStart=np.where(frameTimes>tStart)[0][0]
    indEnd=np.where(frameTimes>tEnd)[0][0]-1
    
    arrayNew=frameArray[:,indStart:indEnd]
    frameTimesNew=frameTimes[indStart:indEnd]
    
    return arrayNew,frameTimesNew

def get_analysis_path_phase(analysisRoot, targetFreq, interp=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False,averageFrames=None):
    #Cesar Echavarria 2/2017

    interpString=''
    excludeString=''
    averageString=''
    removeRollingString=''
    #DEFINE DIRECTORIES
    if motionCorrection:
        imgOperationDir='motionCorrection'
    else:
        imgOperationDir='noMotionCorrection'


    procedureDir=''
    if interp:
        interpString='interpolate_'
          
    if excludeEdges:
        excludeString='excludeEdges_'

    if averageFrames is not None:
        averageString='averageFrames_'+str(averageFrames)+'_'

    if removeRollingMean:
        removeRollingString='minusRollingMean'

    procedureDir=interpString+excludeString+averageString+removeRollingString

    phaseDir='%s/phase/%s/%s/targetFreq_%sHz/'%(analysisRoot,imgOperationDir,procedureDir,targetFreq)


    return phaseDir


def analyze_periodic_data_per_run(sourceRoot, targetRoot, sessID, runList, stimFreq, frameRate, \
    interp=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False,averageFrames=None,loadCorrectedFrames=True,saveImages=True,makeMovies=True,\
    stimType=None,mask=None):
     # DEFINE DIRECTORIES
    anatSource=targetRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'

    analysisRoot=targetRoot+'/Sessions/'+sessID+'/Analyses/';
    analysisDir=get_analysis_path_phase(analysisRoot, stimFreq, interp, excludeEdges,removeRollingMean, \
        motionCorrection,averageFrames)

    fileOutDir=analysisDir+'SingleRunData/Files/'
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    if saveImages:
        figOutDirRoot=analysisDir+'SingleRunData/Figures/'


    #    # OUTPUT TEXT FILE WITH PACKAGE VERSION
    # outFile=fileOutDir+'analysis_version_info.txt'
    # versionTextFile = open(outFile, 'w+')
    # versionTextFile.write('WiPy version '+__version__+'\n')
    # versionTextFile.close()

    condList = get_condition_list(sourceRoot,sessID,runList)
    if interp:
        #GET INTERPOLATION TIME
        newStartT,newEndT=get_interp_extremes(sourceRoot,sessID,runList,stimFreq)
        newTimes=np.arange(newStartT+(1.0/frameRate),newEndT,1.0/frameRate)#always use the same time points


    for runCount,run in enumerate(runList):
        print('run='+str(run))
        #DEFINE DIRECTORIES
        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')
        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"


        # READ IN FRAME TIMES FILE
        frameTimes,frameCond,frameCount=get_frame_times(planFolder)
        cond=frameCond[0]

        if saveImages:
            figOutDir=figOutDirRoot+'cond'+str(int(cond))+'/'
            if not os.path.exists(figOutDir):
                os.makedirs(figOutDir)

        #READ IN FRAMES
        print('Loading frames...')
        if motionCorrection:

            if loadCorrectedFrames:
                #LOAD MOTION CORRECTED FRAMES
                inFile=motionFileDir+sessID+'_run'+str(run)+'_correctedFrames.npz'
                f=np.load(inFile)
                frameArray=f['correctedFrameArray']
            else:
                #GET REFFERNCE FRAME
                imRef=get_reference_frame(sourceRoot,sessID,runList[0])
                szY,szX=imRef.shape

                # READ IN FRAMES
                frameArray=np.zeros((szY,szX,frameCount))
                for f in range (0,frameCount):
                    imFile=frameFolder+'frame'+str(f)+'.tiff'
                    im0=misc.imread(imFile)
                    frameArray[:,:,f]=np.copy(im0)

                #-> load warp matrices
                inFile=motionFileDir+sessID+'_run'+str(run)+'_motionRegistration.npz'
                f=np.load(inFile)
                warpMatrices=f['warpMatrices']

                #APPLY MOTION CORRECTION
                frameArray=apply_motion_correction(frameArray,warpMatrices)

            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']

            #APPLY BOUNDARIES
            frameArray = apply_motion_correction_boundaries(frameArray,boundaries)

            szY,szX = np.shape(frameArray[:,:,0])


        else:
            #GET REFFERNCE FRAME
            imRef=get_reference_frame(sourceRoot,sessID,runList[0])
            szY,szX=imRef.shape

            # READ IN FRAMES
            frameArray=np.zeros((szY,szX,frameCount))
            for f in range (0,frameCount):
                imFile=frameFolder+'frame'+str(f)+'.tiff'
                im0=misc.imread(imFile)
                frameArray[:,:,f]=np.copy(im0)

        frameArray=np.reshape(frameArray,(szY*szX,frameCount))

        #INTERPOLATE FOR CONSTANT FRAME RATE
        if interp:
            print('Interpolating....')
            frameArray=interpolate_array(frameTimes,frameArray,newTimes)
            frameTimes=newTimes

        #EXCLUDE FIRST AND LAST PERIOD
        if excludeEdges:
                print('Excluding first and last periods...')
                frameArray,frameTimes=exclude_edge_periods(frameArray,frameTimes,stimFreq)

        meanPixelValue=np.mean(frameArray,1)

        # REMOVE ROLLING AVERAGE
        if removeRollingMean:

            print('Removing rolling mean....')
            detrendedFrameArray=np.zeros(np.shape(frameArray))
            rollingWindowSz=int(np.ceil((np.true_divide(1,stimFreq)*2)*frameRate))

            for pix in range(0,np.shape(frameArray)[0]):

                tmp0=frameArray[pix,:];
                tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                rollingAvg=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                rollingAvg=rollingAvg[rollingWindowSz:-rollingWindowSz]


                detrendedFrameArray[pix,:]=np.subtract(tmp0,rollingAvg)
            frameArray=detrendedFrameArray
            del detrendedFrameArray

        #AVERAGE FRAMES
        if averageFrames is not None:
            print('Pooling frames...')
            smoothFrameArray=np.zeros(np.shape(frameArray))
            rollingWindowSz=averageFrames

            for pix in range(0,np.shape(frameArray)[0]):

                tmp0=frameArray[pix,:];
                tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                tmp2=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                tmp2=tmp2[rollingWindowSz:-rollingWindowSz]

                smoothFrameArray[pix,:]=tmp2
            frameArray=smoothFrameArray
            del smoothFrameArray

        #Get FFT
        print('Analyzing phase and magnitude....')
        fourierData = np.fft.fft(frameArray)
        #Get magnitude and phase data
        magData=abs(fourierData)
        phaseData=np.angle(fourierData)

        signalLength=np.shape(frameArray)[1]
        freqs = np.fft.fftfreq(signalLength, float(1/frameRate))
        idx = np.argsort(freqs)

        freqs=freqs[idx]
        magData=magData[:,idx]
        phaseData=phaseData[:,idx]

        freqs=freqs[np.round(signalLength/2)+1:]#excluding DC offset
        magData=magData[:,np.round(signalLength/2)+1:]#excluding DC offset
        phaseData=phaseData[:,np.round(signalLength/2)+1:]#excluding DC offset


        freqIdx=np.argmin(np.absolute(freqs-stimFreq))
        topFreqIdx=np.where(freqs>1)[0][0]

        #GET PERCENT SIGNAL MODULATION
        meanPixelValue=np.expand_dims(meanPixelValue,1)
        meanPixelValue=np.tile(meanPixelValue,(1,frameArray.shape[1]))
        frameArrayPSC=np.true_divide(frameArray,meanPixelValue)*100

        #OUTPUT TEXT FILE FREQUENCY CHANNEL ANALYZED
        if runCount == 0:

            outFile=fileOutDir+'frequency_analyzed.txt'
            freqTextFile = open(outFile, 'w+')
            freqTextFile.write('RUN '+str(run)+' '+str(np.around(freqs[freqIdx],4))+' Hz\n')

        if saveImages:

            maxModIdx=np.argmax(magData[:,freqIdx],0)
            outFile = figOutDir+sessID+'_run'+str(run)+'_magnitudePlot.png'
            fig=plt.figure()
            plt.plot(freqs,magData[maxModIdx,:])
            fig.suptitle(sessID+' run '+str(run)+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(outFile)
            plt.close()

            outFile = figOutDir+sessID+'_run'+str(run)+'_magnitudePlot_zoom.png'
            fig=plt.figure()
            plt.plot(freqs[0:topFreqIdx],magData[maxModIdx,0:topFreqIdx])
            fig.suptitle(sessID+' run '+str(run)+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(outFile)
            plt.close()

            stimPeriod_t=np.true_divide(1,stimFreq)
            stimPeriod_frames=stimPeriod_t*frameRate
            periodStartFrames=np.round(np.arange(0,len(frameTimes),stimPeriod_frames))

            outFile = figOutDir+sessID+'_run'+str(run)+'_timecourse.png'
            fig=plt.figure()

            plt.plot(frameTimes,frameArrayPSC[maxModIdx,:])
            fig.suptitle(sessID+' run '+str(run)+' timecourse', fontsize=20)
            plt.xlabel('Time (s)',fontsize=16)
            plt.ylabel('Signal Change (%)',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            if interp:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            else:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            axes.set_xlim([frameTimes[0],frameTimes[-1]])
            plt.savefig(outFile)
            plt.close()

        #index into target frequecny and reshape arrays to maps
        magArray=magData[:,freqIdx]
        magMap=np.reshape(magArray,(szY,szX))
        phaseArray=phaseData[:,freqIdx]
        phaseMap=np.reshape(phaseArray,(szY,szX))

        #set phase map range for visualization
        phaseMapDisplay=np.copy(phaseMap)
        phaseMapDisplay[phaseMap<0]=-phaseMap[phaseMap<0]
        phaseMapDisplay[phaseMap>0]=(2*np.pi)-phaseMap[phaseMap>0]

        #get various measures of data quality
        #1) magnitude and ratio of magnitude
        tmp=np.copy(magData)
        np.delete(tmp,freqIdx,1)
        nonTargetMagArray=np.sum(tmp,1)
        magRatio=magArray/nonTargetMagArray
        magRatioMap=np.reshape(magRatio,(szY,szX))
        nonTargetMagMap=np.reshape(nonTargetMagArray,(szY,szX))

        #2) amplitude and variance expained

        t=frameTimes*(2*np.pi)*stimFreq
        t=np.transpose(np.expand_dims(t,1))
        tMatrix=np.tile(t,(phaseData.shape[0],1))

        phi=np.expand_dims(phaseArray,1)
        phiMatrix=np.tile(phi,(1,frameArray.shape[1]))
        Xmatrix=np.cos(tMatrix+phiMatrix)

        betaArray=np.zeros((frameArray.shape[0]))
        varExpArray=np.zeros((frameArray.shape[0]))

        for pix in range(frameArray.shape[0]):
            x=np.expand_dims(Xmatrix[pix,:],1)
            y=frameArrayPSC[pix,:]
            beta=np.matmul(np.linalg.pinv(x),y)
            betaArray[pix]=beta
            yHat=x*beta
            SSreg=np.sum((yHat-np.mean(y,0))**2)
            SStotal=np.sum((y-np.mean(y,0))**2)
            varExpArray[pix]=SSreg/SStotal

        betaMap=np.reshape(betaArray,(szY,szX))
        varExpMap=np.reshape(varExpArray,(szY,szX))

        outFile=fileOutDir+sessID+'_run'+str(run)+'_maps'
        np.savez(outFile,phaseMap=phaseMap,magMap=magMap,magRatioMap=magRatioMap,\
                 nonTargetMagMap=nonTargetMagMap,betaMap=betaMap,varExpMap=varExpMap)

        if saveImages:

            outFile = '%s_run%s_%s_Hz_magMap.png'%\
            (figOutDir+sessID,str(run),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(magMap)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' magMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = '%s_run%s_%s_Hz_magRatioMap.png'%\
            (figOutDir+sessID,str(run),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(magRatioMap)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' magRatioMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = '%s_run%s_%s_Hz_amplitudeMap.png'%\
            (figOutDir+sessID,str(run),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(betaMap)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' Amplitude', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = '%s_run%s_%s_Hz_varianceExplained.png'%\
            (figOutDir+sessID,str(run),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(varExpMap)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' Variance Explained', fontsize=14)
            plt.savefig(outFile)
            plt.close()

            outFile = '%s_run%s_%s_Hz_phaseMap.png'%\
            (figOutDir+sessID,str(run),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(phaseMapDisplay,'nipy_spectral',vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' phaseMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()


            #load surface for overlay
            #READ IN SURFACE
            imFile=anatSource+'frame0_registered.tiff'
            if not os.path.isfile(imFile):
                imFile=anatSource+'frame0.tiff'

            imSurf=cv2.imread(imFile,-1)
            szY,szX=imSurf.shape
            imSurf=np.true_divide(imSurf,2**12)*2**8

            if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])

                phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

            #plot
            outFile = '%s_run%s_%sHz_phaseMap_overlay.png'%\
                (figOutDir+sessID,str(int(run)),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(imSurf, 'gray')
            plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            #output masked image as well, if indicated
            if mask is not None:
                #load mask
                maskFile=targetRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
                f=np.load(maskFile)
                maskM=f['maskM']

                #apply mask
                phaseMapDisplay=phaseMapDisplay*maskM

                #plot
                outFile = '%s_run%s_%sHz_phaseMap_mask_%s.png'%\
                    (figOutDir+sessID,str(int(run)),str(np.around(freqs[freqIdx],4)),mask)
                fig=plt.figure()
                plt.imshow(imSurf, 'gray')
                plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
                plt.colorbar()
                fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
                plt.savefig(outFile)
                plt.close()

            #define legend matrix
            if stimType=='bar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(0, 2*np.pi, szScreenX)
                y = np.linspace(0, 2*np.pi, szScreenX)
                xv, yv = np.meshgrid(x, y)


                if cond==1:
                    legend=xv[296:1064,:]
                elif cond==2:
                    xv=(2*np.pi)-xv
                    legend=xv[296:1064,:]
                elif cond==3:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, legend = np.meshgrid(x, y)

                elif cond==4:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, yv = np.meshgrid(x, y)
                    legend=(2*np.pi)-yv

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()
            elif stimType=='polar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(-1, 1, szScreenX)
                y = np.linspace(-1, 1, szScreenX)
                xv, yv = np.meshgrid(x, y)

                rad,theta=cart2pol(xv,yv)

                x = np.linspace(-szScreenX/2, szScreenX/2, szScreenX)
                y = np.linspace(-szScreenY/2, szScreenY/2, szScreenY)
                xv, yv = np.meshgrid(x, y)

                radMask,thetaMask=cart2pol(xv,yv)


                thetaLegend=np.copy(theta)
                thetaLegend[theta<0]=-theta[theta<0]
                thetaLegend[theta>0]=(2*np.pi)-thetaLegend[theta>0]
                if cond == 1:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==2:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    thetaLegend=(2*np.pi)-thetaLegend
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==3:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)


                elif cond ==4:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)
                    legend=(2*np.pi)-legend
                    legend[radMask>szScreenY/2]=0

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()


        freqTextFile.close()


def smooth_phase_array(theta,sigma,sz):
    #build 2D Gaussian Kernel
    kernelX = cv2.getGaussianKernel(sz, sigma); 
    kernelY = cv2.getGaussianKernel(sz, sigma); 
    kernelXY = kernelX * kernelY.transpose(); 
    kernelXY_norm=np.true_divide(kernelXY,np.max(kernelXY.flatten()))
    
    #get x and y components of unit-length vector
    componentX=np.cos(theta)
    componentY=np.sin(theta)
    
    #convolce
    componentX_smooth=signal.convolve2d(componentX,kernelXY_norm,mode='same',boundary='symm')
    componentY_smooth=signal.convolve2d(componentY,kernelXY_norm,mode='same',boundary='symm')

    theta_smooth=np.arctan2(componentY_smooth,componentX_smooth)
    return theta_smooth

def smooth_phase_array(theta,sigma,sz):
    #build 2D Gaussian Kernel
    kernelX = cv2.getGaussianKernel(sz, sigma); 
    kernelY = cv2.getGaussianKernel(sz, sigma); 
    kernelXY = kernelX * kernelY.transpose(); 
    kernelXY_norm=np.true_divide(kernelXY,np.max(kernelXY.flatten()))
    
    #get x and y components of unit-length vector
    componentX=np.cos(theta)
    componentY=np.sin(theta)
    
    #convolce
    componentX_smooth=signal.convolve2d(componentX,kernelXY_norm,mode='same',boundary='symm')
    componentY_smooth=signal.convolve2d(componentY,kernelXY_norm,mode='same',boundary='symm')

    theta_smooth=np.arctan2(componentY_smooth,componentX_smooth)
    return theta_smooth

def smooth_array(inputArray,fwhm,phaseArray=False):
    szList=np.array([None,None,None,11,None,21,None,27,None,31,None,37,None,43,None,49,None,53,None,59,None,55,None,69,None,79,None,89,None,99])
    sigmaList=np.array([None,None,None,.9,None,1.7,None,2.6,None,3.4,None,4.3,None,5.1,None,6.4,None,6.8,None,7.6,None,8.5,None,9.4,None,10.3,None,11.2,None,12])
    sigma=sigmaList[fwhm]
    sz=szList[fwhm]
    if phaseArray:
        outputArray=smooth_phase_array(inputArray,sigma,sz)
    else:
        outputArray=cv2.GaussianBlur(inputArray, (sz,sz), sigma, sigma)
    return outputArray

def average_phase_per_condition(sourceRoot,targetRoot,sessID,runList,analysisDir,motionCorrection=False,saveImages=True,\
    stimType=None,mask=None):

    # DEFINE DIRECTORIES
    inDir = analysisDir+'SingleRunData/Files/'
    anatSource=targetRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'


    fileOutDir=analysisDir+'SingleConditionData/Files/'
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    if saveImages:
        figOutDirRoot=analysisDir+'SingleConditionData/Figures/'

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=fileOutDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()



    condList = get_condition_list(sourceRoot,sessID,runList)

    for cond in np.unique(condList):
        print('cond='+str(int(cond)))
        if saveImages:
            figOutDir=figOutDirRoot+'cond'+str(int(cond))+'/'
            if not os.path.exists(figOutDir):
                os.makedirs(figOutDir)
        condRunList=np.where(condList==cond)[0]
        for counter in range(len(condRunList)):
            print(counter)
            idx=condRunList[counter]
            run = runList[idx]

            #LOAD MAP FILE
            inFile=inDir+sessID+'_run'+str(run)+'_maps.npz'
            f=np.load(inFile)
            phaseMap=f['phaseMap']
            magMap=f['magMap']
            nonTargetMagMap=f['nonTargetMagMap']


            #AGGREGATE MAPS
            if counter == 0:
                phaseMapAll=phaseMap
                magMapAll=magMap
                nonTargetMagMapAll=nonTargetMagMap
            else:
                phaseMapAll=np.dstack((phaseMapAll,phaseMap))
                magMapAll=np.dstack((magMapAll,magMap))
                nonTargetMagMapAll=np.dstack((nonTargetMagMapAll,nonTargetMagMapAll))

        #AVERAGE ACROSS RUNS
        if len(condRunList)>1:
            magMap=np.mean(magMapAll,2)
            nonTargetMagMap=np.mean(nonTargetMagMapAll,2)
             #*for phase maps first convert to cartesian and add
            tmpX=np.sum(np.cos(phaseMapAll),2)
            tmpY=np.sum(np.sin(phaseMapAll),2)
            #*then get arctangent
            phaseMapMean=np.arctan2(tmpY,tmpX)
        else:
            magMap=magMapAll
            nonTargetMagMap=nonTargetMagMapAll
            phaseMapMean=phaseMapAll
        magRatioMap=np.true_divide(magMap,nonTargetMagMap)


        #set phase map range for visualization
        phaseMapDisplay=np.copy(phaseMapMean)
        phaseMapDisplay[phaseMapMean<0]=-phaseMapMean[phaseMapMean<0]
        phaseMapDisplay[phaseMapMean>0]=(2*np.pi)-phaseMapMean[phaseMapMean>0]


        #SAVE AVERAGE MAPS
        outFile=fileOutDir+sessID+'_cond'+str(int(cond))+'_avgMaps'

        np.savez(outFile,phaseMap=phaseMapMean,magMap=magMap,nonTargetMagMap=nonTargetMagMap,magRatioMap=magRatioMap)

        if saveImages:

            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_meanMagMap.png'
            fig=plt.figure()
            plt.imshow(magMap)
            plt.colorbar()
            fig.suptitle(sessID+' cond '+str(int(cond))+' Mean Mag', fontsize=14)
            plt.savefig(outFile)
            plt.close()


            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_meanMagRatioMap.png'
            fig=plt.figure()
            plt.imshow(magRatioMap)
            plt.colorbar()
            fig.suptitle(sessID+' cond '+str(int(cond))+' Mean Mag Ratio', fontsize=14)
            plt.savefig(outFile)
            plt.close()

            outFile = '%s_cond%s_meanPhaseMap.png'%(figOutDir+sessID,str(int(cond)))
            fig=plt.figure()
            plt.imshow(phaseMapDisplay,'nipy_spectral',vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond '+str(int(cond))+' Mean Phase', fontsize=14)
            plt.savefig(outFile)
            plt.close()

                        #load surface for overlay
            #READ IN SURFACE
            imFile=anatSource+'frame0_registered.tiff'
            if not os.path.isfile(inFile):
                imFile=anatSource+'frame0.tiff'

            imSurf=cv2.imread(imFile,-1)
            szY,szX=imSurf.shape
            imSurf=np.true_divide(imSurf,2**12)*2**8

            if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])
                print(((padDown,padUp),(padLeft,padRight)))
                phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

            #plot
            outFile = '%s_cond%s_phaseAverage_overlay.png'%\
                (figOutDir+sessID,str(int(cond)))


            fig=plt.figure()
            plt.imshow(imSurf, 'gray')
            plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            #output masked image as well, if indicated
            if mask is not None:
                #load mask
                maskFile=targetRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
                f=np.load(maskFile)
                maskM=f['maskM']

                #apply mask
                phaseMapDisplay=phaseMapDisplay*maskM

                #plot
                outFile = '%s_cond%s_phaseAverage_mask_%s.png'%\
                    (figOutDir+sessID,str(int(cond)),mask)

                fig=plt.figure()
                plt.imshow(imSurf, 'gray')
                plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
                plt.colorbar()
                fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
                plt.savefig(outFile)
                plt.close()

            #define legend matrix
            if stimType=='bar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(0, 2*np.pi, szScreenX)
                y = np.linspace(0, 2*np.pi, szScreenX)
                xv, yv = np.meshgrid(x, y)


                if cond==1:
                    legend=xv[296:1064,:]
                elif cond==2:
                    xv=(2*np.pi)-xv
                    legend=xv[296:1064,:]
                elif cond==3:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, legend = np.meshgrid(x, y)

                elif cond==4:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, yv = np.meshgrid(x, y)
                    legend=(2*np.pi)-yv

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()
            elif stimType=='polar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(-1, 1, szScreenX)
                y = np.linspace(-1, 1, szScreenX)
                xv, yv = np.meshgrid(x, y)

                rad,theta=cart2pol(xv,yv)

                x = np.linspace(-szScreenX/2, szScreenX/2, szScreenX)
                y = np.linspace(-szScreenY/2, szScreenY/2, szScreenY)
                xv, yv = np.meshgrid(x, y)

                radMask,thetaMask=cart2pol(xv,yv)


                thetaLegend=np.copy(theta)
                thetaLegend[theta<0]=-theta[theta<0]
                thetaLegend[theta>0]=(2*np.pi)-thetaLegend[theta>0]
                if cond == 1:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==2:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    thetaLegend=(2*np.pi)-thetaLegend
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==3:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)
                    
                    
                elif cond ==4:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)
                    legend=(2*np.pi)-legend
                    legend[radMask>szScreenY/2]=0

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()


            

def angle_array_to_complex_array(inputAngle):
    imagZ=np.empty(np.shape(inputAngle))
    imagZ[:]=np.NAN
    Z=np.zeros(np.shape(inputAngle),dtype=np.complex)

    Q1_ind = np.logical_and(np.cos(inputAngle)>=0,np.sin(inputAngle)>=0)
    Q2_ind = np.logical_and(np.cos(inputAngle)<0,np.sin(inputAngle)>=0)
    Q3_ind = np.logical_and(np.cos(inputAngle)>=0,np.sin(inputAngle)<0)
    Q4_ind = np.logical_and(np.cos(inputAngle)<0,np.sin(inputAngle)<0)


    imagZ[Q1_ind]=np.tan(inputAngle[Q1_ind])
    imagZ[Q2_ind]=np.tan(inputAngle[Q2_ind])
    imagZ[Q3_ind]=np.tan(inputAngle[Q3_ind])
    imagZ[Q4_ind]=np.tan(inputAngle[Q4_ind])

    indList=np.where(Q1_ind)
    if np.size(indList)>0:
        for ind in range(np.size(indList[0])):
            i=indList[0][ind]
            j=indList[1][ind]
            Z[i,j]=np.complex(1,imagZ[i,j])

    indList=np.where(Q2_ind)
    if np.size(indList)>0:
        for ind in range(np.size(indList[0])):
            i=indList[0][ind]
            j=indList[1][ind]
            Z[i,j]=np.complex(-1,-imagZ[i,j])
        
    indList=np.where(Q3_ind)
    if np.size(indList)>0:
        for ind in range(np.size(indList[0])):
            i=indList[0][ind]
            j=indList[1][ind]
            Z[i,j]=np.complex(1,imagZ[i,j])
        
    indList=np.where(Q4_ind)
    if np.size(indList)>0:
        for ind in range(np.size(indList[0])):
            i=indList[0][ind]
            j=indList[1][ind]
            Z[i,j]=np.complex(-1,-imagZ[i,j])
    
    return Z

def combine_phase_conditions(sourceRoot,analysisDir,sessID,motionCorrection=False,\
                             condPairs=None,condPairLabels=None,saveImages=True,\
                             stimType=None,mask=None):


    # DEFINE DIRECTORIES
    inDir = analysisDir+'SingleConditionData/Files/'
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'

    fileOutDir=analysisDir+'AbsolutePositionData/Files/'

    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

     #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=fileOutDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()

    if saveImages:
        figOutDirRoot=analysisDir+'AbsolutePositionData/Figures/'
        if not os.path.exists(figOutDirRoot):
            os.makedirs(figOutDirRoot)


    for (pairCounter,condPair) in enumerate(condPairs):
        if saveImages:
            figOutDir=figOutDirRoot+condPairLabels[pairCounter]+'/'
            if not os.path.exists(figOutDir):
                os.makedirs(figOutDir)

        for (counter,cond) in enumerate(condPair):

            print('cond='+str(int(cond)))

            #LOAD MAP FILE
            inFile=inDir+sessID+'_cond'+str(int(cond))+'_avgMaps.npz'
            f=np.load(inFile)
            phaseMap=f['phaseMap'] 
            magMap=f['magMap']
            nonTargetMagMap=f['nonTargetMagMap']


            #AGGREGATE MAPS
            if counter == 0:
                phaseMapAll=phaseMap
                magMapAll=magMap
                nonTargetMagMapAll=nonTargetMagMap
            else:
                phaseMapAll=np.dstack((phaseMapAll,phaseMap))
                magMapAll=np.dstack((magMapAll,magMap))
                nonTargetMagMapAll=np.dstack((nonTargetMagMapAll,nonTargetMagMapAll))



        #AVERAGE ACROSS CONDITIONS
        magMap=np.mean(magMapAll,2)
        nonTargetMagMap=np.mean(nonTargetMagMapAll,2)
        magRatioMap=np.true_divide(magMap,nonTargetMagMap)

        #GET HALF ANGLE MAPS
        halfAngleMap=np.true_divide(phaseMapAll[:,:,0],2)
        halfAngleMap2=np.true_divide(phaseMapAll[:,:,1],2)

        #get initial absolute phase map
        absPhaseMap=halfAngleMap-halfAngleMap2

        #correct values where numerical problems are expected to arise
        regionToFix1=np.logical_and(phaseMapAll[:,:,0]<=-np.true_divide(np.pi,2),phaseMapAll[:,:,1]<=-np.true_divide(np.pi,2))
        regionToFix2=np.logical_and(phaseMapAll[:,:,0]>=np.true_divide(np.pi,2),phaseMapAll[:,:,1]>=np.true_divide(np.pi,2))
        regionToFix=np.logical_or(regionToFix1,regionToFix2)

        tmp=np.copy(absPhaseMap)

        logicalInd=np.logical_and(regionToFix,absPhaseMap>0)
        tmp[logicalInd]=absPhaseMap[logicalInd]-(np.pi)

        logicalInd=np.logical_and(regionToFix,absPhaseMap<0)
        tmp[logicalInd]=(np.pi)+absPhaseMap[logicalInd]

        absPhaseMap=tmp

        phaseMapDisplay=np.copy(absPhaseMap)
        phaseMapDisplay[absPhaseMap<0]=-absPhaseMap[absPhaseMap<0]
        phaseMapDisplay[absPhaseMap>0]=(2*np.pi)-absPhaseMap[absPhaseMap>0]

        #get delay map
        delayMap=halfAngleMap+halfAngleMap2
        delayMap[regionToFix1]=np.pi+delayMap[regionToFix1]
        delayMap[regionToFix2]=delayMap[regionToFix2]-np.pi



        #SAVE AVERAGE MAPS

        outFile=fileOutDir+sessID+'_'+condPairLabels[pairCounter]+'_avgMaps'
        np.savez(outFile,absPhaseMap=absPhaseMap,delayMap=delayMap,magMap=magMap,\
            nonTargetMagMap=nonTargetMagMap,magRatioMap=magRatioMap)

        if saveImages:

            outFile = '%s_%s_meanMagMap.png'%\
            (figOutDir+sessID,condPairLabels[pairCounter])
            fig=plt.figure()
            plt.imshow(magMap)
            plt.colorbar()
            fig.suptitle(sessID+' '+condPairLabels[pairCounter]+' Mean Mag', fontsize=14)
            plt.savefig(outFile)
            plt.close()

            outFile = '%s_%s_meanMagRatio.png'%\
            (figOutDir+sessID,condPairLabels[pairCounter])
            fig=plt.figure()
            plt.imshow(magRatioMap)
            plt.colorbar()
            fig.suptitle(sessID+' '+condPairLabels[pairCounter]+' Mean Mag Ratio', fontsize=14)
            plt.savefig(outFile)
            plt.close()


            outFile = '%s_%s_delayMap.png'%\
            (figOutDir+sessID,condPairLabels[pairCounter])
            fig=plt.figure()
            plt.imshow(delayMap,'nipy_spectral',vmin=-np.pi,vmax=np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' '+condPairLabels[pairCounter]+'Delay', fontsize=14)
            plt.savefig(outFile)
            plt.close()


            outFile = '%s_%s_absPhaseMap.png'%\
            (figOutDir+sessID,condPairLabels[pairCounter])

            fig=plt.figure()
            plt.imshow(phaseMapDisplay,'nipy_spectral',vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' '+condPairLabels[pairCounter]+' Absolute Phase', fontsize=14)
            plt.savefig(outFile)
            plt.close()

          #  load surface for overlay
            #READ IN SURFACE
            imFile=anatSource+'frame0_registered.tiff'
            if not os.path.isfile(inFile):
                imFile=anatSource+'frame0.tiff'

            imSurf=cv2.imread(imFile,-1)
            szY,szX=imSurf.shape
            imSurf=np.true_divide(imSurf,2**12)*2**8

            if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])

                phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))
                delayMap=np.pad(delayMap,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((np.nan, np.nan),(np.nan, np.nan)))

            #plot
            outFile = '%s_%s_absPhaseMap_overlay.png'%\
                (figOutDir+sessID,condPairLabels[pairCounter])

            fig=plt.figure()
            plt.imshow(imSurf, 'gray')
            plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()


            #output masked image as well, if indicated
            if mask is not None:
                #load mask
                maskFile=sourceRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
                f=np.load(maskFile)
                maskM=f['maskM']

                #apply mask
                phaseMapDisplay=phaseMapDisplay*maskM
                delayMap[maskM==0]=np.nan

                #plot
                outFile = '%s_%s_absPhaseMap_mask_%s.png'%\
                    (figOutDir+sessID,condPairLabels[pairCounter],mask)
                fig=plt.figure()
                plt.imshow(imSurf, 'gray')
                plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
                plt.colorbar()
                fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
                plt.savefig(outFile)
                plt.close()

                #plot
                outFile = '%s_%s_delayMap_mask_%s.png'%\
                    (figOutDir+sessID,condPairLabels[pairCounter],mask)


                fig=plt.figure()
                plt.imshow(imSurf, 'gray')
                plt.imshow(delayMap,'nipy_spectral',alpha=.5,vmin=-np.pi,vmax=np.pi)
                plt.colorbar()
                fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
                plt.savefig(outFile)
                plt.close()

            #define legend matrix
            if stimType=='bar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(0, 2*np.pi, szScreenX)
                y = np.linspace(0, 2*np.pi, szScreenX)
                xv, yv = np.meshgrid(x, y)


                if condPairLabels[pairCounter]=='Azimuth':
                    legend=xv[296:1064,:]

                elif condPairLabels[pairCounter]=='Elevation':
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, legend = np.meshgrid(x, y)

                elif cond==4:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, yv = np.meshgrid(x, y)
                    legend=(2*np.pi)-yv

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()
            elif stimType=='polar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(-1, 1, szScreenX)
                y = np.linspace(-1, 1, szScreenX)
                xv, yv = np.meshgrid(x, y)

                rad,theta=cart2pol(xv,yv)

                x = np.linspace(-szScreenX/2, szScreenX/2, szScreenX)
                y = np.linspace(-szScreenY/2, szScreenY/2, szScreenY)
                xv, yv = np.meshgrid(x, y)

                radMask,thetaMask=cart2pol(xv,yv)


                thetaLegend=np.copy(theta)
                thetaLegend[theta<0]=-theta[theta<0]
                thetaLegend[theta>0]=(2*np.pi)-thetaLegend[theta>0]

                if condPairLabels[pairCounter]=='Theta':
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0

                elif condPairLabels[pairCounter]=='Radius':
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)


                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()


def get_analysis_path_timecourse(analysisRoot,  interpolate=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False,averageFrames=None):
    #Cesar Echavarria 11/2016

    imgOperationDir=''
    #DEFINE DIRECTORIES
    if motionCorrection:
        imgOperationDir=imgOperationDir+'motionCorrection'
    else:
        imgOperationDir=imgOperationDir+'noMotionCorrection'


    procedureDir=''
    if interpolate:
        procedureDir=procedureDir+'interpolate'
        
    if excludeEdges:
        procedureDir=procedureDir+'excludeEdges'

    if averageFrames is not None:
        procedureDir=procedureDir+'_averageFrames_'+str(averageFrames)


    if removeRollingMean:
        procedureDir=procedureDir+'_minusRollingMean'

    timecourseDir=analysisRoot+'/timecourse/'+imgOperationDir+'/'+procedureDir+'/'

    return timecourseDir

def get_interp_extremes(sourceRoot,sessID,runList,stimFreq):

    run=runList[0]
    runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')
    frameFolder=runFolder[0]+"/frames/"
    planFolder=runFolder[0]+"/plan/"
    frameTimes,frameCond,frameCount = get_frame_times(planFolder)

    tMin=np.true_divide(1,stimFreq)
    nCycles=np.round(frameTimes[-1]/np.true_divide(1,stimFreq))
    tMax=np.true_divide(1,stimFreq)*(nCycles-1)


    return tMin,tMax

def interpolate_array(t0,array0,tNew):
    interpF = interpolate.interp1d(t0, array0,1)
    
    arrayNew=np.zeros((array0.shape[0],tNew.size))
    if np.any(tNew<t0[0]):
        arrayNew[:,tNew<t0[0]]=np.expand_dims(array0[:,0],1)
        ind0=np.where(tNew<t0[0])[0][-1]+1
    else:
        ind0=0
    if np.any(tNew>t0[-1]):
        arrayNew[:,tNew>t0[-1]]=np.expand_dims(array0[:,-1],1)
        ind1=np.where(tNew>t0[-1])[0][0]
    else:
        ind1=tNew.size

    arrayNew[:,ind0:ind1]=interpF(tNew[ind0:ind1])
    
    return arrayNew

def analyze_complete_timecourse(sourceRoot, targetRoot, sessID, runList, stimFreq, frameRate, \
                               interp=False, excludeEdges=False, removeRollingMean=False, \
                               motionCorrection=False,averageFrames=None, groupPeriods = None, loadCorrectedFrames=True,\
                               SDmaps=False,makeSingleRunMovies=False,makeSingleCondMovies=True):


    # DEFINE DIRECTORIES

    motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'

    analysisRoot=targetRoot+'/Sessions/'+sessID+'/Analyses/'
    analysisDir=get_analysis_path_timecourse(analysisRoot, interp, excludeEdges, removeRollingMean, \
        motionCorrection,averageFrames)

    fileOutDir=analysisDir+'SingleConditionData/Files/'
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=fileOutDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()


    if makeSingleCondMovies:
        avgMovieOutDir=analysisDir+'SingleConditionData/Movies/'
        if not os.path.exists(avgMovieOutDir):
            os.makedirs(avgMovieOutDir) 
    if makeSingleRunMovies:
        singleRunMovieOutDir=analysisDir+'SingleRunData/Movies/'
        if not os.path.exists(singleRunMovieOutDir):
            os.makedirs(singleRunMovieOutDir) 
    if SDmaps:
        figOutDir=analysisDir+'SingleConditionData/Figures/'
        if not os.path.exists(figOutDir):
            os.makedirs(figOutDir) 

    condList = get_condition_list(sourceRoot,sessID,runList)
    if interp:
        #GET INTERPOLATION TIME
        newStartT,newEndT=get_interp_extremes(sourceRoot,sessID,runList,stimFreq)
        newTimes=np.arange(newStartT+(1.0/frameRate),newEndT,1.0/frameRate)#always use the same time points


    for cond in np.unique(condList):
        print('cond='+str(cond))
        condRunList=np.where(condList==cond)[0]
        for counter in range(len(condRunList)):
            print('counter='+str(counter))
            idx=condRunList[counter]
            run = runList[idx]
            print('run='+str(run))
            #DEFINE DIRECTORIES
            runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')
            frameFolder=runFolder[0]+"/frames/"
            planFolder=runFolder[0]+"/plan/"
            frameTimes,frameCond,frameCount = get_frame_times(planFolder)

            #READ IN FRAMES
            print('Loading frames...')
            if motionCorrection:

                if loadCorrectedFrames:
                    #LOAD MOTION CORRECTED FRAMES
                    inFile=motionFileDir+sessID+'_run'+str(run)+'_correctedFrames.npz'
                    f=np.load(inFile)
                    frameArray=f['correctedFrameArray']
                else:
                    #GET REFFERNCE FRAME
                    imRef=get_reference_frame(sourceRoot,sessID,runList[0])
                    szY,szX=imRef.shape

                    # READ IN FRAMES
                    frameArray=np.zeros((szY,szX,frameCount))
                    for f in range (0,frameCount):
                        imFile=frameFolder+'frame'+str(f)+'.tiff'
                        im0=misc.imread(imFile)
                        frameArray[:,:,f]=np.copy(im0)

                    #-> load warp matrices
                    inFile=motionFileDir+sessID+'_run'+str(run)+'_motionRegistration.npz'
                    f=np.load(inFile)
                    warpMatrices=f['warpMatrices']

                    #APPLY MOTION CORRECTION
                    frameArray=apply_motion_correction(frameArray,warpMatrices)

                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']

                #APPLY BOUNDARIES
                frameArray = apply_motion_correction_boundaries(frameArray,boundaries)

                szY,szX = np.shape(frameArray[:,:,0])


            else:
                #GET REFFERNCE FRAME
                imRef=get_reference_frame(sourceRoot,sessID,runList[0])
                szY,szX=imRef.shape

                # READ IN FRAMES
                frameArray=np.zeros((szY,szX,frameCount))
                for f in range (0,frameCount):
                    imFile=frameFolder+'frame'+str(f)+'.tiff'
                    im0=misc.imread(imFile)
                    frameArray[:,:,f]=np.copy(im0)

            frameArray=np.reshape(frameArray,(szY*szX,frameCount))
            
            #INTERPOLATE FOR CONSTANT FRAME RATE
            if interp:
                print('Interpolating...')
                frameArray=interpolate_array(frameTimes,frameArray,newTimes)
                frameTimes=newTimes
            if counter == 0:   
                meanPixelValue=np.mean(frameArray,1)
            else:
                meanPixelValue=np.vstack((meanPixelValue,np.mean(frameArray,1)))
            # REMOVE ROLLING AVERAGE
            if removeRollingMean:
                print('Removing Rolling Mean...')
                #INITIALIZE VARIABLES
                detrendedFrameArray=np.zeros(np.shape(frameArray))
                rollingWindowSz=int(np.ceil((np.true_divide(1,stimFreq)*2)*frameRate))

                #REMOVE ROLLING MEAN; KEEP MEAN OFFSET
                for pix in range(0,np.shape(frameArray)[0]):
                    tmp0=frameArray[pix,:]
                    tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                    rollingAvg=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                    rollingAvg=rollingAvg[rollingWindowSz:-rollingWindowSz]


                    detrendedFrameArray[pix,:]=np.subtract(tmp0,rollingAvg)
                frameArray=detrendedFrameArray
                del detrendedFrameArray

            #AVERAGE FRAMES
            if averageFrames is not None:
                print('Pooling frames values....')
                smoothFrameArray=np.zeros(np.shape(frameArray))
                rollingWindowSz=averageFrames

                for pix in range(0,np.shape(frameArray)[0]):

                    tmp0=frameArray[pix,:];
                    tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                    tmp2=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                    tmp2=tmp2[rollingWindowSz:-rollingWindowSz]

                    smoothFrameArray[pix,:]=tmp2
                frameArray=smoothFrameArray
                del smoothFrameArray
            
            #EXCLUDE FIRST AND LAST PERIOD
            if excludeEdges:
                print('Excluding first and last periods...')
                frameArray,frameTimes=exclude_edge_periods(frameArray,frameTimes,stimFreq)

            if makeSingleRunMovies:
                tmp=frameArray
                #RESHAPE
                nPix,nPts=np.shape(tmp)
                tmp=np.reshape(tmp,(szY,szX,nPts))

                #NORMALIZE ARRAY
                tmp=normalize_stack(tmp)

                #MAKE MOVIE
                outFile=singleRunMovieOutDir+'cond'+str(cond)+'_run'+str(counter+1)+'_movie.mp4'
                make_movie_from_stack(singleRunMovieOutDir,tmp,frameRate,outFile)
            

            if counter == 0:
                frameArrayAll=frameArray
            else:
                frameArrayAll=np.dstack((frameArrayAll,frameArray))

            if groupPeriods is not None:
                if counter ==0:
                    stimPeriod=1.0/stimFreq
                    framePeriod=1.0/frameRate
                    nCycles=np.round(frameTimes[-1]/stimPeriod)
                    if excludeEdges:
                        nCycles=nCycles-1
                    nGroups=int(np.floor(nCycles/groupPeriods))
                    print(nCycles,nGroups,groupPeriods)

                for g in range(nGroups):
                    startTime=stimPeriod*((g*groupPeriods)+1)
                    startInd=np.where(frameTimes>startTime)[0][0]
                    endInd=startInd+np.floor((groupPeriods*stimPeriod)/framePeriod)-1

                    print(startInd,endInd)
                    frameArrayGroup=frameArray[:,startInd:endInd]
                    if counter == 0 and g == 0:
                        frameArrayGrouped=frameArrayGroup
                    else:
                        frameArrayGrouped=np.dstack((frameArrayGrouped,frameArrayGroup))
                    del frameArrayGroup
                    
            del frameArray


        #AVERAGE PER CONDITION
        meanPixelValue=np.squeeze(np.mean(meanPixelValue,0))
        if len(frameArrayAll.shape)==3:
            frameArrayAvg=np.mean(frameArrayAll,2)
        else:
            frameArrayAvg=frameArrayAll
        del frameArrayAll
        if groupPeriods is None:
            frameArrayGroupedAvg=[]
        else:
            frameArrayGroupedAvg=np.mean(frameArrayGrouped,2)
        
        outFile=fileOutDir+'cond'+str(cond)+'_averageTimeCourse'
        np.savez(outFile,frameArrayAvg=frameArrayAvg,frameTimes=frameTimes,\
                 szY=szY,szX=szX,frameCount=frameCount,\
                 meanPixelValue=meanPixelValue,\
                groupPeriods=groupPeriods,frameArrayGroupedAvg=frameArrayGroupedAvg)


        if SDmaps:
            #GET STANDARD DEV
            pixSD=np.std(frameArrayAvg,1)
            mapSD=np.reshape(pixSD,(szY,szX))
            #MAKE AND SAVE MAP
            fig=plt.figure()
            plt.imshow(mapSD)
            plt.colorbar()
            plt.savefig(figOutDir+'cond'+str(cond)+'_SDmap.png')

        if makeSingleCondMovies:
            if groupPeriods is None:
                #RESHAPE
                nPix,nPts=np.shape(frameArrayAvg)
                frameArrayAvg=np.reshape(frameArrayAvg,(szY,szX,nPts))

                #NORMALIZE ARRAY
                frameArrayAvg=normalize_stack(frameArrayAvg)

                #MAKE MOVIE
                outFile=avgMovieOutDir+'cond'+str(cond)+'_average_movie.mp4'
                make_movie_from_stack_periodic(avgMovieOutDir,frameArrayAvg,stimFreq,frameRate=frameRate,movFile=outFile)
            else:
                 #RESHAPE
                nPix,nPts=np.shape(frameArrayGroupedAvg)
                frameArrayGroupedAvg=np.reshape(frameArrayGroupedAvg,(szY,szX,nPts))

                #NORMALIZE ARRAY
                frameArrayGroupedAvg=normalize_stack(frameArrayGroupedAvg)

                #MAKE MOVIE
                outFile='%scond%s_average_%sPeriods_movie.mp4'%(avgMovieOutDir,str(cond),str(groupPeriods))
                make_movie_from_stack(avgMovieOutDir,frameArrayGroupedAvg,frameRate=frameRate,movFile=outFile)
                
                del frameArrayGroupedAvg
                

        del frameArrayAvg
    
def analyze_periodic_data_from_timecourse(sourceRoot, targetRoot, sessID, runList, stimFreq, frameRate, \
    interp=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False, averageFrames=None,saveImages=True,mask=None,\
    stimType=None):

    
    analysisRoot=targetRoot+'/Sessions/'+sessID+'/Analyses/';
    analysisDir=get_analysis_path_timecourse(analysisRoot, interp, excludeEdges, removeRollingMean, \
    motionCorrection,averageFrames)

    # DEFINE DIRECTORIES
    fileInDir=analysisDir+'SingleConditionData/Files/'
    anatSource=targetRoot+'Sessions/'+sessID+'/Surface/'

    motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/'
    motionFileDir=motionDir+'Registration/'

    analysisDir=analysisDir+'phaseDecoding/'
    fileOutDir=analysisDir+'SingleConditionData/Files/'
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    if saveImages:
        figOutDirRoot=analysisDir+'SingleConditionData/Figures/'

      #  OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=fileOutDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()
            
    #GET CONDITIONS AVAILABLE
    condList = get_condition_list(sourceRoot,sessID,runList)

    for (condCount,cond) in enumerate(np.unique(condList)):
        print('cond='+str(cond))

        #DEFINE OUTPUT DIRECTORY
        if saveImages:
            figOutDir=figOutDirRoot+'/cond'+str(int(cond))+'/'
            if not os.path.exists(figOutDir):
                os.makedirs(figOutDir)


        #READ IN DATA
        inFile=fileInDir+'cond'+str(cond)+'_averageTimeCourse.npz'
        f=np.load(inFile)
        frameArray=f['frameArrayAvg']
        frameTimes=f['frameTimes']
        szY=f['szY']
        szX=f['szX']
        meanPixelValue=f['meanPixelValue']
        groupPeriods=f['groupPeriods']
        if groupPeriods is not None:
            frameArrayGrouped=f['frameArrayGroupedAvg']


        #Get FFT
        print('Analyzing phase and magnitude....')
        fourierData = np.fft.fft(frameArray)
        #Get magnitude and phase data
        magData=abs(fourierData)
        phaseData=np.angle(fourierData)

        signalLength=np.shape(frameArray)[1]
        freqs = np.fft.fftfreq(signalLength, float(1/frameRate))
        idx = np.argsort(freqs)

        freqs=freqs[idx]
        magData=magData[:,idx]
        phaseData=phaseData[:,idx]

        freqs=freqs[np.round(signalLength/2)+1:]#excluding DC offset
        magData=magData[:,np.round(signalLength/2)+1:]#excluding DC offset
        phaseData=phaseData[:,np.round(signalLength/2)+1:]#excluding DC offset


        freqIdx=np.where(freqs>stimFreq)[0][0]
        topFreqIdx=np.where(freqs>1)[0][0]

        #OUTPUT TEXT FILE FREQUENCY CHANNEL ANALYZED
        if condCount == 0:

            outFile=fileOutDir+'frequency_analyzed.txt'
            freqTextFile = open(outFile, 'w+')
        freqTextFile.write('COND '+str(cond)+' '+str(np.around(freqs[freqIdx],4))+' Hz\n')

        #GET PERCENT SIGNAL MODULATION
        meanPixelValue=np.expand_dims(meanPixelValue,1)
        meanPixelValue=np.tile(meanPixelValue,(1,frameArray.shape[1]))
        frameArrayPSC=np.true_divide(frameArray,meanPixelValue)*100
        if groupPeriods is not None:
            meanPixelValue=meanPixelValue[:,0:frameArrayGrouped.shape[1]]
            frameArrayGroupedPSC=np.true_divide(frameArrayGrouped,meanPixelValue)*100


        if saveImages:
            
            maxModIdx=np.argmax(magData[:,freqIdx],0)
            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_magnitudePlot.png'
            fig=plt.figure()
            plt.plot(freqs,magData[maxModIdx,:])
            fig.suptitle(sessID+' cond '+str(int(cond))+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(outFile)
            plt.close()
            
            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_magnitudePlot_zoom.png'
            fig=plt.figure()
            plt.plot(freqs[0:topFreqIdx],magData[maxModIdx,0:topFreqIdx])
            fig.suptitle(sessID+' cond '+str(int(cond))+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(outFile)
            plt.close()

            stimPeriod_t=np.true_divide(1,stimFreq)
            stimPeriod_frames=stimPeriod_t*frameRate
            periodStartFrames=np.round(np.arange(0,len(frameTimes),stimPeriod_frames))

            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_timecourse.png'
            fig=plt.figure()
            plt.plot(frameTimes,frameArrayPSC[maxModIdx,])
            fig.suptitle(sessID+' cond '+str(int(cond))+' phase'+str(np.round(np.rad2deg(phaseData[maxModIdx,freqIdx]))), fontsize=10)
            plt.xlabel('Time (s)',fontsize=16)
            plt.ylabel('Signal Change (%)',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            if interp:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            else:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            axes.set_xlim([frameTimes[0],frameTimes[-1]])
            plt.savefig(outFile)
            plt.close()

            if groupPeriods is not None:
                periodStartFramesGrouped=periodStartFrames[0:groupPeriods]
                outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_timecourse_grouped.png'
                fig=plt.figure()
                plt.plot(frameTimes[0:frameArrayGroupedPSC.shape[1]],frameArrayGroupedPSC[maxModIdx,])
                fig.suptitle(sessID+' cond '+str(int(cond))+' phase'+str(np.round(np.rad2deg(phaseData[maxModIdx,freqIdx]))), fontsize=10)
                plt.xlabel('Time (s)',fontsize=16)
                plt.ylabel('Signal Change (%)',fontsize=16)
                axes = plt.gca()
                ymin, ymax = axes.get_ylim()
                if interp:
                    for f in periodStartFramesGrouped:
                        plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
                else:
                    for f in periodStartFramesGrouped:
                        plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
                axes.set_xlim([frameTimes[0],frameTimes[frameArrayGroupedPSC.shape[1]]])
                plt.savefig(outFile)
                plt.close()



        magArray=magData[:,freqIdx]
        magMap=np.reshape(magArray,(szY,szX))
        phaseArray=phaseData[:,freqIdx]
        phaseMap=np.reshape(phaseArray,(szY,szX))

        #set phase map range for visualization
        phaseMapDisplay=np.copy(phaseMap)
        phaseMapDisplay[phaseMap<0]=-phaseMap[phaseMap<0]
        phaseMapDisplay[phaseMap>0]=(2*np.pi)-phaseMap[phaseMap>0]


        #get various measures of data quality
        #1) magnitude and ratio of magnitude
        tmp=np.copy(magData)
        np.delete(tmp,freqIdx,1)
        nonTargetMagArray=np.sum(tmp,1)
        magRatio=magArray/nonTargetMagArray
        magRatioMap=np.reshape(magRatio,(szY,szX))
        nonTargetMagMap=np.reshape(nonTargetMagArray,(szY,szX))

        #2) amplitude and variance expained
        t=frameTimes*(2*np.pi)*stimFreq
        t=np.transpose(np.expand_dims(t,1))
        tMatrix=np.tile(t,(phaseData.shape[0],1))

        phi=np.expand_dims(phaseArray,1)
        phiMatrix=np.tile(phi,(1,frameArray.shape[1]))
        Xmatrix=np.cos(tMatrix+phiMatrix)

        betaArray=np.zeros((frameArray.shape[0]))
        varExpArray=np.zeros((frameArray.shape[0]))

        for pix in range(frameArray.shape[0]):
            x=np.expand_dims(Xmatrix[pix,:],1)
            y=frameArrayPSC[pix,:]
            beta=np.matmul(np.linalg.pinv(x),y)
            betaArray[pix]=beta
            yHat=x*beta
            if pix == maxModIdx:
                signalFit=yHat
            SSreg=np.sum((yHat-np.mean(y,0))**2)
            SStotal=np.sum((y-np.mean(y,0))**2)
            varExpArray[pix]=SSreg/SStotal

        betaMap=np.reshape(betaArray,(szY,szX))
        varExpMap=np.reshape(varExpArray,(szY,szX))

        outFile=fileOutDir+sessID+'_cond'+str(int(cond))+'_maps'
        np.savez(outFile,phaseMap=phaseMap,magMap=magMap,magRatioMap=magRatioMap,\
             nonTargetMagMap=nonTargetMagMap,betaMap=betaMap,varExpMap=varExpMap)



        if saveImages:

            outFile = figOutDir+sessID+'_cond'+str(int(cond))+'_timecourse_fit.png'
            fig=plt.figure()
            plt.plot(frameTimes,frameArrayPSC[maxModIdx,],'b')
            plt.plot(frameTimes,signalFit,'r')
            fig.suptitle(sessID+' cond '+str(int(cond))+' phase'+str(np.round(np.rad2deg(phaseData[maxModIdx,freqIdx]))), fontsize=10)
            plt.xlabel('Time (s)',fontsize=16)
            plt.ylabel('Signal Change (%)',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            if interp:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            else:
                for f in periodStartFrames:
                    plt.axvline(x=frameTimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
            axes.set_xlim([frameTimes[0],frameTimes[-1]])
            plt.savefig(outFile)
            plt.close()


            outFile = outFile = '%s_run%s_%s_Hz_magMap.png'%\
            (figOutDir+sessID,str(cond),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(magMap)
            plt.colorbar()#
            fig.suptitle(sessID+' cond'+str(cond)+' magMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = outFile = '%s_run%s_%s_Hz_magRatioMap.png'%\
            (figOutDir+sessID,str(cond),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(magRatioMap)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' magRatioMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = outFile = '%s_run%s_%s_Hz_amplitude.png'%\
            (figOutDir+sessID,str(cond),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(betaMap)
            plt.colorbar()#
            fig.suptitle(sessID+' cond'+str(cond)+' Amiplitude', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            outFile = outFile = '%s_run%s_%s_Hz_varianceExplainedMap.png'%\
            (figOutDir+sessID,str(cond),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(varExpMap)
            plt.colorbar()#
            fig.suptitle(sessID+' cond'+str(cond)+' Variance Explained', fontsize=14)
            plt.savefig(outFile)
            plt.close()


            outFile = figOutDir+sessID+'_cond'+str(cond)+'_'+str(np.around(freqs[freqIdx],4))+'Hz_phaseMap.png'
            fig=plt.figure()
            plt.imshow(phaseMapDisplay,'nipy_spectral',vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            #load surface for overlay
            #READ IN SURFACE
            imFile=anatSource+'frame0_registered.tiff'
            if not os.path.isfile(inFile):
                imFile=anatSource+'frame0.tiff'

            imSurf=cv2.imread(imFile,-1)
            szY,szX=imSurf.shape
            imSurf=np.true_divide(imSurf,2**12)*2**8


            if motionCorrection:
                #LOAD MOTION CORRECTED BOUNDARIES
                inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
                f=np.load(inFile)
                boundaries=f['boundaries']
                padDown=int(boundaries[0])
                padUp=int(szY-boundaries[1])
                padLeft=int(boundaries[2])
                padRight=int(szX-boundaries[3])

                phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((0, 0),(0,0)))

            #plot
            outFile = '%s_cond%s_%sHz_phaseMap_overlay.png'%\
                (figOutDir+sessID,str(int(cond)),str(np.around(freqs[freqIdx],4)))
            fig=plt.figure()
            plt.imshow(imSurf, 'gray')
            plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(outFile)
            plt.close()

            #output masked image as well, if indicated
            if mask is not None:
                #load mask
                maskFile=targetRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
                f=np.load(maskFile)
                maskM=f['maskM']

                #apply mask
                phaseMapDisplay=phaseMapDisplay*maskM

                #plot
                outFile = '%s_cond%s_%sHz_phaseMap_mask_%s.png'%\
                    (figOutDir+sessID,str(int(cond)),str(np.around(freqs[freqIdx],4)),mask)
                fig=plt.figure()
                plt.imshow(imSurf, 'gray')
                plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
                plt.colorbar()
                fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
                plt.savefig(outFile)
                plt.close()

            #define legend matrix
            if stimType=='bar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(0, 2*np.pi, szScreenX)
                y = np.linspace(0, 2*np.pi, szScreenX)
                xv, yv = np.meshgrid(x, y)


                if cond==1:
                    legend=xv[296:1064,:]
                elif cond==2:
                    xv=(2*np.pi)-xv
                    legend=xv[296:1064,:]
                elif cond==3:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, legend = np.meshgrid(x, y)

                elif cond==4:
                    y = np.linspace(0, 2*np.pi, szScreenY)
                    xv, yv = np.meshgrid(x, y)
                    legend=(2*np.pi)-yv

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()
            elif stimType=='polar':
                szScreenY=768
                szScreenX=1360

                x = np.linspace(-1, 1, szScreenX)
                y = np.linspace(-1, 1, szScreenX)
                xv, yv = np.meshgrid(x, y)

                rad,theta=cart2pol(xv,yv)

                x = np.linspace(-szScreenX/2, szScreenX/2, szScreenX)
                y = np.linspace(-szScreenY/2, szScreenY/2, szScreenY)
                xv, yv = np.meshgrid(x, y)

                radMask,thetaMask=cart2pol(xv,yv)


                thetaLegend=np.copy(theta)
                thetaLegend[theta<0]=-theta[theta<0]
                thetaLegend[theta>0]=(2*np.pi)-thetaLegend[theta>0]
                if cond == 1:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==2:
                    thetaLegend=(2*np.pi)-thetaLegend
                    thetaLegend=thetaLegend-np.true_divide(np.pi,2)
                    thetaLegend=(thetaLegend + np.pi) % (2*np.pi)
                    thetaLegend=(2*np.pi)-thetaLegend
                    legend=thetaLegend[296:1064,:]
                    legend[radMask>szScreenY/2]=0
                elif cond ==3:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)
                    
                    
                elif cond ==4:
                    rad=rad[296:1064,:]
                    rad[radMask>szScreenY/2]=0
                    legend=np.true_divide(rad,np.max(rad))*(2*np.pi)
                    legend=(2*np.pi)-legend
                    legend[radMask>szScreenY/2]=0

                outFile = figOutDir+sessID+'_cond'+str(cond)+'_legend.png'
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(outFile)
                plt.close()

    freqTextFile.close()

