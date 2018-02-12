import sys
sys.path.append('/nas/volume1/shared/cesar/widefield/dataAnalysis/Python_scripts/WiPy_package')
import numpy as np
import glob
import optparse
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors
from WiPy import *

def get_surface(sourceRoot,targetRoot,sessID):

        
    #DEFINE DIRECTORY
    sourceDir=glob.glob(sourceRoot+'/'+sessID+'_surface*')
    surfDir=sourceDir[0]+'/Surface/'
    outDir=targetRoot+'/Surface/';
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

def analyze_periodic_data_per_run(sourceRoot, targetRoot, sessID, runList, stimFreq, frameRate, \
    interp=False, excludeEdges=False, removeRollingMean=False, \
    motionCorrection=False,averageFrames=None,loadCorrectedFrames=True,saveImages=True,makeMovies=True,\
    stimType=None,mask=None):
     # DEFINE DIRECTORIES
    anatSource=os.path.join(targetRoot,'Surface')
    motionDir=os.path.join(targetRoot,'Motion')
    motionFileDir=os.path.join(motionDir, 'Registration')


    analysisRoot = os.path.join(targetRoot,'Analyses')
    analysisDir=get_analysis_path_phase(analysisRoot, stimFreq, interp, excludeEdges,removeRollingMean, \
        motionCorrection,averageFrames)
    fileOutDir=os.path.join(analysisDir,'SingleRunData','Files')
    if not os.path.exists(fileOutDir):
        os.makedirs(fileOutDir)

    if saveImages:
        figOutDirRoot=os.path.join(analysisDir,'SingleRunData','Figures')


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
        print('Current Run: %s'%(run))
        #DEFINE DIRECTORIES

        runFolder=glob.glob('%s/%s_%s_*'%(sourceRoot,sessID,run))
        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"


        # READ IN FRAME TIMES FILE
        frameTimes,frameCond,frameCount=get_frame_times(planFolder)
        cond=frameCond[0]

        if saveImages:
            figOutDir=os.path.join(figOutDirRoot,'cond%s'%(cond))
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

            outFile=os.path.join(fileOutDir,'frequency_analyzed.txt')
            freqTextFile = open(outFile, 'w+')
            freqTextFile.write('RUN '+str(run)+' '+str(np.around(freqs[freqIdx],4))+' Hz\n')

        if saveImages:

            maxModIdx=np.argmax(magData[:,freqIdx],0)
            figName = 'magnitude_%s_%s.png'%(sessID,run)
            fig=plt.figure()
            plt.plot(freqs,magData[maxModIdx,:])
            fig.suptitle(sessID+' run '+str(run)+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()

            figName = 'magnitude_%s_%s.png'%(sessID,run)
            fig=plt.figure()
            plt.plot(freqs[0:topFreqIdx],magData[maxModIdx,0:topFreqIdx])
            fig.suptitle(sessID+' run '+str(run)+' magnitude', fontsize=20)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Magnitude',fontsize=16)
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            plt.axvline(x=freqs[freqIdx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()

            stimPeriod_t=np.true_divide(1,stimFreq)
            stimPeriod_frames=stimPeriod_t*frameRate
            periodStartFrames=np.round(np.arange(0,len(frameTimes),stimPeriod_frames))

            figName = 'timecourse_%s_%s.png'%(sessID,run)
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
            plt.savefig(os.path.join(figOutDir,figName))
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

        fileName='%s_%s_map'%(sessID,run)
        np.savez(os.path.join(fileOutDir,fileName),phaseMap=phaseMap,magMap=magMap,magRatioMap=magRatioMap,\
                 nonTargetMagMap=nonTargetMagMap,betaMap=betaMap,varExpMap=varExpMap)

        if saveImages:
            figName = 'mag_map_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)

            fig=plt.figure()
            plt.imshow(magMap)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' magMap', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()

            figName = 'mag_ratio_map_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)
            fig=plt.figure()
            plt.imshow(magRatioMap)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' magRatioMap', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()


            figName = 'amplitude_map_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)
            fig=plt.figure()
            plt.imshow(betaMap)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' Amplitude', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()


            figName = 'variance_explained_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)
            fig=plt.figure()
            plt.imshow(varExpMap)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' Variance Explained', fontsize=14)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()


            figName = 'phase_map_%s_Hz_%s_%s.png'%(str(np.around(freqs[freqIdx],4)),sessID,run)
            fig=plt.figure()
            plt.imshow(phaseMapDisplay,'nipy_spectral',vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' run'+str(run)+' phaseMap', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
            plt.close()


            #load surface for overlay
            #READ IN SURFACE
            imFile=anatSource+'/frame0_registered.tiff'
            if not os.path.isfile(imFile):
                imFile=anatSource+'/frame0.tiff'

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
            fileName = 'phase_map_overlay_%s_Hz_%s_%s'%(str(np.around(freqs[freqIdx],4)),sessID,run)

            fig=plt.figure()
            plt.imshow(imSurf, 'gray')
            plt.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
            plt.colorbar()
            fig.suptitle(sessID+' cond'+str(cond)+' phaseMap', fontsize=20)
            plt.savefig(os.path.join(figOutDir,figName))
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

                figName = '%s_cond%s_legend.png'%(sessID,cond)
                fig=plt.figure()
                plt.imshow(legend,'nipy_spectral',vmin=0,vmax=2*np.pi)
                plt.savefig(os.path.join(figOutDir,figName))
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
        runFolder=glob.glob('%s/%s_%s*'%(sourceRoot,sessID,run))

        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"
        frameTimes,frameCond,frameCount = get_frame_times(planFolder)
        condList[idx]=frameCond[0]
      return condList

def get_interp_extremes(sourceRoot,sessID,runList,stimFreq):

    run=runList[0]
    
    runFolder=glob.glob('%s/%s_%s*'%(sourceRoot,sessID,run))
    frameFolder=runFolder[0]+"/frames/"
    planFolder=runFolder[0]+"/plan/"
    frameTimes,frameCond,frameCount = get_frame_times(planFolder)

    tMin=np.true_divide(1,stimFreq)
    nCycles=np.round(frameTimes[-1]/np.true_divide(1,stimFreq))
    tMax=np.true_divide(1,stimFreq)*(nCycles-1)
    return tMin, tMax

def get_reference_frame(sourceRoot,sessID,refRun=1):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot is not defined!")
    if sessID is None:
        raise TypeError("sessID is not defined!")
    
    runFolder=glob.glob('%s/%s_%s_*'%(sourceRoot,sessID,refRun))
    frameFolder=runFolder[0]+"/frames/"
        
    imFile=frameFolder+'frame0.tiff'
    imRef=misc.imread(imFile)
    return imRef

def get_run_parameters(sourceRoot,sessID,run):
      runFolder=glob.glob('%s/%s_%s_*'%(sourceRoot,sessID,run))

      frameFolder=runFolder[0]+"/frames/"
      planFolder=runFolder[0]+"/plan/"
      #GET STIM FREQUENCY
      planFile=open(planFolder+'parameters.txt')

      for line in planFile:
        if 'Cycle Rate' in line:
            break     
      idx = line.find(':')
      stimfreq = float(line[idx+1:])

      planFile.close()

      #GET FRAME RATE
      planFile=open(planFolder+'performance.txt')

      #READ HEADERS AND FIND RELEVANT COLUMNS
      headers=planFile.readline()
      headers=headers.split()

      count = 0
      while count < len(headers):
        if headers[count]=='frameRate':
            idx=count
            break
        count = count + 1

      x=planFile.readline().split()
      framerate = float(x[idx])
      planFile.close()

      return framerate, stimfreq

def run_get_surface(options):
      root = options.rootdir
      animal_id = options.animalid
      session = options.session

      source_root = os.path.join(root,'raw_data',animal_id,session)
      target_root = os.path.join(root,'analyzed_data',animal_id,session)

      get_surface(source_root, target_root, session)



def run_analysis(options):
      root = options.rootdir
      animalid = options.animalid
      session = options.session
      run = options.run

      source_root = os.path.join(root,'raw_data',animalid,session)
      target_root = os.path.join(root,'analyzed_data',animalid,session)


      framerate, stimfreq = get_run_parameters(source_root, session, run)


      interp = options.interpolate
      exclude_edges= options.exclude_edges
      rolling_mean= options.rolling_mean
      time_average = options.time_average
      if time_average is not None:
            time_average = int(time_average)
      motion = options.motion

      # #ANALYZE PHASE PER RUN
      analyze_periodic_data_per_run(source_root, target_root, session, [run], stimfreq, framerate, \
          interp, exclude_edges, rolling_mean, motion,saveImages=True,\
          loadCorrectedFrames=False,averageFrames=time_average,stimType='bar')
      # #adapt analysis thing to take list

def extract_options(options):

      parser = optparse.OptionParser()

      # PATH opts:
      parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/widefield', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/widefield]')
      parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
      parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
      parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: run1)")

      parser.add_option('-m', action='store_true', dest='motion', default=False, help="use motion corrected data")
      parser.add_option('-p', action='store_true', dest='interpolate', default=True, help="interpolate to an assumed steady frame rate")
      parser.add_option('-e', action='store_true', dest='exclude_edges', default=True, help="exclude first and last cycle of run")

      parser.add_option('-g', '--rollingmean', action='store_true', dest='rolling_mean', default=True, help='Boolean to indicate whether to subtract rolling mean from signal')
      parser.add_option('-w', '--timeaverage', action='store', dest='time_average', default=None, help='Size of time window with which to average frames (integer)')


      parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

      (options, args) = parser.parse_args(options)


      return options


def main(options):
      options = extract_options(options)
      run_get_surface(options)
      run_analysis(options)



if __name__ == '__main__':
    main(sys.argv[1:])
