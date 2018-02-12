import sys
sys.path.append('/nas/volume1/shared/cesar/widefield/dataAnalysis/Python_scripts/WiPy_package')
import numpy as np
import glob
import optparse
import os
import cv2
import matplotlib
import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors
from WiPy import *


def write_dict_to_json(pydict, writepath):
    jstring = json.dumps(pydict, indent=4, allow_nan=True, sort_keys=True)
    f = open(writepath, 'w')
    print >> f, jstring
    f.close()

def cm_to_degrees(size_cm,view_distance):
    size_degrees = 2 * np.degrees(np.arctan((size_cm/2.0)/view_distance))
    return size_degrees

def convert_boundary(boundary_pix ,center_deg ,cm_per_pix ,view_distance):
    
    boundary_cm = boundary_pix*cm_per_pix
    boundary_deg = cm_to_degrees(boundary_cm,view_distance)

    boundary_relative_deg = boundary_deg - center_deg
    
    return boundary_relative_deg

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

def get_run_parameters(sourceRoot,sessID,run):
      runFolder=glob.glob('%s/%s_%s_*'%(sourceRoot,sessID,run))
      print('%s/%s_%s_*'%(sourceRoot,sessID,run))

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


def visualize_single_run(sourceRoot, targetRoot, sessID, runList, smooth_fwhm=None, magRatio_thresh=None,\
    analysisDir=None,motionCorrection=False, flip = False, modify_range=True, mask =None):

    anatSource=os.path.join(targetRoot,'Surface')
    motionDir=os.path.join(targetRoot,'Motion')
    motionFileDir=os.path.join(motionDir, 'Registration')


    fileInDir=os.path.join(analysisDir,'SingleRunData','Files')
    figOutDirRoot=os.path.join(analysisDir,'SingleRunData','Figures')
    fileOutDirRoot=os.path.join(analysisDir,'SingleRunData','Files')
    #for file name
    smoothString=''
    threshString=''

    condList = get_condition_list(sourceRoot,sessID,runList)

    # runCount = 0
    # run = runList[runCount]

    for runCount,run in enumerate(runList):
        print('Current Run: %s'%(run))
        cond = condList[runCount]
        figOutDir=os.path.join(figOutDirRoot,'cond%s'%(str(int(cond))))
        if not os.path.exists(figOutDir):
            os.makedirs(figOutDir)

        #LOAD MAPS
        fileName = '%s_%s_map.npz'%(sessID, run)
        f=np.load(os.path.join(fileInDir,fileName))
        phaseMap=f['phaseMap']
        magRatioMap=f['magRatioMap']

        if smooth_fwhm is not None:
            phaseMap=smooth_array(phaseMap,smooth_fwhm,phaseArray=True)
            magRatioMap=smooth_array(magRatioMap,smooth_fwhm)
            smoothString='_fwhm_'+str(smooth_fwhm)


        #set phase map range for visualization
        if modify_range:
            phaseMapDisplay=np.copy(phaseMap)
            phaseMapDisplay[phaseMap<0]=-phaseMap[phaseMap<0]
            phaseMapDisplay[phaseMap>0]=(2*np.pi)-phaseMap[phaseMap>0]

            rangeMin=0
            rangeMax=2*np.pi
        else:
            phaseMapDisplay=np.copy(phaseMap)
            rangeMin=-np.pi
            rangeMax=np.pi


        #apply threshhold
        if magRatio_thresh is not None:
            phaseMapDisplay[magRatioMap<magRatio_thresh]=np.nan
            threshString='_thresh_'+str(magRatio_thresh)
        else:
            magRatiothresh = np.max(magRatioMap)
            phaseMapDisplay[magRatioMap<magRatio_thresh]=np.nan
            threshString='_thresh_'+str(magRatio_thresh)

        #load surface for overlay
        #READ IN SURFACE

        
        imFile=anatSource+'/frame0_registered.tiff'
        if not os.path.isfile(imFile):
            imFile=anatSource+'/frame0.tiff'

        imSurf=cv2.imread(imFile,-1)
        szY,szX=imSurf.shape
        imSurf=np.true_divide(imSurf,2**12)*2**8

        if flip:
            print('Flipping Images')
            imSurf = np.fliplr(imSurf)
            phaseMapDisplay = np.fliplr(phaseMapDisplay)

        if motionCorrection:
            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']
            padDown=int(boundaries[0])
            padUp=int(szY-boundaries[1])
            padLeft=int(boundaries[2])
            padRight=int(szX-boundaries[3])

            phaseMapDisplay=np.pad(phaseMapDisplay,((padDown,padUp),(padLeft,padRight)),'constant',constant_values=((np.nan, np.nan),(np.nan,np.nan)))
        #plot
        fileName = 'overlay_images_%s_cond%s%s%s.png'%(sessID,str(int(cond)),smoothString,threshString)


        dpi = 80
        szY,szX = imSurf.shape
        # What size does the figure need to be in inches to fit the image?
        figsize = szX / float(dpi), szY / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis('off')

        ax.imshow(imSurf, 'gray')
        ax.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=rangeMin,vmax=rangeMax)

        fig.savefig(os.path.join(figOutDir,fileName), dpi=dpi, transparent=True)
        plt.close()

        #output masked image as well, if indicated
        if mask is not None:
            #load mask
            maskFile=targetRoot+'/Sessions/'+sessID+'/masks/Files/'+mask+'.npz'
            f=np.load(maskFile)
            maskM=f['maskM']

            #apply mask
            phaseMapDisplay[maskM==0]=np.nan

            #plot
            outFile=outFile = '%s_cond%s%s%s_phaseMap_mask_%s_image.png'%\
            (figOutDir+sessID,str(int(cond)),smoothString,threshString,mask)

            #Create a figure of the right size with one axes that takes up the full figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])

            # Hide spines, ticks, etc.
            ax.axis('off')
            ax.imshow(imSurf, 'gray')
            ax.imshow(phaseMapDisplay,'nipy_spectral',alpha=.5,vmin=rangeMin,vmax=rangeMax)

            fig.savefig(outFile, dpi=dpi, transparent=True)
            plt.close()


        unique_phase = np.unique(phaseMapDisplay[~np.isnan(phaseMapDisplay)])
        
        data_dict = dict()
        #hard-coding screen parameters, for now
        szScreenY=1024
        szScreenX=1920
        
        screen_size_x_pixels = szScreenX
        screen_size_x_cm = 103.0
        screen_size_y_cm = 58.0
        view_distance_cm = 60.0
        cm_per_pix = float(screen_size_x_cm)/float(screen_size_x_pixels)
        #TODO: Create a text file with these values that the user provides
        screen_size_x_degrees = cm_to_degrees(screen_size_x_cm,view_distance_cm)
        screen_size_y_degrees = cm_to_degrees(screen_size_y_cm,view_distance_cm)
        
        data_dict['screen_params']=dict()
        data_dict['screen_params']['screen_size_x_pixels'] = szScreenX
        data_dict['screen_params']['screen_size_y_pixels'] = szScreenY
        data_dict['screen_params']['screen_size_x_cm'] = screen_size_x_cm
        data_dict['screen_params']['screen_size_y_cm'] = screen_size_y_cm
        data_dict['screen_params']['view_distance_cm'] = view_distance_cm
        data_dict['screen_params']['screen_size_x_degrees'] = screen_size_x_degrees
        data_dict['screen_params']['screen_size_t_degrees'] = screen_size_y_degrees

        x = np.linspace(0, 2*np.pi, szScreenX)
        y = np.linspace(0, 2*np.pi, szScreenY)
        xv, yv = np.meshgrid(x, y)

        thick= 5

        if cond==1 or cond ==2:
            if cond ==1:
                legend=xv
                boundary_left = np.where(legend[0,:]>=np.min(unique_phase))[0][0]
                boundary_right = np.where(legend[0,:]>=np.max(unique_phase))[0][0]  
            else:
                xv=(2*np.pi)-xv
                legend = xv
                boundary_left = np.where(legend[0,:]<=np.max(unique_phase))[0][0]
                boundary_right = np.where(legend[0,:]<=np.min(unique_phase))[0][0] 
                
            legend[:,boundary_left-thick:boundary_left+thick] = 0
            legend[:,boundary_right-thick:boundary_right+thick] = 0  
            
            #convert to degrees
            boundary_left_deg = convert_boundary(boundary_left, screen_size_x_degrees/2.0, cm_per_pix, view_distance_cm)
            boundary_right_deg = convert_boundary(boundary_right, screen_size_x_degrees/2.0, cm_per_pix, view_distance_cm)
            
            print('****************************************************************')
            print('Boundaries in Degrees')
            print('----------------------------------------------------------------')
            print('Left: %10.4f, Right: %10.4f'%(boundary_left_deg, boundary_right_deg))
            print('***************************************************************')
            
            data_dict['screen_boundaries']=dict()
            data_dict['screen_boundaries']['boundary_left_pixels'] = boundary_left
            data_dict['screen_boundaries']['boundary_right_pixels'] = boundary_right
            data_dict['screen_boundaries']['boundary_left_degrees'] = boundary_left_deg
            data_dict['screen_boundaries']['boundary_right_degrees'] = boundary_right_deg
            
    #         #save to Json
            fileName ='screen_boundaries_%s_%s%s%s.json'% (sessID,run,smoothString,threshString)
            write_dict_to_json(data_dict, os.path.join(figOutDir,fileName))
        elif cond==3 or cond==4:
            if cond ==3:
                y = np.linspace(0, 2*np.pi, szScreenY)
                xv, legend = np.meshgrid(x, y)
                boundary_down =  np.where(legend[:,0]>=np.max(unique_phase))[0][0]
                boundary_up = np.where(legend[:,0]>=np.min(unique_phase))[0][0]
            else:
                y = np.linspace(0, 2*np.pi, szScreenY)
                xv, yv = np.meshgrid(x, y)
                legend=(2*np.pi)-yv

                boundary_down =  np.where(legend[:,0]<=np.min(unique_phase))[0][0]
                boundary_up = np.where(legend[:,0]>=np.max(unique_phase))[0][0]
                  
            legend[boundary_up-thick:boundary_up+thick,:] = 0
            legend[boundary_down-thick:boundary_down+thick,:] = 0  
            
            #convert to degrees
            boundary_down_deg = convert_boundary(boundary_down, screen_size_y_degrees/2.0, cm_per_pix, view_distance_cm)
            boundary_up_deg = convert_boundary(boundary_up, screen_size_y_degrees/2.0, cm_per_pix, view_distance_cm)

            print('****************************************************************')
            print('Boundaries in Degrees')
            print('----------------------------------------------------------------')
            print('Up: %10.4f, Down: %10.4f'%(boundary_up_deg, boundary_down_deg))
            print('****************************************************************')
            data_dict['screen_boundaries']=dict()
            data_dict['screen_boundaries']['boundary_up_pixels'] = boundary_up
            data_dict['screen_boundaries']['boundary_down_pixels'] = boundary_down
            data_dict['screen_boundaries']['boundary_up_degrees'] = boundary_up_degrees
            data_dict['screen_boundaries']['boundary_down_degrees'] = boundary_down_degrees
        fileName = 'screen_boundaries_%s_%s%s%s.png'% (sessID,run,smoothString,threshString)
        plt.imshow(legend,'nipy_spectral',vmin=0, vmax=2*np.pi)
        plt.savefig(os.path.join(figOutDir,fileName))
        plt.close()


def visualize_run(options):
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

    ratio_thresh = options.ratio_thresh
    if ratio_thresh is not None:
        ratio_thresh = float(ratio_thresh)
    smooth_fwhm = options.smooth_fwhm
    if smooth_fwhm is not None:
        smooth_fwhm = int(smooth_fwhm)
    flip = options.flip

    analysis_root = os.path.join(target_root,'Analyses')
    analysis_dir=get_analysis_path_phase(analysis_root, stimfreq, interp, exclude_edges, rolling_mean, \
    motion, time_average)

    visualize_single_run(source_root, target_root, session, [run], smooth_fwhm, ratio_thresh, analysis_dir, motion,flip)

def extract_options(options):

      parser = optparse.OptionParser()

      # PATH opts:
      parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/widefield', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/widefield]')
      parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
      parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
      parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: run1)")

      #specifications of analysis to visualize
      parser.add_option('-m', action='store_true', dest='motion', default=False, help="use motion corrected data")
      parser.add_option('-p', action='store_true', dest='interpolate', default=True, help="interpolate to an assumed steady frame rate")
      parser.add_option('-e', action='store_true', dest='exclude_edges', default=True, help="exclude first and last cycle of run")
      parser.add_option('-g', '--rollingmean', action='store_true', dest='rolling_mean', default=True, help='Boolean to indicate whether to subtract rolling mean from signal')
      parser.add_option('-w', '--timeaverage', action='store', dest='time_average', default=None, help='Size of time window with which to average frames (integer)')

      #visualization options
      parser.add_option('-f', '--fwhm', action='store', dest='smooth_fwhm', default=None, help='full-width at half-max size of kernel for smoothing')
      parser.add_option('-t', '--thresh', action='store', dest='ratio_thresh', default=None, help='magnitude ratio cut-off threshold')
      parser.add_option('-l', '--flip', action='store_true', dest='flip', default=False, help='boolean to indicate whether to perform horizontal flip on phase map images (to match actual orientation of FOV)')

      parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

      (options, args) = parser.parse_args(options)


      return options


def main(options):
      options = extract_options(options)
      visualize_run(options)


if __name__ == '__main__':
    main(sys.argv[1:])

