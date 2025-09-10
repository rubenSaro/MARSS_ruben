from nipype.interfaces import fsl
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.stats import zscore
import numpy as np
import os
import seaborn as sns
import shutil
import re
from pathlib import Path

# Modifications by Ruben Sanchez-Romero, Sep 2025 
# Add functionality to work with XXX.nii.gz files
# Uses Path instead of os.path to create directories
# adds a function split_nii_ext to work extract filename and extension in XXX.nii and XXX.nii.gz files

# Added by Ruben Sanchez-Romero, September 2025
def split_nii_ext(filename: str):
    """
    Return (stem, extension) where extension is either
    '.nii', '.nii.gz', or the normal last extension.
    """
    base = Path(filename).name
    m = re.match(r"^(?P<stem>.*?)(?P<ext>\.nii(?:\.gz)?)$", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"File does not end with .nii or .nii.gz: {filename}")
    return m.group("stem"), m.group("ext")


def MARSS_getMPs(fn, MB, workingDir):
    # Ruben: this get motion parameters using mcflirt

    # Load volume timeseries
    V = nib.load(fn)

    if V.shape[2] % MB != 0:
        raise ValueError(f"Number of slices and MB factor are incompatible for {fn}.")

    #p, f = os.path.split(fn)
    #f = os.path.splitext(f)[0]
    # the above will have problems dealing with files xxx.nii.gz
    f,ext = split_nii_ext(fn)

    # Check workingDir for MPs
    # Use Path instead of os.path
    #mp_path = os.path.join(workingDir, f"rp_{f}.txt")
    mp_path = Path(workingDir) / f"rp_{f}.txt"

    if not os.path.exists(mp_path):
        
        mcflt = fsl.MCFLIRT()
        
        mcflt.inputs.in_file = fn
        
        mcflt.inputs.cost = 'mutualinfo'
        
        #rp_path = os.path.join(workingDir, f"rp_{f}{ext}")
        rp_path = Path(workingDir) / f"rp_{f}{ext}"
        mcflt.inputs.out_file = rp_path
        mcflt.inputs.output_type = "NIFTI"

        mcflt.inputs.save_rms = False
        mcflt.inputs.save_plots = True
        mcflt.inputs.save_mats = False
        
        res = mcflt.run()

        # delete motion-corrected output
        os.remove(rp_path)

    # Copy the MP text file to the working directory
    if not os.path.exists(mp_path):
        
        mp_file_src = Path(workingDir) / f"{rp_path}.par"      
        shutil.copyfile(mp_file_src, mp_path)

    return mp_path

def MARSS_main(timeseriesFile, MB, workingDir,*args):
    # Get name of run
    #runName = os.path.splitext(os.path.basename(timeseriesFile))[0]
    # the above has problem handling files like XXX.nii.gz
    runName,ext = split_nii_ext(timeseriesFile)

    # Create a new folder
    #runDir = os.path.join(workingDir, runName)
    #if not os.path.exists(runDir):
    #    os.makedirs(runDir)
    runDir = Path(workingDir) / runName
    runDir.mkdir(parents=True, exist_ok=True)

    varargin = args
    
    userMPprovided = False
    
    if len(varargin) > 0:
        if len(varargin) > 1:
            raise TypeError("Too many arguments (expected one optional argument).")
        else:
            userMPpath = varargin[0]
            print('User provided motion parameters. Skipping motion parameter estimation...')
            userMPprovided = True
            shutil.copy2(userMPpath,runDir)
            _, f = os.path.split(userMPpath)
            preMARSS_MPpath = os.path.join(runDir,f)
            matrix = text_to_matrix(preMARSS_MPpath)


    print('Generating pre-MARSS Motion Parameters and Slice Correlations...')
    # Generate motion parameters for that run and save them
    if not userMPprovided:
        preMARSS_MPpath = MARSS_getMPs(timeseriesFile, MB, runDir)
        matrix = text_to_matrix(preMARSS_MPpath)
        
    MARSS_mbCorrPlot(timeseriesFile, matrix, MB,runDir)  
    
    print('Performing MARSS Procedure...')
    [postMARSS_fname, postMARSS_avgSlcArt_fname] = MARSS_removeSliceArtifact(timeseriesFile, MB, matrix, runDir)
    
    print('Generating post-MARSS Motion Parameters and Slice Correlations...')
    postMARSS_MPpath = MARSS_getMPs(postMARSS_fname, MB, runDir)
    matrix = text_to_matrix(preMARSS_MPpath)
    MARSS_mbCorrPlot(postMARSS_fname, matrix, MB,runDir)   

def text_to_matrix(file_path):
    """Converts a text file to a numpy matrix."""

    with open(file_path, 'r') as f:
        lines = f.readlines()

    matrix = []
    for line in lines:
        row = [float(x) for x in line.split()]  # Adjust split() if delimiter is different
        matrix.append(row)

    return np.array(matrix)

def MARSS_mbCorrPlot(fname,MPs,MB,runDir):
    timeSeriesDat = nib.load(fname);
    timeSeriesDat = timeSeriesDat.get_fdata()
    d = np.mean(np.mean(timeSeriesDat, axis = 0), axis = 0)
    d = d.T    

    # % nuisance regressor design matrix [intercept, linearDetrend, MPs,squared
    # % MPs, derivatives of MPs, squared derivatives]
    intercept = (np.ones((int(d.shape[0]),1)))
    linearDetrend = (np.arange(1, (int(d.shape[0]))+1))
    linearDetrend = linearDetrend[:,np.newaxis]
    MPs_derivatives = np.vstack(((np.zeros(6), np.diff(MPs,axis = 0))))
    squared_MPs_derivatives =  np.vstack((np.zeros(6), np.diff(MPs,axis = 0)))**2 #cannot assign to function call
    X = np.hstack((intercept, linearDetrend,  MPs, MPs**2, MPs_derivatives, squared_MPs_derivatives))
    
    # rd = zeros(size(d));
    rd = np.zeros(d.shape)

    for i in range(d.shape[1]):
        # Solve the least-squares problem X \ d[:,i] -> in Python it's np.linalg.lstsq
        beta = np.linalg.lstsq(X, d[:, i], rcond=None)[0]
        
        # Calculate the residual: d[:, i] - X @ beta
        rd[:, i] = d[:, i] - X @ beta

    [avgS_Z, avgNS_Z] = MARSS_avgMBcorr(rd,MB)

    Z = np.mean(avgS_Z - avgNS_Z)
    correlation_difference = (np.exp(2 * Z) - 1) / (np.exp(2 * Z) + 1)
   
    # out.corrMat_motionRegressed = corr(rd);
    corr = np.corrcoef(rd, rowvar=False) 
    
    # Create the heatmap
    # Modified by Ruben Sanchez-Romero, Sept 2025, to use seaborn and a seismic plot.
    plt.figure(figsize=(8, 6))
    #heat = plt.pcolor(corr, cmap='viridis')
    #plt.gca().invert_yaxis()
    #cbar = plt.colorbar(heat)
    #cbar.set_label('Pearson\'s R', rotation = 270, fontsize = 16, labelpad=10)
    ax = sns.heatmap(corr,cmap="seismic",center=0,cbar_kws={"label": "Pearson's R"})
    ax.invert_yaxis()
    plt.title("ΔR = " + str(np.round(correlation_difference, 4)), fontsize=20)
    plt.xlabel("Slice #", fontsize=16)
    plt.ylabel("Slice #", fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.yaxis.label.set_rotation(270)
    cbar.ax.yaxis.labelpad = 10

    #p, f = os.path.split(fname)
    #f = os.path.splitext(f)[0]
    f,ext = split_nii_ext(fname)

    plt.savefig(os.path.join(runDir, f"corrMatrix_{f}.png"))
    # save correlation matrix as png to runDir (will require some filename manipulations)

def MARSS_avgMBcorr(dat, MB):
    c = np.corrcoef(dat, rowvar=False)
    cZ = 0.5 * np.log((1 + c) / (1 - c))
    sgap = c.shape[0] / MB

    num_rows = c.shape[0]  
    avgS_Z = np.zeros((num_rows, 1))
    avgNS_Z = np.zeros((num_rows, 1))

    # empty array to append ind to 
    rows = []

    for i in range(c.shape[0]):
        # from i-sgap to 1 in increments of -sgap
        ind = np.concatenate((np.arange(i - sgap, -1, -sgap), np.arange(i + sgap, num_rows, sgap)))     
        ind = np.array(ind, dtype=int)
        
        avgS_Z[i] = np.mean(cZ[i, ind])
        
        #adjacent slices
        ind = np.concatenate((ind - 1, ind + 1))  # Create adjacent indices
        ind = ind[(ind >= 0) & (ind < num_rows)]  # Keep only valid indices
        avgNS_Z[i] = np.mean(cZ[i, ind-1])  # Adjust index for zero-based indexing

    return [avgS_Z, avgNS_Z]
    
def MARSS_estimateSliceArtifact(Y,MB,X):
    if Y.shape[2] % MB != 0:
        raise ValueError("# of slices and MB factor are incompatible.")

    nSlices = Y.shape[2]
    nSliceSets = nSlices // MB
    data = np.squeeze(np.mean(np.mean(Y, axis=1), axis=0)).T

    X = zscore(X, axis=0, ddof=1) 

    sameSetData = np.zeros_like(data)
    artifactEstimate = np.zeros_like(data)
    globalSignal = np.zeros_like(data)

    for j in range(nSlices):
        sliceSet = j % nSliceSets

        sameSliceSet = np.zeros(nSlices, dtype=bool)  # Initialize boolean array for same slice set
        lowerSlices = np.arange(sliceSet, j, nSliceSets)  # Get lower slices
        higherSlices = np.arange(j + nSliceSets, nSlices, nSliceSets)  # Get higher slices
        sameSliceSet[np.concatenate((lowerSlices, higherSlices))] = True  # Set slices in the same slice set

        # Calculate sameSetData for the current slice j
        sameSetData[:, j] = np.mean(data[:, sameSliceSet], axis=1)
    
        # Exclude current slice from global signal calculation
        sameSliceSet[j] = True
 
        globalSignal[:, j] = np.mean(data[:, ~sameSliceSet], axis=1)
    
        # Design matrix: [intercept, globalSignal, MPs (X)]
        Xout = np.column_stack((np.ones(data.shape[0]), zscore(globalSignal[:, j],ddof = 1), X))
    
        # Perform the regression (solve the system of equations)
        B = np.linalg.lstsq(Xout, sameSetData[:, j], rcond=None)[0]
    
        # Compute artifact estimate
        artifactEstimate[:, j] = sameSetData[:, j] - Xout @ B

    return [artifactEstimate, globalSignal]


def MARSS_removeSliceArtifact(filename,MB,MPs,working_dir):
    
    working_dir = Path(working_dir)
    
    img = nib.load(filename)
    Y = img.get_fdata()
    #V = img.header
    hdr = img.header
    hdr.set_data_dtype(np.float32)  
    #Y = Y.astype(np.float64)

    if Y.shape[2] % MB != 0:
        print(f"Warning: # of slices and MB factor are incompatible for {filename}. Skipping.")

    MPs = (MPs - np.mean(MPs)) / np.std(MPs)  # Z-score normalization
    # Nuisance parameter design matrix
    Xest = np.hstack([
            MPs,
            MPs ** 2,
            np.vstack([np.zeros((1, 6)), np.diff(MPs, axis=0)]),
            np.vstack([np.zeros((1, 6)), np.diff(MPs, axis=0)]) ** 2])
    
    # Estimate artifact signal in each slice
    [artifactEstimate, nonsliceGlobalSignal] = MARSS_estimateSliceArtifact(Y, MB, Xest) 

    Ya = np.zeros_like(Y)
    Yart = np.zeros_like(Y)   
    for j in range(artifactEstimate.shape[1]):
        # Final MARSS design matrix
        Xcalc = np.column_stack([np.ones(artifactEstimate.shape[0]), (artifactEstimate[:, j] - np.mean(artifactEstimate[:, j])) / np.std(artifactEstimate[:, j]), 
                                 (nonsliceGlobalSignal[:, j] - np.mean(nonsliceGlobalSignal[:, j])) / np.std(nonsliceGlobalSignal[:, j]),
                                 (Xest - np.mean(Xest, axis=0)) / np.std(Xest, axis=0)])

        Yt = Y[:, :, j, :].reshape(-1, Y.shape[3]).T
        # Perform regression
        B = np.linalg.lstsq(Xcalc, Yt, rcond=None)[0]

        art = np.outer(Xcalc[:, 1], B[1, :])
        # Subtract artifact estimation from original timeseries data
        Y_diff = Yt.T - art.T
        Yta = np.reshape(Y_diff, (Y.shape[0], Y.shape[1], Y.shape[3]))

        Yartt = np.reshape(art.T, (Y.shape[0], Y.shape[1], Y.shape[3]))
        Ya[:, :, j, :] = Yta
        Yart[:, :, j, :] = Yartt

        # Create filenames for output
        #base_name = os.path.basename(filename)
        #f, x = os.path.splitext(base_name)
        f,ext = split_nii_ext(filename)

    for j in range(Y.shape[3]):
        # Corrected Data
        corrected_filename = working_dir / f'za_{f}{ext}'
        
        # Isolated artifact timeseries
        artifact_filename = working_dir / f'{f}_slcart{ext}'
        
    nib.save(nib.Nifti1Image(Ya, img.affine), corrected_filename)
    nib.save(nib.Nifti1Image(Yart, img.affine), artifact_filename)
             
    Yaavg = np.mean(np.abs(Yart), axis=3)

    # Average artifact distribution
    avg_artifact_filename = working_dir / f'{f}_AVGslcart{ext}'
    nib.save(nib.Nifti1Image(Yaavg, img.affine), avg_artifact_filename)

    postMARSS_fname = working_dir / f'za_{f}{ext}'
    postMARSS_avgSlcArt_fname = working_dir / f'{f}_AVGslcart{ext}'    
        
    return [postMARSS_fname, postMARSS_avgSlcArt_fname]