#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:48:25 2019

@author: shariba
"""

import os
import pydicom
import numpy as np
import scipy.ndimage

# import write functions from util
from utils import writeDicom2Png, reshapeDicomVol_256, writeDicom2Png_256
from plotFunc import plot_3d, plot_2d

def get_dicomInfo(ds):
    if 'PixelData' in ds:
        print("Slice location...:", ds.get('SliceLocation', "(missing)")) 
        rows = int(ds.Rows)
        cols = int(ds.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(ds.PixelData)))
        if 'PixelSpacing' in ds:
            print("Pixel spacing....:", ds.PixelSpacing)
            
# Load the scans in given folder path
def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# resampling (isotropic volume of 1mmx1mmx1mm)
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor  
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing

# some preprocessing
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

# zero centering so that mean is 0
PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image
    
def writeDicom2nii3D_256(dicomVol, resultFolder, fileName):        
    import nibabel as nib
    AffineMat = np.array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
    niiDataFile = os.path.join(resultFolder, fileName + '.nii.gz')
    dicomReshapedVol =  nib.Nifti1Image(reshapeDicomVol_256(np.array(dicomVol, dtype=np.float)), AffineMat)
    nib.save(dicomReshapedVol, niiDataFile) 
 
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="converter from dicom to png --> npz compression?", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataDir", type=str, default="/Users/shariba/dataset/EMPIRE/lyonInsaLung/4DCT-Dicom_0/", help="see code")
    parser.add_argument("--resultDir", type=str, default="pngConverted", help="see code")
    args = parser.parse_args()
    return args
             
if __name__ == "__main__":
    
    # use these setting only to debug or see the outputs (both 2D, 3D available)
    useDebug = 0
    useViz3D = 0
    writeDicom=1
    writeniiVolume=1
    
    writePngImages=1
    valArgs = get_args()
    
    inputDIR = valArgs.dataDir
    patients = os.listdir(inputDIR)
    patients.sort()
    
    # loop it here if you want all the images to be converted
    for k in range (0, len(patients)):
        first_patient = load_scan(inputDIR + patients[k])
        get_dicomInfo(first_patient[k])
        first_patient_pixels = get_pixels_hu(first_patient)
        
        if useDebug:
            print('before')
            plot_2d(first_patient_pixels)
            
        # resampling here
        pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
        print("Shape before resampling\t", first_patient_pixels.shape)
        print("Shape after resampling\t", pix_resampled.shape)
        
        if useViz3D:
            plot_3d(pix_resampled, 400.0)
        
        # perform other preprocessing
        # Note: you can do this during training
        pix_resampled = normalize(pix_resampled)
        pix_resampled = zero_center(pix_resampled)
        
        if useDebug:
            print('after')
            plot_2d(pix_resampled)
            
        if writePngImages:
            resultFolder = os.path.join(inputDIR, valArgs.resultDir, patients[k])
            os.makedirs(resultFolder, exist_ok=True)
            writeDicom2Png_256(pix_resampled, resultFolder)


        # TODO: convert to .nii (needed for 3D volume registration)
        if writeniiVolume:
            resultFolder = os.path.join(inputDIR, valArgs.resultDir)
            writeDicom2nii3D_256(pix_resampled, resultFolder, patients[k])

