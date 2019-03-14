#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:38:30 2019

@author: shariba
"""

import os
import numpy as np
from skimage import transform

def writeDicom2Png(dicomVol, resultFolder):
    import cv2
    for i in range (len(dicomVol)):
        image = (np.maximum(dicomVol[i, :,:],0) / dicomVol[i, :,:].max()) * 255.0
        img_2d_scaled = np.uint8(image)
        cv2.imwrite(os.path.join(resultFolder, str(i)+'.png'),img_2d_scaled)
    
def writeDicom2Png_256(dicomVol, resultFolder):
    import cv2
    for i in range (len(dicomVol)):
        image = (np.maximum(dicomVol[i,:, :],0) / dicomVol[i, :,:].max()) * 255.0
        img_2d_scaled = transform.resize(image, [256, 256])
        cv2.imwrite(os.path.join(resultFolder, str(i)+'.png'),img_2d_scaled)

# volume
def reshapeDicomVol_256(dicomVol):
    img_reshape=np.zeros([256, 256, len(dicomVol)])
    for i in range (len(dicomVol)):
        image = (np.maximum(dicomVol[i, :,:],0) / dicomVol[i, :,:].max()) * 255.0
        img_reshape[:, :, i] = transform.resize(image, [256, 256]).T
    return img_reshape
