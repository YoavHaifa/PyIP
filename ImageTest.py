# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 06:38:05 2024

@author: USER
"""

#import numpy as np
#import torch

from Image import CImage


"""
def testReadAndWrite():
    origImages = ReadImages('F:/Data/Noise reduction DATA/SCANPLAN_7346_WideFOV_DFS/Offline_14_01_2021 - NoANR/cf483db6_0_Recon0.raw')
    print('pimage.size()', origImages.size())
    nImages = len(origImages)
    origCentral = origImages[int(nImages/2),:,:] 
    print('nImages {}, image size {}'.format(nImages, origCentral.size()))
    WriteImage(origCentral, 'Try/central', 512, 512)

    plt.matshow(origCentral, cmap='gray')
    plt.show()  

    targetImages = ReadImages('F:/Data/Noise reduction DATA/SCANPLAN_7346_WideFOV_DFS/Offline_14_01_2021 - ANR5/cf483db6_0_Recon0_ANR5.raw')
    targetCentral = targetImages[int(nImages/2),:,:] 

    plt.matshow(targetCentral, cmap='gray')
    plt.show()  

    diff = origCentral - targetCentral

    plt.matshow(diff, cmap='gray')
    plt.show()  
    """
   
def TestCImage():
    print('*** Test CImage')
    image = CImage(512,512)
    image.ReadImage('D:\Data\ImageTest/10688773_0_Recon0_HR_width512_height512_im157_SRC__1a.float.rimg')
    image.WriteToFile('SavedFromCImage')
    image.Show()
    image2 = image.CreateCopy()
    image.Multiply(0.5)
    image.WriteToFile('SavedFromCImage_halved')
    image.Show()
    image2.WriteToFile('SavedFromCImage_copy')
    image2.Show()

def TestPatterns():
    image = CImage(4,10,bInit=True)
    image.SetConstant(10)
    print(image.__dict__)
    image.AddConstantVertically(5, 3)
    print(image.__dict__)
    

def main():
    print('*** Test CImage Main')
    #testReadAndWrite()
    TestCImage()
    TestPatterns()

if __name__ == '__main__':
    main()
