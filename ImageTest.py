# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 06:38:05 2024

@author: USER
"""

#import numpy as np
#import torch


def WriteImage(image, fPath):
    npimage = image.numpy()
    with open (fPath, 'wb') as file:
        file.write(npimage.tobytes())
    print('Image saved:', fPath)

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
   
def TestCImage():
    image = CImage(512,512)
    image.Read1ImageOfVolume('F:/Data/Noise reduction DATA/SCANPLAN_7346_WideFOV_DFS/Offline_14_01_2021 - NoANR/cf483db6_0_Recon0.raw')
    image.WriteToFile('Try/centralFromClass')
    image.show()
    image2 = image.CreateCopy()
    image.Multiply(0.5)
    image.WriteToFile('Try/centralFromClass_halved')
    image.show()
    image2.WriteToFile('Try/centralFromClass_copy')
    image2.show()

def TestPatterns():
    image = CImage(4,10,bInit=True)
    image.SetConstant(10)
    print(image.__dict__)
    image.AddConstantVertically(5, 3)
    print(image.__dict__)
    
def TestStringValues():
    s = 'dd0c04c3_0_Recon0_Polar_width1152_height544_hPad16_vPad64_hsmooth51_SRC_im28.float.rimg'
    print ('string', s)
    print ('get value _im', GetValue(s,'_im'))
    print ('set value _im 3', SetValue(s,'_im', 3))
    print ('set value _im 456', SetValue(s,'_im', 456))
    print ('set value _width 22', SetValue(s,'_width', 22))
    print ('Remove value _width', RemoveValue(s,'_width'))
    print ('Remove value _im', RemoveValue(s,'_im'))
    

def main():
    print('Handle Images Main')
    #testReadAndWrite()
    #TestCImage()
    #TestPatterns()
    TestStringValues()

if __name__ == '__main__':
    main()
