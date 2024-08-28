# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 00:03:07 2024
Measure image flatness - tool for building loss function 

@author: yoav.bar
"""

#from os import path
#import math
import torch
import time

#from Utils import GetDataRoot
from Image import CImage
from Volume import CVolume


sVolumeFileNameAi = 'd:\Dump\BP_PolyAI_Output_width512_height512.float.dat'
sVolumeFileNameNominal = 'd:\Dump\BP0_Output_width512_height512.float.dat'

sImageFileName = 'Poly_Calib/Centered_250FOV_All_Slices_matrix512.short.TImage'

threshMin = 850
threshMax = 1150

bTiming = False

def CreateRadiusImage():
    start = time.monotonic()
    image = CImage(512,512,bInit=True)
    centerX = 255.5
    centerY = 255.5
    
    xVector = torch.arange(0,512)
    xDiff = xVector - centerX
    xDiff2 = torch.pow(xDiff,2)
    
    for y in range(512):
        yDiff2 = (y-centerY)**2
        sumVec = xDiff2 + yDiff2
        image.pData[y] = torch.sqrt(sumVec)
        
        #for x in range(512):
        #    radius = math.sqrt(yDiff2 + xDiff2[x])
        #    image.pData[y,x] = radius
    
    if bTiming:
        end = time.monotonic()
        elapsed = end - start
        print(f"<CreateRadiusImage> Elapsed time: {elapsed:.3f} seconds")
        image.WriteToFile("RadiusImage_fast")
    return image

def Peel(mask):
    # Peel Vertical
    maskPre = mask[:,0:510,:]
    maskPost = mask[:,2:512,:]
    #print(f'{maskPre.shape=}')
    #print(f'{maskPost.shape=}')
    center = mask[:,1:511,:]
    center = torch.where(maskPre > 0, center, 0)
    center = torch.where(maskPost > 0, center, 0)
    mask[:,1:511,:] = center
    
    #Peel Horizontal
    maskUp = mask[:,:,0:510]
    maskDown = mask[:,:,2:512]
    #print(f'{maskPre.shape=}')
    #print(f'{maskPost.shape=}')
    center = mask[:,:,1:511]
    center = torch.where(maskUp > 0, center, 0)
    center = torch.where(maskDown > 0, center, 0)
    mask[:,:,1:511] = center
    return mask
    

def ComputeMask(volume):
    volumeShifted = volume.pImages + 1000
    mask = torch.where(volumeShifted >= threshMin, volumeShifted, 0)
    mask = torch.where(mask <= threshMax, mask, 0)
    volume.DumpSimilarVolume(mask, "maskAfterThreshold")
    mask[:,0,:] = 0
    mask[:,-1,:] = 0
    mask[:,:,0] = 0
    mask[:,:,-1] = 0
    volume.DumpSimilarVolume(mask, "maskAfterClip")
    mask = Peel(mask)
    
    print(f'{mask.shape=}')
    return mask

class CPolyScorer:
    """
    Compute scores for poly tables by:
        1) Trying to improve flatness
        2) Do not change CT Values
    """
    def __init__(self, sfInitialVolume):
        """
        Initializations:
            Use radius image
            Compute mask on initial volume by threshold and peel
            Use smapled-set for evaluation
            Use average of initial volume as reference
            
        After reconstruction with new tables:
            Use sample to compute score
            Return numeric score
        """
        
        self.RadiusImage = CreateRadiusImage()
        initialVolume = self.LoadVolume(sfInitialVolume)
        self.maskVolume = ComputeMask(initialVolume)
        initialVolume.DumpSimilarVolume(self.maskVolume, "PolyScorerMask")
        

    def LoadVolume(self, sVolumeFileName):
        vol = CVolume('flatnessVol', sVolumeFileName)
        vol.Print()
        #iImage = int(vol.nImages / 2)
        #im = vol.GetImage(iImage, 'Central_Image')
        #im.Print()
        #im.WriteToFile()
        return vol
    
def SortValues(phanthomImage,radiusImage):
    print('*** Sort Values')
    maxRadius = 1500
    sums = torch.zeros(maxRadius)
    count = torch.zeros(maxRadius)
    n = 0
    for y in range(512):
        if y > 0 and y % 100 == 0:
            print(f'{y}')
        for x in range(512):
            value = phanthomImage.pData[y,x]
            if value > -150 and value < 150:
                n += 1
                iRadius = int(radiusImage.pData[y,x])
                sums[iRadius] += value
                count[iRadius] += 1
                
    onesVector = torch.ones(maxRadius)
    count = torch.maximum(count,onesVector)
    average = sums.div(count)
    sfName = 'd:/Pylog/ValuePerRadius.csv'
    f = open(sfName,'w')
    f.write("radius, count, average\n")
    for i in range(256):
        f.write(f"{i}, {count[i]}, {average[i]}\n")
    f.close()
    print(f'File written: {sfName}')
    relevant = average[0:256]
    std = torch.std(relevant).item()
    print(f'{std=}')
    return std

"""
def ScoreFlatness(bNominal=False):
    print('*** Measure Flatness')
    radiusImage = CreateRadiusImage()
    #radiusImage.Show()
    
    if bNominal:
        phanthomImage = LoadVolume(sVolumeFileNameNominal)
    else:
        phanthomImage = LoadVolume(sVolumeFileNameAi)
    #phanthomImage.Show()
   
    std = SortValues(phanthomImage,radiusImage)
    return std, phanthomImage
    """

"""
def mainOld():
    score, im = ScoreFlatness(bNominal=True)
    print(f'{score=}')
    im.WriteToFile()
    """
    
def main():
    print('*** Test Poly Scorer Class')
    scorer = CPolyScorer(sVolumeFileNameNominal)
    print(f'{scorer=}')

if __name__ == '__main__':
    main()
