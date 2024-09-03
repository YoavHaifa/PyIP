# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:32:28 2024
Mask Volume - flag all relevant voxels.
Used to create samples of relevant voxels.
@author: yoavb
"""

import torch


threshMin = 850
threshMax = 1150

class CMaskVolume:
    """
    """
    def __init__(self, volume):
        """
        """
        volumeShifted = volume.pImages + 1000
        mask = torch.where(volumeShifted >= threshMin, volumeShifted, 0)
        mask = torch.where(mask <= threshMax, mask, 0)
        volume.DumpSimilarVolume(mask, "maskAfterThreshold")
        mask[:,0,:] = 0
        mask[:,-1,:] = 0
        mask[:,:,0] = 0
        mask[:,:,-1] = 0
        volume.DumpSimilarVolume(mask, "maskAfterClip")
        self.mask = self.Peel(mask)
        
        print(f'<CMaskVolume::__init__> {mask.shape=}')

       
    def Peel(self, mask):
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

        
def main():
    print('*** Test Mask Volume')
    mask = CMaskVolume()
    print(f'{mask=}')
    

if __name__ == '__main__':
    main()
