# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:46:22 2024
Deviation Raster - Deviation from target for each image and radius
@author: yoavb
"""

import Config
from Volume import CVolume
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from Sample import CSample
from PolyScorer import CPolyScorer
#from Log import Start, End, Log

maxRadius = 260
maxImages = 280

class CDevRaster:
    """
    """
    def __init__(self, targetLevel, averagePerImRad):
        """
        """
        self.dev = averagePerImRad - targetLevel
        


def main():
    print('*** Test Dev Raster')
    Config.SetSpecialVolDir('D:/PolyCalib/Volumns')
    Config.OnInitRun()
    radIm = CRadiusImage()
    vol = CVolume('nominalVol', Config.sfVolumeNominal)
    maskVol = CMaskVolume(vol)
    sample = CSample(maskVol, radIm)
    scorer = CPolyScorer()
    
    oldScore = scorer.OldScore(Config.sfVolumeNominal, sample)
    scorer.Log()
    print(f'{oldScore=}')
    
    oldScore = scorer.OldScore(Config.sfVolumeAi, sample)
    scorer.Log()
    print(f'{oldScore=}')

    score = scorer.ComputeNewScoreOfVolume(Config.sfVolumeAi, sample)
    scorer.Log()
    print(f'{score=}')
    #devRaster = CDevRaster()

if __name__ == '__main__':
    main()
