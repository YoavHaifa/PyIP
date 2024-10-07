# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:48:55 2024
Create dataset for impulse-response
@author: yoav.bar
"""

from os import path
import torch

import Config
from RunRecon import RunAiRecon, VeifyReconRunning
from RadiusImage import CRadiusImage
from Volume import CVolume

matrix = 256
nRows = 192
nDets = 688

iFirstRow = 0
iFirstDet = 0
iDeltaRow = 10
iDeltaDet = 10
nRunRows = 2 
nRunDets = 3

sfIndices = 'd:/Config/Poly/Impulse.txt'

sfVol = 'd:/PolyCalib/Impulse/Poli_AI_t1_r50_d302_width256_height256_zoom2.float.rvol'

def AnalyzeIR(sfVolume, radIm, fCsv):
    vol = CVolume('scoredVol', sfVolume)
    minVol = vol.pImages.min().item()
    maxVolInitial = vol.pImages.max().item()
    newVol = vol.pImages - minVol
        
    maxVol = newVol.max().item()
    print(f'{minVol=}, max {maxVolInitial} --> {maxVol}')
    nImages = vol.nImages
    iMaxIm = -1
    for iIm in range(nImages):
        imMax = newVol[iIm].max().item()
        if imMax == maxVol:
            print(f'Image {iIm} max {imMax}')
            iMaxIm = iIm
    
    im = newVol[iMaxIm]
    halfMax = maxVol / 2
    radAtHigh = torch.where(im > halfMax, radIm.image.pData, 0)
    n = torch.count_nonzero(radAtHigh)
    if n > 0:
        rad = radAtHigh.sum().item() / n
    else:
        rad = -1 
    print(f'{sfVolume}: {iMaxIm} - {rad:.3f}')
    fCsv.write(f'{maxVol}, {iMaxIm}, {rad:.3f}\n');
    
    

def CreateIR(iTube, radIm):
    print('*** === >>> <CreateIR> tube {iTube}')
    sfName = f'd:/Log/Tube{iTube}_IR_grid.csv'
    with open(sfName, 'w') as fCsv:
        fCsv.write('row, det, max, im, rad\n')
        
        for ir in range(nRunRows):
            iRow = iFirstRow + ir * iDeltaRow
            if iRow > nRows - 1:
                iRow = nRows - 1
            for iD in range(nRunDets):
                iDet = iFirstDet + iD * iDeltaDet
                if iDet > nDets - 1:
                    iDet = nDets - 1
                
                with open(sfIndices,'w') as f:
                    f.write(f'{iTube} {iRow} {iDet}\n')
                    
                sfDump = f'Poli_AI_t{iTube}_r{iRow}_d{iDet}_width{matrix}_height{matrix}_zoom2.float.rvol'
                Config.sfVolumeAi = path.join('d:/PolyCalib/Impulse', sfDump)
                RunAiRecon()
                fCsv.write(f'{iRow}, {iDet}, ');
                AnalyzeIR(Config.sfVolumeAi, radIm, fCsv)
                if iDet == nDets - 1:
                    break
            if iRow == nRows - 1:
                break

def main():
    print('*** ===>>> RunRecon for impulse response')
    VeifyReconRunning()
    Config.OnInitRun('d:/PolyCalib/Impulse')
    #CreateIR()
    radIm = CRadiusImage()
    #AnalyzeIR(sfVol, radIm)
    CreateIR(0, radIm)
    CreateIR(1, radIm)
    
if __name__ == '__main__':
    main()
