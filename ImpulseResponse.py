# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:48:55 2024
Create dataset for impulse-response
@author: yoav.bar
"""

import sys
import os
from os import path
import torch

import Config
from Utils import VerifyDir, VerifyJointDir
from RunRecon import RunAiRecon, VeifyReconRunning
from RadiusImage import CRadiusImage
from Volume import CVolume

matrix = 256
nRows = 192
nDets = 688

sBaseImpulseDir = 'D:/PolyCalib/Impulse'

iTube = 0 

bSingle = False
bBand = True

if bSingle:
    iFirstRow = 70
    iFirstDet = 300
    iDeltaRow = 10
    iDeltaDet = 10
    iLastRow = 70
    iLastDet = 300
    verbosity = 15
elif bBand:
    #rows
    iFirstRow = 67
    iDeltaRow = 1
    iLastRow = 74
    #dets
    iFirstDet = 170
    iDeltaDet = 1
    iLastDet = 386
    #debug
    verbosity = 15
else:
    # Do All
    iFirstRow = 0
    iFirstDet = 0
    iDeltaRow = 10
    iDeltaDet = 10
    iLastRow = 191
    iLastDet = 687
    verbosity = 15

count = 0
bias = -1000
debugAnalysis = False

#sfVol = 'd:/PolyCalib/Impulse/Poli_AI_t1_r50_d302_width256_height256_zoom2.float.rvol'

def AnalyzeIR(sfVolume, radIm, fCsv=None):
    global count
    vol = CVolume('scoredVol', sfVolume)
    minVol = vol.pImages.min().item()
    maxVolInitial = vol.pImages.max().item()
    newVol = vol.pImages - bias
        
    maxVol = newVol.max().item()
    if count < 4:
        print(f'{minVol=}, max {maxVolInitial} --> {maxVol}')
    nImages = vol.nImages
    iMaxIm = -1
    for iIm in range(nImages):
        imMax = newVol[iIm].max().item()
        if imMax == maxVol:
            if count < 4:
                print(f'Image {iIm} max {imMax}')
            iMaxIm = iIm

    # Compute fraction of image coordinate
    imageCoo = iMaxIm
    if iMaxIm > 0 and iMaxIm < nImages - 1:
        prevMax = newVol[iMaxIm-1].max().item()
        nextMax = newVol[iMaxIm+1].max().item()
        threshold = maxVol / 10
        if prevMax > threshold:
            fraction = prevMax / maxVol
            imageCoo = imageCoo - fraction
            if count < 4:
                print(f'prevMax {prevMax}, fraction {fraction} --> imageCoo {imageCoo}')
        if nextMax > threshold:
            fraction = nextMax / maxVol
            imageCoo = imageCoo + fraction
            if count < 4:
                print(f'nextMax {nextMax}, fraction {fraction} --> imageCoo {imageCoo}')
     
    # Compute radius
    imWithMax = newVol[iMaxIm]
    halfMax = maxVol / 2
    if debugAnalysis:
        maxInIm = imWithMax.max().item()
        print(f'DEBUG: {iMaxIm=}, {maxVol=}, {halfMax=}, {maxInIm=}')
        
    radAtHigh = torch.where(imWithMax > halfMax, radIm.image.pData + 1, 0)
    n = torch.count_nonzero(radAtHigh)
    if n > 0:
        rad = (radAtHigh.sum().item() / n) - 1
    else:
        rad = -1 
        print('<AnalyzeIR> ERROR: no value above half max in max image!')
    print(f'{sfVolume}: im {imageCoo:.3f} n high {n} - rad {rad:.3f}')
    if fCsv is not None:
        fCsv.write(f'{maxVol}, {imageCoo:.3f}, {rad:.3f}\n');
    
    

def CreateIR(iTube, radIm, sVolDir):
    global count
    print(f'*** === >>> <CreateIR> tube {iTube}')
    sfName = f'{sVolDir}/Tube{iTube}_IR_grid_r{iFirstRow}_d{iFirstDet}.csv'
    
    
    with open(sfName, 'w') as fCsv:
        fCsv.write('row, det, max, im, rad\n')
        
        for iRow in range(iFirstRow, iLastRow+1, iDeltaRow):
            print(f'{iRow=}')
            for iDet in range(iFirstDet, iLastDet+1, iDeltaDet):
                count += 1
                
                with open(Config.sfImpulseIndices,'w') as f:
                    f.write(f'{iTube} {iRow} {iDet}\n')
                    
                sfDump = f'Poli_AI_t{iTube}_r{iRow}_d{iDet}_width{matrix}_height{matrix}_zoom2.float.rvol'
                Config.sfVolumeAi = path.join(sVolDir, sfDump)
                RunAiRecon('CreateIR')
                fCsv.write(f'{iRow}, {iDet}, ');
                AnalyzeIR(Config.sfVolumeAi, radIm, fCsv)
                os.remove(Config.sfImpulseIndices)
                if iDet == nDets - 1:
                    break
            if iRow == nRows - 1:
                break

def CheckReadyVolume(sVolDir, radIm):
    print('\n*** ===>>> Check impulse response computations on a ready volume')
    sfVol = 'Poli_AI_t0_r70_d343_width256_height256_zoom2.float.rvol'
    sfVol = path.join(sVolDir, sfVol)
    if not path.isfile(sfVol):
        print('Missing file:', sfVol)
        sys/exit()
    print('Input file:', sfVol)
    AnalyzeIR(sfVol, radIm)

def main():
    print('*** ===>>> Test impulse response')
    bTestComputations = False
    
    VeifyReconRunning()
    
    VerifyDir(sBaseImpulseDir)
    sDir = f'Impulse_r{iFirstRow}_{iDeltaRow}_{iLastRow}_c{iFirstDet}_{iDeltaDet}_{iLastDet}'
    sVolDir = VerifyJointDir(sBaseImpulseDir, sDir)
    Config.SetSpecialVolDir(sVolDir)
    
    Config.OnInitRun()
    radIm = CRadiusImage()
    
    if bTestComputations:
        CheckReadyVolume(sVolDir, radIm)
        sys.exit()

    print('\n*** ===>>> RunRecon for impulse response')
    if iTube == 0:
        CreateIR(0, radIm, sVolDir)
    elif iTube == 1:
        CreateIR(1, radIm, sVolDir)
    else:
        CreateIR(0, radIm, sVolDir)
        CreateIR(1, radIm, sVolDir)
    
if __name__ == '__main__':
    main()
