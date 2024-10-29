# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:39:50 2024
Identify ring and its source in the readings space
@author: yoavb
"""

from os import path

from Volume import CVolume
from InverseIR import CInverseIRTablesPerTube
from RunRecon import RunAiRecon, VeifyReconRunning
from Utils import VerifyDir


sConfigDir = 'd:/Config/Poly'
sDumpDir = 'D:/PolyCalib/Impulse'
sfVol = 'Poli_AI_t1_r70_d300_width256_height256_zoom2.float.rvol'

matrix = 256
nDets = 688
nRows = 192

verbosity = 1

def CheckIIR(iTube, sfVol, fCsv, bSecondHalf):
    sfName = path.join(sDumpDir, sfVol)
    vol = CVolume('irVol', sfName)
    if verbosity > 3:
        print(vol)
        vol.Print()
    
    iImage, iLine, iCol = vol.FindMaxPosition()
    radius = vol.findRadius(iLine, iCol)
    iRad = int(radius + 0.5)
    print(f'MAX {iImage=}, {iLine=}, {iCol=}, {radius=}, {iRad=}')
    
    tabs = CInverseIRTablesPerTube(iTube)
    print(tabs)
    
    iRow, iDet = tabs.Inverse(iImage, iRad)
    if bSecondHalf:
        iDet = nDets - iDet - 1
    print(f'<CheckInverse> [{iImage}, {iRad}] ==> [{iRow}, {iDet}]')
    fCsv.write(f'{iImage}, {iRad}, {iRow}, {iDet}, ')
    return iRow, iDet

def ApplyIR(iTube, iRow, iCol):
    VeifyReconRunning()
    print(f'<ApplyIR> {iTube}, {iRow}, {iCol}')
    sfName = path.join(sConfigDir, 'Impulse.txt')
    with open(sfName, 'w') as file:
        file.write(f'{iTube} {iRow} {iCol}\n')
    print(f'File {sfName} written')

    if matrix == 256:
        sZoom = '_zoom2'
    else:
        sZoom = ''
    sfDump = f'Poli_AI_t{iTube}_r{iRow}_d{iCol}_width{matrix}_height{matrix}{sZoom}.float.rvol'
    sfDumpName = path.join(sConfigDir, 'BPDumpFileName.txt')
    with open(sfDumpName, 'w') as file:
        file.write(f'{sDumpDir}/{sfDump}\n')
    print(f'File {sfDumpName} written')

    RunAiRecon(bDefaultOutput = False)
    return sfDump
    
def main():
    print('*** ===>>> Use inverse impulse response table on volume ring')
    VerifyDir(sDumpDir)
    iTube = 0 
    iRow = 130 
    firstDet = 150
    deltaDet = 25
    nDetsToCompute = 17
    
    sfCsv = f'd:/Log/CheckIIR_tube{iTube}_row{iRow}_det{firstDet}_dDet{deltaDet}.csv'
    with open(sfCsv, 'w') as fCsv:
    
        fCsv.write('row, det, image, radius, inv-row, inv-det, delta row, delta det\n')
        for iD in range(nDetsToCompute):
            iDet = firstDet + iD * deltaDet
            fCsv.write(f'{iRow}, {iDet}, ')
            sfVol = ApplyIR(iTube, iRow, iDet)
            if iDet >= nDets / 2:
                bSecondHalf = True
            else:
                bSecondHalf = False
            iInvRow, iInvDet = CheckIIR(iTube, sfVol, fCsv, bSecondHalf)
            dRow = iInvRow - iRow
            dDet = iInvDet - iDet
            fCsv.write(f'{dRow}, {dDet}\n')
            
    print('Results written in', sfCsv)
    
if __name__ == '__main__':
    main()



