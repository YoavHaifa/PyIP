# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:50:05 2024
Reading abd Manipulating Impulse Response Table
@author: yoavb
"""

from os import path
import csv
import torch

nRows = 192
nDets = 688
detMargins = 165

sTableDir = 'D:/PolyCalib/ImpulseResponseTab'

verbosity = 5

def SaveTable(table2Save, iTube, sName):
    sfName = f'Tube{iTube}_{sName}_width{nDets}_height{nRows}_zoom2.float.rtab'
    sfName = path.join(sTableDir, sfName)
        
    npTable = table2Save.numpy()
    #print('<SaveTable>', sfName)
    with open (sfName, 'wb') as file:
        file.write(npTable.tobytes())
    if verbosity > 1:
        print('Table Saved:', sfName)
        
def SaveTableWithMargins(table2Save, iTube, sName):
    table2Save[:,0:detMargins] = 0
    table2Save[:,-detMargins:] = 0
    SaveTable(table2Save, iTube, sName + 'Margined')

def ProcTable(iTube, iCol):
    maxTab = torch.zeros([nRows, nDets])
    sfName = path.join(sTableDir, f'Tube{iTube}_IR_grid.csv')
    print('\nReading Table:', sfName)
    with open(sfName) as file:
        heading = next(file).split(',')
        sName = heading[iCol].strip()
        print('Header:', heading)
        print('Creating:', sName)
        reader = csv.reader(file)
        count = 0
        iPrevRow = -1
        iPrevDet = -1
        prevMax = 0
        iPrevFilledRow = -1
        for row in reader:
            iRow = int(row[0])
            iDet = int(row[1])
            maxVal = float(row[iCol])
            count += 1
            if count <= 5:
                print(f'{iRow}, {iDet}: {maxVal}')
            
            maxTab[iRow,iDet] = maxVal
            
            # Interpolate between detectors
            if iRow == iPrevRow:
                n = iDet - iPrevDet - 1
                if n > 0:
                    nDeltas = n + 1
                    deltaVal = maxVal - prevMax
                    for i in range(n):
                        iFill = i + 1
                        maxTab[iRow,iPrevDet+iFill] = prevMax + deltaVal * iFill / nDeltas
            
            # On row end - interpolate missing rows
            if iDet == nDets - 1:
                if iRow > 0:
                    nRowsToFill = iRow - iPrevFilledRow - 1
                    if nRowsToFill > 0:
                        nDeltas = nRowsToFill + 1
                        for iF in range(nRowsToFill):
                            iFill = iF + 1
                            ratio = iFill / nDeltas
                            maxTab[iPrevFilledRow+iFill] = maxTab[iPrevFilledRow] * (1 - ratio) +  maxTab[iRow] * ratio
            
                
                iPrevFilledRow = iRow
            iPrevRow = iRow
            iPrevDet = iDet
            prevMax = maxVal
                
    SaveTable(maxTab, iTube, sName)
    SaveTableWithMargins(maxTab, iTube, sName)

def main():
    print('*** ===>>> Analyze impulse response')
    for iTab in range(2):
        for iCol in range(3):
            ProcTable(iTab,iCol+2)
    
if __name__ == '__main__':
    main()
