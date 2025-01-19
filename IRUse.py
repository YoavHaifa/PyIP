# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:07:34 2025
Impulse Response Usage - read tables and answer queries
@author: yoavb
"""

from os import path
import numpy as np
import torch

import Config

nRows = 192 
nDets = 688 
margins = 170

sDir = 'D:/PolyCalib/ImpulseResponseTab'
sIRRIm = 'Tube1_imMargined_width688_height192_zoom2.float.rtab'
sIRRRad = 'Tube1_radMargined_width688_height192_zoom2.float.rtab'

class CIRUse:
    """
    """
    def __init__(self, iTab):
        """
        """
        self.iTab = iTab
        self.imTab = self.LoadTable('im')
        self.radTab = self.LoadTable('rad')
        
        
    def LoadTable(self, sType):
        sfName = f'Tube{self.iTab}_{sType}Margined_width{nDets}_height{nRows}_zoom2.float.rtab'
        sfFullName = path.join(sDir, sfName)

        npTab = np.memmap(sfFullName, dtype='float32', mode='r').__array__()
           
        tab = torch.from_numpy(npTab.copy())
        tab = tab.view(nRows, nDets)
        print(f'<LoadTable> Loaded: {sfFullName}')
        return tab
    
    def GetTraget(self, iRow, iDet):
        iIm = self.imTab[iRow, iDet]
        iRad = self.radTab[iRow, iDet]
        return iIm, iRad
    
    def GetTragetI(self, iRow, iDet):
        iIm = int(self.imTab[iRow, iDet])
        iRad = int(self.radTab[iRow, iDet])
        return iIm, iRad

gIR1 = CIRUse(1)

def Check(iRow, iDet):
    iIm, iRad = gIR1.GetTragetI(iRow, iDet)
    print(f'<Check> [{iRow}, {iDet}] --> [{iIm}, {iRad}]')

def main():
    Config.OnInitRun()
    print('*** Test Impulse Response Usage')
    Check(30, 300)
    Check(31, 340)
    Check(90, 400)
    Check(180, 344)

if __name__ == '__main__':
    main()
