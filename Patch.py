# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:57:01 2024
C Patch - patches to be applied to poly tables
@author: yoavb
"""

import torch
import math

import Config

class CPatch:
    """
    Class Patch to add to Poly Table
    """
    def __init__(self, radius):
        """
        """
        self.radius = radius
        self.sName = f'Patch{radius}'
        self.diameter = radius * 2
        self.side = self.diameter + 2
        self.raster = torch.zeros([self.side, self.side])
        
        #Fill each pixel within radius distance from center with 1/distance
        centerX = radius - 0.5
        centerY = radius - 0.5
        
        for iLine in range(self.side):
            dy = centerY - iLine
            dy2 = dy * dy
            for iCol in range(self.side):
                dx = centerX - iCol
                dx2 = dx * dx
                distance = math.sqrt(dx2+dy2)
                if distance < radius:
                    self.raster[iLine, iCol] =1 - distance / radius

    def Dump(self):
        sfName = f'Patch_radius{self.radius}'
        Config.WriteMatrixToFile(self.raster, sfName)
        
        
def main():
    Config.OnInitRun()
    print('*** Test Patch')
    patch = CPatch(200)
    patch.Dump()
    patch = CPatch(10)
    patch.Dump()
    


if __name__ == '__main__':
    main()
