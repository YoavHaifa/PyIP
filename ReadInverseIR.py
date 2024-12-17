# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:29:49 2024
Read  inverted IR tables
@author: yoavb
"""

import sys
import numpy as np
import torch
from os import path

nImages = 280 
nRadiuses = 260 
nEffectiveRadiuses = 120

iImage = 163


sDir = 'D:/PolyCalib/InverseIRTables'


def ReadTable(sfName):
    sfFullPath = path.join(sDir, sfName)
    if not path.isfile(sfFullPath):
        print('Missing file', sfFullPath)
        sys.exit()

    table = np.memmap(sfFullPath, dtype='int16', mode='r').__array__()
    table = torch.from_numpy(table.copy())
    table = table.view(nImages,nRadiuses)
    print('Table read:', sfFullPath)
    return table

def FindImageSource(rowTab, colTab):
    nr = nRadiuses
    sfCsv = path.join(sDir, f'Image{iImage}_sources.csv')
    count = 0
    with open(sfCsv, 'w') as f:
        f.write('radius, row, col\n')
        for r in range(nr):
            row = rowTab[iImage,r]
            col = colTab[iImage,r]
            if col < 1:
                break
            f.write(f'{count}, {row}, {col}\n')
            count += 1
    print(f'<FindImageSource> (image {iImage}, nr {count}) prepared', sfCsv)

def main():
    print('*** Read inverted IR tables')
    colTab = ReadTable('InverseIR_Tube0a_col_per_image_and_radius_width260_height280_dzoom2.short.rtab')
    rowTab = ReadTable('InverseIR_Tube0a_row_per_image_and_radius_width260_height280_dzoom2.short.rtab')
    FindImageSource(rowTab, colTab)

if __name__ == '__main__':
    main()
