# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:16:28 2024
Information for a single point in the table
@author: yoavb
"""


import sys
from os import path
#import csv
import pandas as pd


sIRDir = 'D:/PolyCalib/Impulse/Impulse_r67_1_70_c170_1_346'
sfIR = 'Tube0_IR_grid_r67_d170.csv'

class CTabValue:
    """
    Single value in the poly table
    """
    def __init__(self, iTube, csvIRLine):
        """
        """
        print(f'{csvIRLine=}')
        self.iTube = iTube
        self.row = csvIRLine['row'].item()
        self.det = csvIRLine[' det'].item()
        self.iIm = int(csvIRLine[' im'].item())
        self.iRad = int(csvIRLine[' rad'].item())
        
        self.prevVal = 1.0 
        
        
    def Print(self):
        print(f'CTabValue: [{self.row},{self.det}] --> [{self.iIm}, {self.iRad}]')


def ReadData(sfName):
    print('<ReadData>', sfName)
    if not path.isfile(sfName):
        print('Missing file:', sfName)
        sys.exit()
        
    df = pd.read_csv(sfName)

    # Display the DataFrame
    print(df)  
    return df     

def main():
    print('*** Read Impulse Response Data')
    sfName = path.join(sIRDir, sfIR)
    df = ReadData(sfName)
    tabVal = CTabValue(0, df.iloc[3])
    tabVal.Print()
    
if __name__ == '__main__':
    main()
