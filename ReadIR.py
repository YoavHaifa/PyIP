# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:45:22 2024
Read Impulse Response Data
@author: yoavb
"""


import sys
from os import path
#import csv
import pandas as pd



sIRDir = 'D:/PolyCalib/Impulse/Impulse_r67_1_70_c170_1_346'
sfIR = 'Tube0_IR_grid_r67_d170.csv'

def ReadData(sfName):
    print('<ReadData>', sfName)
    if not path.isfile(sfName):
        print('Missing file:', sfName)
        sys.exit()
        
    df = pd.read_csv(sfName)

    # Display the DataFrame
    print(df)       
    
    print(f'{df.iloc[0]=}')
    print(f'{df.iloc[0,3]=}')
    val = df.iloc[0,3]
    print(f'{type(val)=}')
    val1 = val.item()
    print(f'{type(val1)=}')
    
    val0 = 0.1
    print(f'{type(val0)=}')
    """
    with open(sfName, mode ='r')as file:
      csvFile = csv.reader(file)
      i = 0
      for line in csvFile:
          i += 1 
          if i == 0:
              continue
          
          values = map(float, line)
          print(i, ':', values)
          if i == 10:
              break
          """

def main():
    print('*** Read Impulse Response Data')
    sfName = path.join(sIRDir, sfIR)
    ReadData(sfName)
    
if __name__ == '__main__':
    main()
