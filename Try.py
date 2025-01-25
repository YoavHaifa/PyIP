# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:28:19 2024
Just try some simple code examples
@author: yoav.bar
"""

import os
import torch
import random

sAiFlag = 'd:/Config/Poly/GetAiTable.txt'
sAiFlagRemoved = 'd:/Config/Poly/GetAiTable_x.txt'

def TryRename():
    #print('Try Rename')
    if os.path.exists(sAiFlag):
        os.rename(sAiFlag, sAiFlagRemoved)
    
def TryIndices():
    print('Try Indices')
    mat = torch.zeros([4,8])
    for iLine in range(4):
        for iCol in range(8):
            mat[iLine,iCol] = (iLine+1)*10 + iCol+1
    print(mat)
    lines = [1,2,0]
    cols = [3,5,4]
    selected = []
    for i in range(3):
        selected.append(mat[lines[i],cols[i]].item())
    print(selected)
    fast = mat[lines,cols]
    print(fast)
    
def tryRandom():
    print('Try Random')
    count = torch.zeros(11)
    for i in range(220):
        ir = random.randint(0,10)
        count[ir] += 1
    print(count)
    print(f'{count.sum()=}')
    
def AvoidSmallNumbers(t):
    minPos = 0.5
    bPos = t > 0
    bSmall = t < minPos
    print(f'{bSmall=}')
    bSmallPos = torch.logical_and(bPos, bSmall)
    print(f'{bSmallPos=}')
    t1 = torch.where(bSmallPos, minPos, t)
    bNeg = t1 < 0
    bSmall = t > -minPos
    bSmallNeg = torch.logical_and(bNeg, bSmall)
    t2 = torch.where(bSmallNeg, -minPos, t1)
    return t2

def TryAvoid():
    t = torch.randn(10)
    print(t)
    t = AvoidSmallNumbers(t)
    print(t)
    
def main():
    print('*** Just try')
    #TryRename()
    #TryIndices()
    #tryRandom()
    TryAvoid()
    print('Try Finished')


if __name__ == '__main__':
    main()
