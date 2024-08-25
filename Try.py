# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:28:19 2024
Just try some simple code examples
@author: yoav.bar
"""

import os

sAiFlag = 'd:\Config\Poly\GetAiTable.txt'
sAiFlagRemoved = 'd:\Config\Poly\GetAiTable_x.txt'

def TryRename():
    #print('Try Rename')
    if os.path.exists(sAiFlag):
        os.rename(sAiFlag, sAiFlagRemoved)
    

def main():
    print('*** Just try')
    TryRename();
    print('Try Finished')


if __name__ == '__main__':
    main()
