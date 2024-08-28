# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:30:07 2024
Random sample for scoring flatness
@author: yoavb
"""

import torch

maxRadius = 300
maxPerRadius = 100


class CSample:
    """
    Hold 
    """
    def __init__(self, maskVolume):
        """
        Parameters
        ----------
        maskVolume : TYPE
            DESCRIPTION.

        """
        self.n = 0
        self.nPerRadius = torch.zeros(maxRadius, dtype=torch.int16)
        
        
def main():
    sample = CSample()
    

if __name__ == '__main__':
    main()
