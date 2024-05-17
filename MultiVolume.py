# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 01:39:24 2021

@author: yoavb
"""

import os
from os import path
import sys
import random
import torch

#import Config
#from Task import CTask
from Volume import CVolume
from VoxelEnv import CVoxelEnv
#from Rings import CRings
#from PointsMap import CPointsMap
#from Histogram import CGapHistogram
#from Utils import GetTaskRoot

debug = 0
debugSel = 0
debugFlags = 1

sVolumeFileTypes = [".raw", ".dat", ".float.bin", ".rdat", ".plane", ".TImage"]

class CMultiVolume():
    """
    Class holding several representations of the same volume, with optional target volume
    """

    def __init__(self, sDir, bTargetRequired=False):
        """
        Args:
            sDir: root directory containing relevant data
                The root directory may contain files or directories (preferable one kind of them)
                Each file or directory is assumed to contain a representation of the volume
                Each representation is compiled into an object of CVolume class
                All but 'target' are in one array order by alphabetic order
                File or directory with target data should have the string "_target" in its name
                Files with the text '_ignore' are ignored
                Volume files are recognized by special types as stated in 'sVolumeFileTypes'
        """
        print(f'<CMultiVolume::__init__> sDir {sDir}')
        self.sDir = sDir
        self.bTargetRequired = bTargetRequired
        self.LoadVolumes()

        # Support for random selection of a voxel    
        self.maxSelectionsFromSameImage = 100
        self.nSelectedFromCurrentImage = self.maxSelectionsFromSameImage
        self.iImage = 0
        self.iLines = 0;
        self.iCol = 0
        self.debug = 0
        self.fDebug = None
        
        # normalization mode for voxel environment
        self.bNormalize = False
        self.bNormalizeAll = False
        self.iAvoidLines = 2
        
    def LoadVolumes(self):
        self.input = []
        self.nVols = 0
        self.nInLayersFromFirstInput = 1
        self.target = None
        self.fList = []

        self.ListVolumeFilesInDir(self.sDir)
        if debug & 8:
            print('files list:', self.fList)
        self.fList.sort()
        self.AddFilesFromList()
        self.Verify()
        self.vol0 = self.input[0]
        self.vol0Images = self.vol0.pImages
        
        self.nImages = self.vol0.nImages
        self.nLines = self.vol0.nLines
        self.nCols = self.vol0.nCols
        print(f'<CMultiVolume> self.nImages {self.nImages}')

    def SetNorm(self, nInLayersFromFirstInput):
        if debug:
            print('<SetNorm>')
        self.bNormalizeAll = False
        self.bNormalize = True
        self.nInLayersFromFirstInput = nInLayersFromFirstInput
        
    def SetNorm0(self):
        if debug:
            print('<SetNorm0>')
        self.bNormalizeAll = False
        
    def SetNoNorm(self):
        if debug:
            print('<SetNoNorm>')
        self.bNormalize = False
        self.bNormalizeAll = False
        
    def SetEnvSize(self, nLayersFrom1stInput, nInputChannels, nLinesOnEachSide, nColsOnEachSide):
        self.nLayersFrom1stInput = nLayersFrom1stInput
        self.nLayersOnEachSide = (int)((nLayersFrom1stInput - 1) / 2)
        self.nInputChannels = nInputChannels
        self.nLinesOnEachSide = nLinesOnEachSide
        self.nColsOnEachSide = nColsOnEachSide
        self.segLen = 2 * self.nColsOnEachSide + 1
        
    def StartDebug(self, f, debug=None):
        self.fDebug = f
        if debug is not None:
            self.debug = debug
            
    def IsVolumeFile(self, sfName):
        for sExt in sVolumeFileTypes: 
            if sfName.endswith(sExt):
                return True
        return False

    def ListVolumeFilesInDir(self, sDirName):
        if debug:
            print('<ListVolumeFilesInDir>', sDirName)
        fList = list(os.listdir(sDirName))
        for filename in fList:
            sFullName = path.join(sDirName, filename)
            if self.IsVolumeFile(filename):
                if path.exists(sFullName):
                    print("<ListVolumeFilesInDir> Adding", sFullName)
                    self.fList.append(sFullName)
            elif path.isdir(sFullName):
                self.ListVolumeFilesInDir(sFullName)
        
            
    def AddFilesFromList(self):
        for fName in self.fList:
            if fName.find('_ignore') >= 0:
                print('Ignoring:', fName)
            elif fName.find('_target') >= 0:
                if self.target is None:
                    if debug:
                        print('Adding Target:', fName)
                    self.target = CVolume('target', fName)
                else:
                    print('<CMultiVolume::AddFilesFromList>')
                    print('Input error: Only one target is allowed.')
                    print(f'Found {fName}')
                    print('Exiting...')
                    sys.exit()
            else:
                if debug:
                    print(f'Adding Source {self.nVols+1}:', fName)
                vol = CVolume('input', fName)
                self.input.append(vol)
                self.nVols = len(self.input)
                
    def HasTarget(self):
        return self.target is not None
                
    def Verify(self):
        """
        Make sure that
        1) There is input
        2) There is target (if required)
        3) All input and target volumes are of the same dimensions
        Returns
        -------
            None. Exit if any error

        """
        if self.nVols < 1:
            print('<CMultiVolume::Verify> Multi Volume is empty', self.sDir)
            sys.exit()
        if self.bTargetRequired and self.target is None:
            print('<CMultiVolume::Verify> No target volume found', self.sDir)
            sys.exit()
        for inVol in self.input[1:]:
            self.vol0.AssertCompatible(inVol)
        if self.target is not None:
            self.vol0.AssertCompatible(self.target)
        
    def SelectImageLine(self):
        if self.nSelectedFromCurrentImage >= self.maxSelectionsFromSameImage:
            self.iImage = random.randint(self.nLayersOnEachSide, self.nImages-1-self.nLayersOnEachSide)
            self.nSelectedFromCurrentImage = 1
        else:
            self.nSelectedFromCurrentImage += 1
            
        if self.iAvoidLines == 2:
            iMaxLine = int((self.nLines - 1 - self.nLinesOnEachSide)/2)-1
            self.iLine = random.randint(int(self.nLinesOnEachSide/2), iMaxLine)*2+1
        else:
            self.iLine = random.randint(self.nLinesOnEachSide, self.nLines - 1 - self.nLinesOnEachSide)
            if self.iAvoidLines > 2:
                while ((self.iLine % self.iAvoidLines) == 0):
                    self.iLine = random.randint(self.nLinesOnEachSide, self.nLines - 1 - self.nLinesOnEachSide)

    def SetRange(self):
        self.iFirstCol = self.iCol - self.nColsOnEachSide
        self.iFirstLine = self.iLine - self.nLinesOnEachSide
        self.iAfterLine = self.iLine + self.nLinesOnEachSide + 1
        self.x0 = self.vol0.pImages[self.iImage, self.iFirstLine:self.iAfterLine,self.iFirstCol:self.iFirstCol+self.segLen]
                
    def SelectRandomVoxel(self):
        self.SelectImageLine()
        
        self.iCol = random.randint(self.nColsOnEachSide, self.nCols - 1 - self.nColsOnEachSide)
        vol0Images = self.vol0.pImages
        image = vol0Images[self.iImage]
        gap = abs(image[self.iLine-1,self.iCol]-image[self.iLine+1,self.iCol])
        if debugSel:
            print(f'<SelectRandomVoxel> [{self.iImage},{self.iLine},{self.iCol}] - gap {gap:.6f}')
        if debugSel:
            print('Bingo!')
        self.SetRange()

        
    def SelectRealRandomVoxel(self):
        n = 0
        while 1:
            self.SelectRandomVoxel()
            if self.x0.min() + 0.001 < self.x0.max():
                return     
            n += 1
            if n == 10000:
                print(f'{n} times random choose null data!')
                sys.exit()
                
    def SelectByMap(self):
        n = 0
        while 1:
            if self.nSmallSteps >= self.maxSmallSteps:
                self.iInMap = random.randint(100, self.nPoints - 200 - 1)
                self.nSmallSteps = 0
            else:
                step = random.randint(0, 200) - 100
                if step == 0:
                    step = 1
                self.iInMap += step
                if self.iInMap < 100:
                    self.iInMap = 200 + step
                elif self.iInMap >= self.nPoints - 100:
                    self.iInMap = self.nPoints - 200 + step
                
            self.iImage = self.pointsMap[self.iInMap][0]
            self.iLine = self.pointsMap[self.iInMap][1]
            self.iCol = self.pointsMap[self.iInMap][2]
            if self.iImage >= self.nLayersOnEachSide and self.iImage < self.nImages - self.nLayersOnEachSide:
                if self.iLine >= self.nLinesOnEachSide and self.iLine < self.nLines - self.nLinesOnEachSide:
                    if self.iCol >= self.nColsOnEachSide and self.iCol < self.nCols - self.nColsOnEachSide:
                        self.SetRange()
                        return
            n += 1
            if n == 10000:
                print(f'{n} times select by map choose point out of range!')
                sys.exit()
        
    def GetInputPerImage(self, iImage):
        deb = False
        x = torch.zeros([self.nInputChannels, self.nLines, self.nCols])
        if deb:
            print('<GetInputPerImage>', iImage)
            print(f'{x.shape=}')
            print(f'{self.nLayersFrom1stInput} input layers from {self.vol0.fName}')


        for iLayer in range(self.nLayersFrom1stInput):
            iPage = max(0, iImage - self.nLayersOnEachSide + iLayer)
            iPage = min(self.nImages-1, iPage)
            if deb:
                print(f'{iLayer=}, {iPage=}')
            x[iLayer] = self.vol0Images[iPage]

        i = self.nLayersFrom1stInput
        if self.nInputChannels > i:
            for inp in self.input[1:]:
                if inp.nImages == 1:
                    x[i] = inp.pImages[0]
                else:
                    x[i] = inp.pImages[iImage]
                if deb:
                    print(f'Extra input {i} from {inp.fName}')
                i += 1
        return x
        
    def GetInputPerVoxel(self):
        deb = False
        x = torch.zeros([self.nInputChannels, 2*self.nLinesOnEachSide+1, self.segLen])
        if deb:
            print(f'<GetInputPerVoxel> [{self.iImage}, {self.iLine}, {self.iCol}]')
            print(f'{self.vol0Images.shape=}')
            print(f'{x.shape=}')
            print(f'{self.iFirstLine=}')
            print(f'{self.iAfterLine=}')
            print(f'{self.segLen=}')
        i = 0
        if self.nLayersFrom1stInput == 1:
            x[i] = self.x0
            i = 1
        else:
            for iLayer in range(self.nLayersFrom1stInput):
                if deb:
                    print(f'{iLayer=}')
                iPage = self.iImage - self.nLayersOnEachSide + iLayer
                if deb:
                    print(f'{iPage=}')
                x[i] = self.vol0Images[iPage, self.iFirstLine:self.iAfterLine,self.iFirstCol:self.iFirstCol+self.segLen]
                i += 1
                
        if self.nInputChannels > i:
            for inp in self.input[1:]:
                if inp.nImages == 1:
                    x[i] = inp.pImages[0, self.iFirstLine:self.iAfterLine,self.iFirstCol:self.iFirstCol+self.segLen]
                else:
                    x[i] = inp.pImages[self.iImage, self.iFirstLine:self.iAfterLine,self.iFirstCol:self.iFirstCol+self.segLen]
                i += 1

        if debugSel:
            print('<GetInputPerVoxel> x[1] - ', x[1])
            print(f'{self.bNormalize=}')

        y = self.target.pImages[self.iImage, self.iLine, self.iCol:self.iCol+1] # return tensor of length 1
        yInterp = self.vol0Images[self.iImage, self.iLine, self.iCol:self.iCol+1] # return tensor of length 1
        yPrev = self.vol0Images[self.iImage, self.iLine-1, self.iCol:self.iCol+1] # return tensor of length 1
        yNext = self.vol0Images[self.iImage, self.iLine+1, self.iCol:self.iCol+1] # return tensor of length 1
        
        ve = CVoxelEnv(self.iImage, self.iLine, self.iCol, x, y, yInterp, yPrev, yNext, self.bNormalize, self.bNormalizeAll, self.nInLayersFromFirstInput)
        if self.rings is not None:
            ve.bRing = self.rings.IsRing(self.iImage, self.iLine)
        return ve
    
    def GetItem(self):
        if self.nPoints > 0:
            self.SelectByMap()
        else:
            self.SelectRealRandomVoxel()
        return self.GetInputPerVoxel()
    
    def NInFiles4Model(self):
        return len(self.input)

    def GetNFiles(self):
        n = len(self.input)
        if self.target is not None:
            n += 1
        return n
    
    def Print(self):
        print('<CMultiVolume::Print>')
        print(f'sDir {self.sDir}, nImages {self.nImages}')
        print('Input:')
        for vol in self.input:
            vol.Print()
        if self.target:
            print('Target:')
            self.target.Print()
        else:
            print('No Target')
        print('---')

    def GetImages(self, iImage, nInputLayers, sDesc = 'SRC'):
        aImages = []
        
        nVolsLeft = self.nVols
        iVol = 0
        if nInputLayers > 1:
            nOnEachSide = (int)((nInputLayers - 1) / 2)
            iCur = iImage - nOnEachSide
            for i in range (nInputLayers):
                iAct = min(self.nImages-1, max(0, iCur-1))
                aImages.append(self.input[0].GetImageUnPad(iAct, sDesc))
                iCur += 1
                
            nVolsLeft -= 1
            iVol = 1
                
        if nVolsLeft > 0:
            for i in range(nVolsLeft):
                aImages.append(self.input[iVol].GetImageUnPad(iImage-1, sDesc))
                iVol += 1
            
        # Different solution for 3D
        #if self.modelObj.nInputChannels > 1 and len(aImages) == 1:
        #    aImages = self.testIntputVols[0].GetPaddedImages(iImage, self.modelObj.nInputChannels, sDesc)
        return aImages
    
    def GetTargetImage(self, iImage, sDesc = 'TARGET'):
        return self.target.GetImageUnPad(iImage, sDesc)

    def Print2File(self, f):
        f.write(f'<CMultiVolume::Print2File> {self.sDir}\n')
        for i, inp in enumerate(self.input):
            f.write(f'Input {i}: {inp.fName}\n')
        if self.target is not None:
            f.write(f'Target: {self.target.fName}\n')
            
    def IsEqual(self, other):
        n = self.nVols
        for i in range(n):
            if not self.input[i].IsEqual(other.input[i]):
                return False
        return True
    
    def SetSelectConsecutive(self):
        self.bSelectConsecutive = True
        self.iInMap = 0


