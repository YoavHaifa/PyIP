# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 20:18:10 2021

@author: yoavb
"""
import numpy as np
import torch
import sys
from Utils import FPrivateName
from MyString import GetValue, GetValue2, SetValue, RemoveValue, AddDesc
from HandleImages import CImage
from ProgressBar import ProgressBar
from Externals import Polar2Cart

debug = 0


def Clip(minValue, value, maxValue):
    if value <= minValue:
        return minValue
    if value >= maxValue:
        return maxValue
    return value
    

class CVolume():
    """
    Class holding one volume of images
    Can read files with short integers or float values
    Width, height, hPad and vPad are inferred from file name
    Default shape of data in file is 512*512 short
    nLines and nCols include Padded lines and columns
    Internally data is always held as 32-bit float
    """
    def __init__(self, name, fileName):
        """
        Args:
            imagesFileName: name of file to read - values are short - 512*512*nImages
        """
        self.name = name;
        self.fName = fileName
        self.fPrivateName = FPrivateName(fileName)
        mat = GetValue2(fileName, '_mat', '_matrix')
        if mat > 0:    
            self.nLines = mat
            self.nCols = mat
        else:
            self.nLines = 512
            self.nCols = 512
            nCols1 =  GetValue(fileName, '_width')
            if nCols1 > 0:
                self.nCols = nCols1
            nLines1 =  GetValue(fileName, '_height')
            if nLines1 > 0:
                self.nLines = nLines1
        
        self.nhPad = GetValue(fileName, '_hPad')
        self.nvPad = GetValue(fileName, '_vPad')
        self.ReadImages()
        self.i = 0

    def ReadImages(self):
        if debug:
            print(f'<CVolume::ReadImages> Reading volume {self.fName}...')
        bSrcIsFloat = self.fName.find('.float.') > 0
        if bSrcIsFloat:
            images = np.memmap(self.fName, dtype='float32', mode='r').__array__()
        else:
            images = np.memmap(self.fName, dtype='int16', mode='r').__array__()
            
        pImages = torch.from_numpy(images.copy())
        pImages = pImages.view(-1,self.nLines,self.nCols)
        if bSrcIsFloat:
            self.pImages = pImages
        else:
            self.pImages = pImages.float()
            if self.fName.find('.raw') > 0:
                self.fName = self.fName.replace('.raw', '.float.rimg')

        self.nImages = self.pImages.size()[0]
        if debug:
            print('<CVol::ReadImages> nImages', self.nImages)
        
    def IsPadded(self):
        return self.nvPad > 0 or self.nhPad > 0

    def GetImage(self, iImage, sDesc):
        imageData = self.pImages[iImage].clone()
        image = CImage(self.nLines, self.nCols, pData=imageData, fName=self.fName, i=iImage)
        sImageName = self.fName
        sImageName = SetValue(sImageName, '_width', self.nCols)
        sImageName = SetValue(sImageName, '_height', self.nLines)
        sImageName = SetValue(sImageName, '_im', iImage+1);
        if not sDesc is None:
            sImageName = AddDesc(sImageName, sDesc)
        image.fName = sImageName
        if debug:
            print(f'<Volume::GetImage> {iImage} ==> {sImageName}')
        return image

    def GetImageUnPad(self, iImage, sDesc):
        if not self.IsPadded():
            return self.GetImage(iImage, sDesc)
        
        imageData = self.pImages[iImage,self.nvPad:self.nLines-self.nvPad,self.nhPad:self.nCols-self.nhPad].clone()
        image = CImage(self.nLines-2*self.nvPad, self.nCols-2*self.nhPad, pData=imageData, fName=self.fName, i=iImage)
        
        #print('<GetImageUnPad> self.fName', self.fName)
        sImageName = self.fName
        sImageName = RemoveValue(sImageName, '_hPad')
        sImageName = RemoveValue(sImageName, '_vPad')
        sImageName = SetValue(sImageName, '_width', self.nCols - 2 * self.nhPad)
        sImageName = SetValue(sImageName, '_height', self.nLines - 2 * self.nvPad)
        image.fName = SetValue(image.fName, '_im', iImage);
        if not sDesc is None:
            sImageName = AddDesc(sImageName, sDesc)
        image.fName = sImageName

        return image

    def GetImagePadded(self, iImage, sDesc):
        imageData = self.pImages[iImage].clone()
        image = CImage(self.nLines, self.nCols, pData=imageData, fName=self.fName, i=iImage)
        
        image.fName = self.fName
        image.fName = SetValue(image.fName, '_im', iImage);
        image.fName = AddDesc(image.fName, sDesc);
        
        image.nhPad = self.nhPad
        image.nvPad = self.nvPad
        return image
    
    def GetPaddedImages(self, iImage, n, sDesc):
        iFirst = iImage - int(n/2)
        aImages = []
        for i in range(n):
            iCur = Clip(0, iFirst + i, self.nImages - 1)
            aImages.append(self.GetImagePadded(iCur, sDesc))
        return aImages
    
    def OnInCompatible(self, other, sText):
        print('*** Input ERROR: Incompatible volumes ***')
        self.Print()
        other.Print()
        print(sText)
        print('Exiting on error...')
        sys.exit()
    
    def AssertCompatible(self, other):
        if self.nImages != other.nImages and other.nImages != 1:
            self.OnInCompatible(other, 'Different number of images')
        if self.nLines != other.nLines or self.nCols != other.nCols:
            self.OnInCompatible(other, 'Different image shape')

    def Print(self):
        print(f'CVolume {self.name} with {self.nImages} images')
        print(f'File name: {self.fName}')

    def IsEqual(self, other):
        return torch.equal(self.pImages, other.pImages)


def TestSave(task,vol):
    newName = task.GetTestImageNewDimName('TRY', vol.nLines, vol.nCols)
    
    with open (newName, 'wb') as file:
        items = list(range(0, vol.nImages))
        for item in ProgressBar(items, prefix = 'Saving:', suffix = 'Complete', length = 50):
            image = vol.GetImage(item)
            npimage = image.pData.numpy()
            file.write(npimage.tobytes())
    print('Image saved:', newName)
    
def TestExtractImage(task,vol):
    iImage = task.iTest
    image = vol.GetImageUnPad(iImage, 'SRC')
    print('<vol.GetImageUnPad>', image.__dict__)
    image.WriteToFilePath()
    if task.bPolar2Cart:
        Polar2Cart(image.fName)
    image = vol.GetImagePadded(iImage, 'SRCPAD')
    image.WriteToFilePath()
    upImage = image.CreateUnPaddedCopy()
    upImage.fName = upImage.fName.replace('SRCPAD', 'SRCUnPad')
    upImage.WriteToFilePath()
    
def TestVolume(fName):
    print('*** Test Volume ***')
    vol = CVolume('test', fName)
    vol.Print()
    iImage = int(vol.nImages / 2)
    im = vol.GetImage(iImage, 'Central Image')
    im.Print()
    im.Show()
    if vol.IsPadded():
        imUnPad = vol.GetImageUnPad(iImage,'Un Padded')
        imUnPad.Print()
        imUnPad.Show()
    
def main():
    print('Test Volume')
    #task = CTask()
    #vol = CVolume(task.GetTestImageFullName())
    #print(vol)
    #print(vol.__dict__)
    #TestSave(task,vol)
    #TestExtractImage(task,vol)
    #print(atoi('512')*2)
    #print('mat', mat)
    #CheckTaskTrainVolumes(task)
    #CheckTaskTestVolumes(task)
    TestVolume('D:\Data\Test\Volumes\SCANPLAN_7346_WideFOV_DFS\Offline_14_01_2021 - NoANR/cf483db6_0_Recon0.raw')
    TestVolume('D:\Data\Test\Volumes\Data with padding/Scan314_Rings200_Polar_width1152_height544_hPad64_vPad16.float.dat')

if __name__ == '__main__':
    main()

