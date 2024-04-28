# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:32:57 2024

@author: USER
"""

from os import path

from Utils import GetDataDir
from ProgressBar import ProgressBar
from Volume import CVolume


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
    image = vol.GetImagePadded(iImage, 'SRCPAD')
    image.WriteToFilePath()
    upImage = image.CreateUnPaddedCopy()
    upImage.fName = upImage.fName.replace('SRCPAD', 'SRCUnPad')
    upImage.WriteToFilePath()
    
def TestVolume(fName):
    print('*** Test Volume', fName)
    vol = CVolume('test', fName)
    vol.Print()
    iImage = int(vol.nImages / 2)
    im = vol.GetImage(iImage, 'Central Image')
    im.Print()
    im.Show()
    im.WriteToFile()
    if vol.IsPadded():
        imUnPad = vol.GetImageUnPad(iImage,'Un Padded')
        imUnPad.Print()
        imUnPad.Show()

def TestVolumes():
    print('*** Test Volumes ***')
    sDataDir = GetDataDir('VolumeTest')
    sfNamesList = ['dd0c04c3_0_Recon0_dn2.raw']
    sfNamesList.append('10688773_0_Recon0_HR_width512_height512_im155_SRC__1a.float.rimg')
    for sfName in sfNamesList:
        sfPath = path.join(sDataDir, sfName)
        TestVolume(sfPath)

def main():
    print('Test Volume')
    TestVolumes()
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

if __name__ == '__main__':
    main()
