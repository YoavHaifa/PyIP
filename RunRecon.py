# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 23:30:13 2024
Run Recon for Poly Calibration
@author: yoav.bar
"""

import subprocess
import psutil
import sys

import Config
from Log import Start, End

verbose = 1
count = 0

nToPrint = 3

sHSProg = 'D:/Software-Recon/Output/release/Bin/HostSimulator.exe'
#sHSProg = 'G:\Software-Recon-v1.3.1\Output\Debug\Bin\HostSimulator.exe'
#sParams = '-exec OfflineRecon -ScanId 80c39ec3 -shotnum 0 -data "E:\Data\Poly_Calib\CenteredBig250\Scan_Plan" -transfer "d:\ReconTest\Recon\TransPointer.txt" -output d:\ReconTest\Output -tlog f:\Log\ReconUT\\MultiTest_Debug_timing_0042.csv -test xx -reconparam ReconParams_e95faa2dd826c08ad72508dc4103ad3a.csv'
#sParams = '-exec OfflineRecon -ScanId 80c39ec3 -shotnum 0 -data E:\Data\Poly_Calib\CenteredBig250\Scan_Plan -output d:\PolyCalib -test xx -reconparam ReconParams_e95faa2dd826c08ad72508dc4103ad3a.csv'
lParams = ['-exec', 'OfflineRecon', 
           '-ScanId', '80c39ec3', 
           '-shotnum', '0', 
           '-data', 'D:/SpotlightScans', 
           '-output', 'd:/PolyCalib/ex', 
           '-test', 'xx', 
           '-reconparam', 
           'ReconParams_e95faa2dd826c08ad72508dc4103ad3a.csv']

def RunRecon(sWhy = None):
    global count
    count += 1
    Start('Recon')
    #args = [sHSProg, sParams]
    args = [sHSProg]
    args.extend(lParams)
    if verbose > 1 or count <= nToPrint:
        if sWhy is not None:
            print(f'*** Calling Recon {count} for {sWhy}')
        else:
            print(f'*** Calling Recon {count}')
    if verbose > 2:
        print(args)
    subprocess.run(args)
    End('Recon')
    VeifyReconRunning(bOnEnd = True)

def RunOriginalRecon(sWhy = None):
    Config.SetPolyNominal()
    RunRecon(sWhy)

def RunAiRecon(sWhy, bDefaultOutput = True):
    Start('RunAiRecon')
    Config.SetPolyByAi(bDefaultOutput)
    RunRecon(sWhy)
    End('RunAiRecon')
    
def VeifyReconRunning(bOnEnd = False):
    Start('VeifyReconRunning')
    bRunning = "AppRunner.exe" in (p.name() for p in psutil.process_iter())
    if not bRunning:
        if bOnEnd:
            print('AppRunner crushed!')
        else:   
            print('AppRunner is not running!')
        sys.exit()
    End('VeifyReconRunning')

def main():
    n = 1
    print('*** Verify that recon is running')
    VeifyReconRunning()
    print('*** Run Original Recon by Host Simulator Command Line')
    RunOriginalRecon('Test Original Recon')
    print('Nominal Recon Finished')
    if n > 1:
        RunAiRecon('test AI recon')
        print('AI Recon Finished')
 
if __name__ == '__main__':
    main()
