# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 23:30:13 2024
Run Recon for Poly Calibration
@author: yoav.bar
"""

import subprocess

verbose = 1

sHSProg = 'G:\Software-Recon-v1.3.1\Output\Debug\Bin\HostSimulator.exe'
#sParams = '-exec OfflineRecon -ScanId 80c39ec3 -shotnum 0 -data "E:\Data\Poly_Calib\CenteredBig250\Scan_Plan" -transfer "d:\ReconTest\Recon\TransPointer.txt" -output d:\ReconTest\Output -tlog f:\Log\ReconUT\\MultiTest_Debug_timing_0042.csv -test xx -reconparam ReconParams_e95faa2dd826c08ad72508dc4103ad3a.csv'
#sParams = '-exec OfflineRecon -ScanId 80c39ec3 -shotnum 0 -data E:\Data\Poly_Calib\CenteredBig250\Scan_Plan -output d:\PolyCalib -test xx -reconparam ReconParams_e95faa2dd826c08ad72508dc4103ad3a.csv'
lParams = ['-exec', 'OfflineRecon', 
           '-ScanId', '80c39ec3', 
           '-shotnum', '0', 
           '-data', 'D:\SpotlightScans', 
           '-output', 'd:\PolyCalib\ex', 
           '-test', 'xx', 
           '-reconparam', 
           'ReconParams_e95faa2dd826c08ad72508dc4103ad3a.csv']

def RunRecon():
    #args = [sHSProg, sParams]
    args = [sHSProg]
    args.extend(lParams)
    print('*** Calling Recon')
    if verbose > 2:
        print(args)
    subprocess.run(args)
    

def main():
    print('*** Run Recon by Host Simulator Command Line')
    RunRecon();
    print('Recon Finished')
 
if __name__ == '__main__':
    main()
