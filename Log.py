# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:40:22 2024
Log class - log to file - with times for each line
@author: yoavb
"""

import Config
import datetime
import time
import sys

class CLog:
    """
    """
    def __init__(self, sfName):
        """
        

        Parameters
        ----------
        sfName : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfName = Config.LogFileName(sfName)
        self.count = 0
        self.blanks = ''
        self.sections = []
        self.times = []
        
    def __del__(self):
        print(f'<CLog::__del__> {self.sfName} logged {self.count} lines.')
        
    def Log(self, s):
        now = datetime.datetime.now()
        if self.count:
            sMode = 'a'
        else:
            sMode = 'w'
            
        with open(self.sfName, sMode) as f:
            f.write(f'{now}: {self.blanks} {s}\n')
            
        self.count += 1
        
    
    def Start(self, sSection, sComment=''):
        self.Log(f'Start {sSection} {sComment}')
        self.blanks = self.blanks + '  '
        self.sections.append(sSection)
        start = time.monotonic()
        self.times.append(start)

    def End(self, sSection, sComment=''):
        sTopSection = self.sections[-1]
        if sSection != sTopSection:
            sError = f'<CLog::End> error: {sSection=} != {sTopSection=}'
            self.Log(sError)
            print(sError)
            sys.exit()
    
        start = self.times[-1]
        elapsed = time.monotonic() - start
    
        self.Log(f'End {sSection} {sComment} - Elapsed {elapsed:.3f} seconds')
        if len(self.blanks) > 1:
            self.blanks = self.blanks[0:-2]
        self.sections.pop()
        self.times.pop()

gLog = CLog('GPolyTrainer')


def Log(s):
    if gLog:
        gLog.Log(s)

def Start(sSection, sComment=''):
    if gLog:
        gLog.Start(sSection, sComment)

def End(sSection, sComment=''):
    if gLog:
        gLog.End(sSection, sComment)


def main():
    Config.OnInitRun()
    print('*** Test Log')
    log = CLog('Test')
    print(f'{log.sfName=}')
    log.Log('First line')
    log.Log('Second line')
    time.sleep(0.22)
    log.Log('After sleeping 0.22 seconds')
    log.Start('section0', "Try Enter")
    log.Log('within section 0')
    log.End('section0')
    log.Log('Third line')

    log.Start('section1', "Try Enter")
    time.sleep(0.1)
    log.Log('Slept 0.1 within section 1')
    log.Start('section2', "within section 1")
    time.sleep(0.2)
    log.Log('Slept 0.2 within section 2')
    log.End('section2', 'comment on end')
    log.End('section1')

    log.Start('sectionWillErr', "Try Enter")
    time.sleep(1)
    log.Log('Slept 1 within section')
    log.End('sectionErred')
    
    
    
    
if __name__ == '__main__':
    main()
