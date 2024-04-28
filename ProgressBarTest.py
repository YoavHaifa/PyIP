# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 00:37:05 2024

@author: USER
"""

import time
from ProgressBar import ProgressBar


def main():
    print('*** Test Progress Bar')
    
    items = range(30)
    for item in ProgressBar(items, prefix = 'Progress50:', suffix = 'Complete', length = 50):
        time.sleep(1)
    
    items = range(20)
    for item in ProgressBar(items, prefix = 'Progress30:', suffix = 'Complete', length = 30):
        time.sleep(1)

    print('--- TestFinished')

if __name__ == '__main__':
    main()
