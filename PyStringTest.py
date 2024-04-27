# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:30:11 2024

@author: USER
"""

from PyString import GetValue, GetValue2, SetValue, RemoveValue

def CheckGetValue(s, key):
    value = GetValue(s, key)
    print(f'GetValue({s}, {key}) ==> {value}')

def CheckGetValue2(s, key1, key2):
    value = GetValue2(s, key1, key2)
    print(f'GetValue2({s}, {key1}, {key2}) ==> {value}')

def TestStringValues():
    print('*** Test String Values')
    s = 'dd0c04c3_0_Recon0_Polar_width1152_height544_hPad16_vPad64_hsmooth51_SRC_im28.float.rimg'
    print ('string', s)
    print ('get value _im', GetValue(s,'_im'))
    print ('set value _im 3', SetValue(s,'_im', 3))
    print ('set value _im 456', SetValue(s,'_im', 456))
    print ('set value _width 22', SetValue(s,'_width', 22))
    print ('Remove value _width', RemoveValue(s,'_width'))
    print ('Remove value _im', RemoveValue(s,'_im'))
    

def main():
    CheckGetValue('abcd_width55_bla.bla', '_width')
    CheckGetValue2('abcd_mat123_bla.bla', '_mat', '_matrix')
    CheckGetValue2('abcd_matrix1024_bla.bla', '_mat', '_matrix')
    CheckGetValue2('abcd_matrixa1024_bla.bla', '_mat', '_matrix')
    TestStringValues()

if __name__ == '__main__':
    main()
