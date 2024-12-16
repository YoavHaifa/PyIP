# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:47:24 2024
Using Recon as external function with tailored autograd
@author: yoavb
"""

import torch

import Config
from RunRecon import RunAiRecon, RunOriginalRecon, VeifyReconRunning
from CsvLog import gCsvLog

gLoss = 0
signedLoss = 0

iTube = 0
iRow = 70
iDet = 300

iImage = 163
iRad = 30

count = 0
tabLen = 0

class CExRecon(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, tabs, scorer, sample):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #ctx.save_for_backward(input)
        global gLoss, count, tabLen, signedLoss
        with torch.no_grad():
            count += 1
            RunAiRecon('TryPolyStep')
            if Config.dump & 16:
                Config.SaveLastBpOutput(count)
                
            loss = scorer.ComputeNewScoreOfVolume1(Config.sfVolumeAi, sample)

            dev = scorer.averagePerImageRing - scorer.targetAverage
            if Config.dump & 4:
                Config.WriteMatrixToFile(dev, f'DevMap_flatTab_save{count}')
                
            tabLen = len(tabs)
            if tabLen == 1:
                signedLoss = dev[iImage,iRad]
                gCsvLog.AddItem(signedLoss)
                loss = abs(signedLoss)
                if Config.debug & 16:
                    print(f'<forward> value {scorer.averagePerImageRing[iImage,iRad]}, target {scorer.targetAverage}')
                    print(f'<forward> dev {dev[iImage,iRad]}, loss {loss}')
            else:
                absDev = dev.abs()
                loss = absDev.mean()
        gLoss = loss
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        #input, = ctx.saved_tensors
        #return grad_output * 1.5 * (5 * input ** 2 - 1)
        print(f'<CExRecon::backward> dev {signedLoss.item()}')
        if Config.debug & 32:
            print('<CExRecon::backward>')
            print(f'On Entry {grad_output=}')
            print(f'{grad_output.shape=}')
        if Config.debug & 2:
            lossVal = gLoss.item()
            print(f'Loss {count}: {lossVal}')
        value = abs(signedLoss.item())
        #gCsvLog.AddItem(value)
        if tabLen == 1:
            tmp = torch.full([1], value)
        else:
            tmp = torch.full([2*192*688], value)
        return tmp, None, None


def main():
    print('*** Check CExRecon')
    VeifyReconRunning()
    Config.OnInitRun()
    tensor = torch.randn(5)
    print(f'{tensor=}')
    #func = CExFunc()
    ctx = None
    t2 = CExRecon.forward(ctx, tensor)
    print(f'{t2=}')
    t2b = CExRecon.backward(ctx, t2)
    print(f'{t2b=}')
    

if __name__ == '__main__':
    main()
