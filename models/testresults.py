from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np 
"""
compute FA MA and Kappa coefficient

"""
class testResult:
    def __init__(self, testImg, refImg):
        self.testImg = np.double(testImg)
        self.refImg  = refImg

    def FA_MA(self):

        self.Nc = np.shape(np.argwhere(self.refImg >0))[0]
        self.Nu = np.shape(np.argwhere(self.refImg ==0))[0]
        testImg = self.testImg.reshape(-1,1)
        refImg  = self.refImg.reshape(-1,1)
        self.N = np.shape(refImg)[0]
        nz_refImg = [0 for i in range(self.N)]
        nz_testImg = [0 for i in range(self.N)]
        for i in range(self.N):
            if refImg[i] == 0:
                nz_refImg[i] = 0
            else:
                nz_refImg[i] = 1
        for j in range(self.N):
            if testImg[j] == 0:
                nz_testImg[j] = 0
            else:
                nz_testImg[j] = 1
        nz_refImg  = np.array(nz_refImg)
        nz_testImg = np.array(nz_testImg)
        FA = np.shape(np.argwhere(nz_testImg * (1 - nz_refImg)) ==0)[0]
        MA = np.shape(np.argwhere(nz_refImg * (1 - nz_testImg)) ==0)[0]
        return FA, MA
    
    def Acc(self, test, ref):  
        ref = ref.reshape(-1,)
        length = len(test)
        a = 0
        for i in range(length):
            if test[i] == ref[i]:
                a = a+1
        Acc = a/length
        return Acc

    def KappaCoef(self, FA, MA):
        TP = self.Nc - MA 
        TN = self.Nu - FA
        PCC = 1 - (FA + MA)/self.N
        PRE = ((TP + FA)*self.Nc + (TN + MA)*self.Nu)/(self.N**2)
        delta_PCCPRE = (PCC - PRE)*100
        delta_1PRE  = (1 - PRE)*100
        kappa = delta_PCCPRE/delta_1PRE
        return kappa  