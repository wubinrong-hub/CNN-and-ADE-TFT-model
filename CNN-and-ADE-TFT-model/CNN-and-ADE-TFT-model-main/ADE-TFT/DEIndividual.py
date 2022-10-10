import numpy as np
#import RNN
#import GRU
import sys
sys.path.append("C:/Users/Administrator/Desktop/ADE-TFT")
import tft1

class DEIndividual:

    '''
    individual of differential evolution algorithm
    '''

    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.

    def generate(self):
        '''
        generate a random chromsome for differential evolution algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
       # self.fitness = ObjFunction.GrieFunc(
       #     self.vardim, self.chrom, self.bound)
    #    print([self.vardim, self.chrom.astype('int'), self.bound])
       # self.fitness = RNN.MyLSTM(self.vardim, self.chrom.astype('int'), self.bound)
#        self.fitness = GRU.MyLSTM(self.vardim, self.chrom.astype('int'), self.bound) 
        self.fitness = tft1.MyTFT(self.vardim, self.chrom.astype('float'), self.bound)  #int
        #self.fitness = ga_objfunc.MyLSTM(self.vardim, self.chrom, self.bound) 