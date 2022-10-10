import pandas as pd
import time
import numpy as np
from DEIndividual import DEIndividual
import random
import copy
import matplotlib.pyplot as plt
import sys
sys.path.append("C:/Users/Administrator/Desktop/ADE-TFT")
import dataout
import math
class DifferentialEvolutionAlgorithm:

    '''
    The class for differential evolution algorithm
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        param: algorithm required parameters, it is a list which is consisting of [crossover rate CR, scaling factor F]
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = DEIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluate(self, x):
        '''
        evaluation of the population fitnesses
        '''
        x.calculateFitness()

    def solve(self):
        '''
        evolution process of differential evolution algorithm
        '''
        self.t = 0
        self.initialize()
        for i in range(0, self.sizepop):
            self.evaluate(self.population[i])
            self.fitness[i] = self.population[i].fitness
        best = np.min(self.fitness)
        bestIndex = np.argmin(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
#        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
#        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness                  
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            for i in range(0, self.sizepop):
                vi = self.mutationOperation(i)
                ui = self.crossoverOperation(i, vi)
                xi_next = self.selectionOperation(i, ui)
                self.population[i] = xi_next
            for i in range(0, self.sizepop):
                self.evaluate(self.population[i])
                self.fitness[i] = self.population[i].fitness
            best = np.min(self.fitness)
            bestIndex = np.argmin(self.fitness)
            if best < self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
#            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
#            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            self.params[1]=1/(1+math.exp(10*(self.t-0.5*self.MAXGEN)/self.MAXGEN));
                    
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print ("Optimal solution is:")
        print (self.best.chrom)
        self.RMSE=0
        self.MAE=0
        self.fitness,self.y = dataout.MyTFT(self.vardim, self.best.chrom.astype('float'), self.bound)  #int
        print(self.fitness,self.y)

    def selectionOperation(self, i, ui):
        '''
        selection operation for differential evolution algorithm
        '''
        xi_next = copy.deepcopy(self.population[i])
        xi_next.chrom = ui
        self.evaluate(xi_next)
        if xi_next.fitness < self.population[i].fitness:
            return xi_next
        else:
            return self.population[i]

    def crossoverOperation(self, i, vi):
        '''
        crossover operation for differential evolution algorithm
        '''
        k = np.random.randint(0, self.vardim - 1)
        ui = np.zeros(self.vardim)
        for j in range(0, self.vardim):
            pick = random.random()
            if pick < self.params[0] or j == k:
                ui[j] = vi[j]
            else:
                ui[j] = self.population[i].chrom[j]
        return ui

    def mutationOperation(self, i):
        '''
        mutation operation for differential evolution algorithm
        '''
        nums = list(range(0, self.sizepop))
        random.shuffle(nums)
        nums[0:3]
        a=nums[0]
        b=nums[1]
        c=nums[2]
#        a = np.random.randint(0, self.sizepop - 1)
#        while a == i:
#            a = np.random.randint(0, self.sizepop - 1)
#        b = np.random.randint(0, self.sizepop - 1)
#        while b == i or b == a:
#            b = np.random.randint(0, self.sizepop - 1)
#        c = np.random.randint(0, self.sizepop - 1)
#        while c == i or c == b or c == a:
#            c = np.random.randint(0, self.sizepop - 1)
        vi = self.population[c].chrom + self.params[1] * \
            (self.population[a].chrom - self.population[b].chrom)
        for j in range(0, self.vardim):
            if vi[j] < self.bound[0, j]:
                vi[j] = self.bound[0, j]
            if vi[j] > self.bound[1, j]:
                vi[j] = self.bound[1, j]
        return vi

    def printResult(self):
        '''
        plot the result of the differential evolution algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
      #  y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
       # plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration ")
        plt.ylabel("The MAPE value of validation set")
        plt.title("DE for the training process of TFT")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
   start_cr_a_fit_net = time.time()

  # bound = np.tile([[1], [10]], 3)
   bound1 = np.tile([[2], [12]], 1)   #Time STEPS
   bound2 = np.tile([[5], [20]], 1)   #BATCH_SIZE
   bound3 = np.tile([[2], [8]], 1)   #NUM_hidden layers
   bound4 = np.tile([[2], [8]], 1)  #NUM_Consecutive hidden layers
#   bound5 = np.tile([[0.00002], [0.2]], 1)  #LEARNING_RATE_INIT
   bound5 = np.tile([[0.001], [0.1]], 1)  #LEARNING_RATE_DECAY_STEPS
   bound6= np.tile([[1], [4]], 1)#NUM_heads
   bound=np.hstack((bound1,bound2,bound3,bound4,bound5,bound6))
   #dea = DifferentialEvolutionAlgorithm(15,20, bound,2, [0.2,  0.3]) #NP，,,genmax,CR,F
   dea = DifferentialEvolutionAlgorithm(15,6, bound,20, [0.2,  1]) #NP，,,genmax,CR,F
   dea.solve()
   end_cr_a_fit_net = time.time()-start_cr_a_fit_net
   print('Running time of creating and fitting the TFT network: %.2f Seconds' % (end_cr_a_fit_net))
'''
    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
'''