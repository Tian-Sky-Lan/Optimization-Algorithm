import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl

flag = 1;

class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high):
        self.dimension = dimension  
        self.time = time  
        self.size = size  
        self.bound = []  
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  
        self.v = np.zeros((self.size, self.dimension))  
        self.p_best = np.zeros((self.size, self.dimension))  
        self.g_best = np.zeros((1, self.dimension))[0]  

        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]  
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, x):
        
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        if flag == 1:
            y = -(2*x1**2 - 3*x2**2 - 4*x1 + 5*x2 + x3)
        if flag == 0:
            y = 2*x1**2 - 3*x2**2 - 4*x1 + 5*x2 + x3
        # print(y)
        return y

    def update(self, size):
        c1 = 2.0  
        c2 = 2.0
        w = 0.8  
        for i in range(size):
            
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            
            self.x[i] = self.x[i] + self.v[i]
            
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            
            if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) > self.fitness(self.g_best):
                self.g_best = self.x[i]

    def pso(self):
        best = []
        self.final_best = np.array([1, 2, 3])
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            print('the temp best position：{}'.format(self.final_best))
            temp = self.fitness(self.final_best)
            print('the temp best fitness：{}'.format(temp))
            best.append(temp)
        t = [i for i in range(self.time)]
        if flag == 1:
            for i,k in enumerate(best):
                best[i]=-best[i]

        plt.figure()
        plt.plot(t, best, color='blue', marker=".")
        plt.margins(0)
        plt.xlabel(u"iteration") 
        plt.ylabel(u"fitneess")  
        plt.title(u"PSO")  
        plt.savefig('pso1.jpg')


if __name__ == '__main__':
    time = 20
    size = 100
    dimension = 3
    v_low = -1
    v_high = 1
    low = [0,0,0]
    up = [15,15,15]
    pso = PSO(dimension, time, size, low, up, v_low, v_high)
    pso.pso()

