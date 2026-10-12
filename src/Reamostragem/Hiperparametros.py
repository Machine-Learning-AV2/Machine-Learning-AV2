import numpy as np
from LOOCV import LOOCV
from K_fold import K_Fold
class FindHiperParam:
    def __init__(self,modelo, grid = [0.1,1,10,100,1000], min = 0.1, max = 1000, step = 10):
        self.grid = grid
        self.min = min
        self.max = max
        self.step = step
        self.modelo = modelo

    def find(self):     
        """Pega o modelo e faz LOOCV, kfold ou booststrap para cada"""
        pass