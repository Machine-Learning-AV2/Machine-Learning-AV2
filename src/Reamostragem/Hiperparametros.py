import numpy as np
from LOOCV import LOOCV
from K_fold import K_Fold


"""Juntando as peças, o procedimento completo é este:

    separe o conjunto de teste e esqueça que ele existe;

    para cada valor candidato do hiperparâmetro e para cada fold da validação cruzada: 
    ajuste o pré-processador e o modelo na parte de treino do fold, e avalie na parte retida;
    
    tome a média sobre os folds, obtendo CV(λ)CV(λ), e escolha λ∗=arg⁡min⁡λCV(λ)λ∗=argminλ​CV(λ);
    
    reajuste o modelo com λ∗λ∗ usando o conjunto de treino inteiro;
    
    meça uma única vez no teste e reporte esse número."""


class FindHiperParam:
    def __init__(self,
            modelo, 
            grid = [0.1,1,10,100,1000],
            min = 0.1, max = 1000, 
            step = 10
        ):
        self.grid = grid
        self.min = min
        self.max = max
        self.step = step
        self.modelo = modelo

    def find(self):     
        """Pega o modelo e faz LOOCV, kfold ou booststrap para cada"""
        lista = []
        for i in range():
            pass