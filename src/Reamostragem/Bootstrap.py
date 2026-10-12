import numpy as np
import copy
from MSE import MSE

class Bootstrap:
    def __init__(self, modelo, dados_x, dados_y, stat = MSE()):     #faz o bootstrap, por padrão, com o mse
        self.modelo = modelo
        self.dados_x = dados_x
        self.dados_y = dados_y
        self.stat = stat

    def mostras(self, n = 1):       #pega dados_x e dados_y e então faz a nova amostra
        le = len(self.dados_x)
        samples = np.random.choice(le,
                                    size=n,
                                    replace=True
                                    )
        samples_x = self.dados_x[samples]
        samples_y = self.dados_y[samples]
        return samples_x, samples_y         #de cada um, eu quero alguma estatistica em especifico

    def boots(self, n=1):
        arr = np.empty(n)
        for i in range(n):
            mostras_x,mostras_y = self.mostras(n=n)
            modelo1 = self.modelo()
            modelo1.train(mostras_x[:,1],mostras_y[:1])
            y_chap = modelo1.predict(mostras_x)
            arr[i] = self.stat(mostras_y[:,i],y_chap)
        return arr[i]

