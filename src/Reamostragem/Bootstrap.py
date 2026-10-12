import numpy as np
from MSE import MSE

class Bootstrap:
    def __init__(self, modelo, dados_x, dados_y, stat = MSE(),n):     #faz o bootstrap, por padrão, com o mse
        self.modelo = modelo
        self.dados_x = dados_x
        self.dados_y = dados_y
        self.stat = stat
        self.n = n
        self.arr = None

    def mostras(self):       #pega dados_x e dados_y e então faz a nova amostra
        le = len(self.dados_x)
        samples = np.random.choice(le,
                                    size=self.n,
                                    replace=True
                                    )
        samples_x = self.dados_x[samples]
        samples_y = self.dados_y[samples]
        return samples_x, samples_y         #de cada um, eu quero alguma estatistica em especifico

    def construir_array(self):
        arr = np.empty(self.n)
        for i in range(self.n):
            mostras_x,mostras_y = self.mostras(n=n)
            modelo1 = self.modelo()
            modelo1.train(mostras_x[:,1],mostras_y[:1])
            y_chap = modelo1.predict(mostras_x)
            arr[i] = self.stat(mostras_y[:,i],y_chap)
        self.arr = arr
        return arr

    def media(self):
        if self.arr == None:
            return self.construir_array()
        return np.mean(self.arr)

    def desv_padrao(self):
        if self.arr == None:
            return self.construir_array()
        return np.std(self.arr)

    def int_de_confianca(self, alpha = 0.05):
        if self.arr == None:
            self.construir_array()
        return np.percentile(self.arr, [100*alpha/2, 100*(1-alpha/2)])