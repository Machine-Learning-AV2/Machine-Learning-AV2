import numpy as np
from MSE import MSE
class LOOCV:
    def __init__(self, modelo, dados_x, dados_y):
        self.modelo = modelo
        self.dados_x = dados_x
        self.dados_y = dados_y

    def LOO(self):
        lista = []
        for i in range(len(self.dados_y)):
            dados_xi = np.delete(self.dados_x,i)
            dados_yi = np.delete(self.dados_y,i)

            modelo_i = self.modelo
            modelo_i.train(dados_xi,dados_yi)

            y_chapi = modelo_i.predict(self.dados_x[i])      #i-ésimo ŷ de teste
            msei = MSE(y_chapi,self.dados_y[i])
            lista.append(msei)

        return np.mean(lista)
