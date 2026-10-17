import numpy as np
from MSE import MSE
#Falta padronizar as funções com as outras partes da biblioteca

class LOOCV:
    def __init__(self, 
            modelo, 
            dados_x : np.ndarray, 
            dados_y : np.ndarray
        ):
        """Inicialização do método LOOCV"""
        self.modelo = modelo
        self.dados_x = dados_x
        self.dados_y = dados_y

        if hasattr(self.modelo(), "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(self.modelo(),"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def LOO(self):
        """Aplicação do LOOCV"""
        lista = []
        for i in range(len(self.dados_y)):
            dados_xi = np.delete(self.dados_x,i)
            dados_yi = np.delete(self.dados_y,i)

            modelo_i = self.modelo()
            modelo_i.fit(dados_xi,dados_yi)
            y_chapi = modelo_i.predict(self.dados_x[i])      #i-ésimo ŷ de teste
            msei = MSE(y_chapi,self.dados_y[i])
            lista.append(msei)

        return np.mean(lista)
