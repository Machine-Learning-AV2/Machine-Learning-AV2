import numpy as np
from MSE import MSE
#Falta padronizar as funções com as outras partes da biblioteca

class K_Fold:
    def __init__(self,
            modelo,
            dados_x,
            dados_y,
            k = 5
        ):
        """Inicialização do método Kfold"""
        self.modelo = modelo
        self.dados_x = dados_x
        self.dados_y = dados_y
        self.k = k

        if hasattr(self.modelo(), "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(self.modelo(),"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def folds(self):
        """Aplicação do K-fold"""
        lista = []
        inds = np.random.permutation(len(self.dados_x))
        
        folds = np.array_split(inds,self.k)
        for i in range(self.k):
            x_treino = self.dados_x[np.delete(inds,folds[i])]
            y_treino = self.dados_y[np.delete(inds,folds[i])]
            x_teste = self.dados_x[folds[i]]
            y_teste = self.dados_y[folds[i]]
            modelo_i = self.modelo()
            modelo_i.fit(x_treino,y_treino)
            y_chapi = modelo_i.predict(x_teste)
            mse_i = MSE(y_chapi,y_teste)
            lista.append(mse_i)
        return np.mean(lista)