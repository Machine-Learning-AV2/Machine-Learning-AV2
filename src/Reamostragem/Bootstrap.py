import numpy as np
from MSE import MSE
#Falta padronizar as funções com as outras partes da biblioteca

class Bootstrap:
    def __init__(self,
            modelo,
            dados_x : np.ndarray,
            dados_y : np.ndarray,
            n: int = 1000,
            stat = MSE
        ):
        """
        O argumento "modelo" deve ser uma classe da bibliteca
        ou que aceita dados no mesmo padrão 
        stat = MSE: é a estatística que se quer calcular
        """
    
        self.modelo = modelo
        self.dados_x = dados_x
        self.dados_y = dados_y
        self.stat = stat
        self.n = n
        self.arr = None

        if hasattr(self.modelo(), "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(self.modelo(),"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def _mostras(self):
        """
        Função que seleciona as amostras de dados_x e dados_y para cada iteração do boostrap
        """

        le = len(self.dados_x)
        sample = np.random.choice(le,
                                    size=le,
                                    replace=True
                                    )
        
        sample_x = self.dados_x[sample]
        sample_y = self.dados_y[sample]

        return sample_x, sample_y

    def construir_array(self):
        """
        Performa o Bootstrap, retornando um array
        com a estatística desejada de cada iteração
        """

        arr = np.empty(self.n)

        for i in range(self.n):
            _mostras_x,_mostras_y = self.mostras()
            modelo1 = self.modelo()
            modelo1.fit(_mostras_x,_mostras_y)
            y_chap = modelo1.predict(_mostras_x)
            arr[i] = self.stat(_mostras_y,y_chap)

        self.arr = arr
        return arr

    def media(self):
        """
        Calcula a média do array obtido. 
        Se não existe, calcula o array primeiro
        """

        if self.arr is None:
            return np.mean(self.construir_array())
        return np.mean(self.arr)

    def desv_padrao(self):
        """
        Calcula o desvio padrão do array obtido. 
        Se não existe, calcula o array primeiro
        """
        if self.arr is None:
            return np.std(self.construir_array())
        return np.std(self.arr)

    def int_de_confianca(self, alpha = 0.05):
        """
        Calcula o intervalo de confiança do array obtido. 
        Se não existe, calcula o array primeiro
        """

        if self.arr is None:
            return np.percentile(self.construir_array(), [100*alpha/2, 100*(1-alpha/2)])
        return np.percentile(self.arr, [100*alpha/2, 100*(1-alpha/2)])