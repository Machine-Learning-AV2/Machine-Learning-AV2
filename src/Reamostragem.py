import numpy as np
from .metricas import MSE

class KFold:
    def __init__(self,
            model,
            X : np.ndarray,
            Y : np.ndarray,
            k: int = 5,
            hip = None,
            stat = MSE,
            seed = None
    
        ):
        """Inicialização do método Kfold
        Parâmetros
        ----------
            model : classe do modelo a ser utilizado
            data_x : np.ndarray com os dados de entrada
            data_y : np.ndarray com os dados de saída
            k : número de folds (Default: 5)
            hip : hiperparâmetros do modelo. Assume que o modelo possui um hiperparâmetro como seu único argumento posicional (Default: None)"""
        self.model = model
        self.X = X
        self.Y = Y
        self.k = k
        self.hip = hip
        self.stat = stat
        self.seed = seed
        temp = self.model(hip) if self.hip is not None else self.model()
        if hasattr(temp, "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(temp,"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def folds(self):
        """Aplicação do K-fold"""

        lista = []             
        rng = np.random.default_rng(self.seed)                         
        inds = rng.permutation(len(self.X))
        folds = np.array_split(inds,self.k)             #Lista de acurácias, permutação dos índices e divisão em k folds

        for i in range(self.k):
            X_train = self.X[np.setdiff1d(inds, folds[i])] #Separa os dados em cada  fold, sendo que o fold i é o de teste e os demais são de treino
            Y_train = self.Y[np.setdiff1d(inds, folds[i])]
            X_teste = self.X[folds[i]]
            Y_teste = self.Y[folds[i]]

            modelo_i = self.model() if self.hip is None else self.model(self.hip)    #modelo com ou sem hiperparâmetros
            modelo_i.fit(X_train,Y_train)

            Y_hat_i = modelo_i.predict(X_teste)     #ŷ_i de cada fold
            mse_i = self.stat(Y_hat_i,Y_teste)
            lista.append(mse_i)

        return np.mean(lista)

class LOOCV:
    def __init__(self, 
            model, 
            X : np.ndarray, 
            Y : np.ndarray,
            hip = None
        ):
        """Inicialização do método LOOCV
        Parâmetros
        ----------
            model : classe do modelo a ser utilizado
            data_x : np.ndarray com os dados de entrada
            data_y : np.ndarray com os dados de saída"""

        self.model = model
        self.X = X
        self.Y = Y
        self.hip = hip

        if hasattr(self.model(hip), "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(self.model(hip),"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def LOO(self):
        """Aplicação do LOOCV"""

        lista = []
        for i in range(len(self.Y)):
            X_i = np.delete(self.X,i, axis=0)
            Y_i = np.delete(self.Y,i, axis=0)

            modelo_i = self.model() if self.hip is None else self.model(self.hip)    #modelo com ou sem hiperparâmetros
            modelo_i.fit(X_i,Y_i)
            Y_hat_i = modelo_i.predict(self.X[i])      #i-ésimo ŷ de teste
            #print(f"y_hat_i: {y_hat_i}, y_true: {self.data_y[i]}")
            msei = MSE(Y_hat_i,np.array([self.Y[i]]))
            lista.append(msei)

        return np.mean(lista)

class Bootstrap:
    def __init__(self,
            model,
            X : np.ndarray,
            Y : np.ndarray,
            n: int = 1000,
            stat = MSE,
            hip = None
        ):
        """
        O argumento "modelo" deve ser uma classe da bibliteca
        ou que aceita dados no mesmo padrão 
        Parâmetros
        ----------
            model : classe do modelo a ser utilizado
            data_x : np.ndarray com os dados de entrada
            data_y : np.ndarray com os dados de saída
            n : número de iterações do bootstrap (default: 1000)
            stat : estatística a ser calculada (default: MSE)"""
    
        self.model = model
        self.X = X
        self.Y = Y
        self.stat = stat
        self.n = n
        self.arr = None
        self.hip = hip
        if hasattr(self.model(hip), "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(self.model(hip),"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def _samples(self):
        """
        Função que seleciona as amostras de dados_x e dados_y para cada iteração do boostrap
        """

        le = len(self.X)
        sample = np.random.choice(le,
                                    size=le,
                                    replace=True
                                    )
        
        sample_x = self.X[sample]
        sample_y = self.Y[sample]

        return sample_x, sample_y

    def build_array(self):
        """
        Performa o Bootstrap, retornando um array
        com a estatística desejada de cada iteração
        """

        arr = np.empty(self.n)

        for i in range(self.n):
            _samples_x,_samples_y = self._samples()
            model1 = self.model(self.hip)
            model1.fit(_samples_x,_samples_y)
            y_chap = model1.predict(_samples_x)
            arr[i] = self.stat(_samples_y,y_chap)

        self.arr = arr
        return arr

    def mean(self):
        """
        Calcula a média do array obtido. 
        Se não existe, calcula o array primeiro
        """

        if self.arr is None:
            return np.mean(self.build_array())
        return np.mean(self.arr)

    def std_dev(self):
        """
        Calcula o desvio padrão do array obtido. 
        Se não existe, calcula o array primeiro
        """
        if self.arr is None:
            return np.std(self.build_array())
        return np.std(self.arr)

    def conf_int(self, alpha = 0.05):
        """
        Calcula o intervalo de confiança do array obtido. 
        Se não existe, calcula o array primeiro
        """

        if self.arr is None:
            return np.percentile(self.build_array(), [100*alpha/2, 100*(1-alpha/2)])
        return np.percentile(self.arr, [100*alpha/2, 100*(1-alpha/2)])


class FindHiperParam:
    def __init__(self,
            model, 
            X : np.ndarray,
            Y : np.ndarray,
            grid = [0.1,1,10,100,1000],
            step = 10,
            seed=None
        ):
        """
        Inicialização do método de busca de hiperparâmetros. A implementação assume 
        que o modelo possui apenas um hiperparâmetro e que este é dado pelo único argumento posicional.
        Parâmetros
        ----------
            model: classe do modelo a ser utilizado
            data_x: np.ndarray com os dados de entrada
            data_y: np.ndarray com os dados de saída
            grid: lista de valores de hiperparâmetros a serem testados (default: [0.1,1,10,100,1000])
            step: passo para a busca (default: 10)"""

        self.grid = grid
        self.X = X
        self.Y = Y
        self.step = step
        self.model = model
        self.seed = seed

    def cv_score(self, hipi):
        """Calcula a acurácia de cada hiperparâmetro da grid"""
        return KFold(self.model,self.X,self.Y, hip = hipi, seed=self.seed).folds()       #Aplica kfold na grid de hiperparâmetros

    def find(self):     
        """Pega o modelo e faz LOOCV, kfold ou booststrap para cada"""
        lista = []              #Lista de acurácias
        for i in range(len(self.grid)):
            lista.append((self.cv_score(self.grid[i]),self.grid[i]))   
        minim = min(lista)[1]           #pega o melhor hiperparâmetro
        best = self.model(minim)
        
        return best.fit(self.X,self.Y), minim
