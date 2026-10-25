import numpy as np
from metricas import MSE

class KFold:
    def __init__(self,
            model,
            data_x,
            data_y,
            k = 5,
            hip = None
        ):
        """Inicialização do método Kfold
        Parâmetros
        ----------
        model: classe do modelo a ser utilizado
        data_x: np.ndarray com os dados de entrada
        data_y: np.ndarray com os dados de saída
        k: número de folds (Default: 5)
        hip: hiperparâmetros do modelo. Assume que o modelo possui um hiperparâmetro como seu único argumento posicional (Default: None)"""

        self.model = model
        self.data_x = data_x
        self.data_y = data_y
        self.k = k
        self.hip = hip
        temp = self.model()
        if hasattr(temp, "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(temp,"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def folds(self):
        """Aplicação do K-fold"""

        lista = []                                      
        inds = np.random.permutation(len(self.data_x))
        folds = np.array_split(inds,self.k)             #Lista de acurácias, permutação dos índices e divisão em k folds

        for i in range(self.k):
            x_train = self.data_x[np.setdiff1d(inds, folds[i])] #Separa os dados em cada  fold, sendo que o fold i é o de teste e os demais são de treino
            y_train = self.data_y[np.setdiff1d(inds, folds[i])]
            x_teste = self.data_x[folds[i]]
            y_teste = self.data_y[folds[i]]

            modelo_i = self.model() if self.hip is None else self.model(self.hip)    #modelo com ou sem hiperparâmetros
            modelo_i.fit(x_train,y_train)

            y_chapi = modelo_i.predict(x_teste)     #ŷ_i de cada fold
            mse_i = MSE(y_chapi,y_teste)
            lista.append(mse_i)

        return np.mean(lista)

class LOOCV:
    def __init__(self, 
            model, 
            data_x : np.ndarray, 
            data_y : np.ndarray
        ):
        """Inicialização do método LOOCV
        Parâmetros
        ----------
        model: classe do modelo a ser utilizado
        data_x: np.ndarray com os dados de entrada
        data_y: np.ndarray com os dados de saída"""
        self.model = model
        self.data_x = data_x
        self.data_y = data_y

        if hasattr(self.model(), "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(self.model(),"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def LOO(self):
        """Aplicação do LOOCV"""
        lista = []
        for i in range(len(self.data_y)):
            data_xi = np.delete(self.data_x,i, axis=0)
            data_yi = np.delete(self.data_y,i, axis=0)

            modelo_i = self.model()
            modelo_i.fit(data_xi,data_yi)
            y_hat_i = modelo_i.predict(self.data_x[i])      #i-ésimo ŷ de teste
            msei = MSE(y_hat_i,self.data_y[i])
            lista.append(msei)

        return np.mean(lista)

class Bootstrap:
    def __init__(self,
            model,
            data_x : np.ndarray,
            data_y : np.ndarray,
            n: int = 1000,
            stat = MSE
        ):
        """
        O argumento "modelo" deve ser uma classe da bibliteca
        ou que aceita dados no mesmo padrão 
        Parâmetros
        ----------
        model: classe do modelo a ser utilizado
        data_x: np.ndarray com os dados de entrada
        data_y: np.ndarray com os dados de saída
        n: número de iterações do bootstrap (default: 1000)
        stat: estatística a ser calculada (default: MSE)"""
    
        self.model = model
        self.data_x = data_x
        self.data_y = data_y
        self.stat = stat
        self.n = n
        self.arr = None

        if hasattr(self.model(), "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(self.model(),"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def _samples(self):
        """
        Função que seleciona as amostras de dados_x e dados_y para cada iteração do boostrap
        """

        le = len(self.data_x)
        sample = np.random.choice(le,
                                    size=le,
                                    replace=True
                                    )
        
        sample_x = self.data_x[sample]
        sample_y = self.data_y[sample]

        return sample_x, sample_y

    def build_array(self):
        """
        Performa o Bootstrap, retornando um array
        com a estatística desejada de cada iteração
        """

        arr = np.empty(self.n)

        for i in range(self.n):
            _samples_x,_samples_y = self._samples()
            model1 = self.model()
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
            data_x : np.ndarray,
            data_y : np.ndarray,
            grid = [0.1,1,10,100,1000],
            step = 10
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
        self.data_x = data_x
        self.data_y = data_y
        self.step = step
        self.model = model

    def cv_score(self, hipi):
        """Calcula a acurácia de cada hiperparâmetro da grid"""
        return KFold(self.model,self.data_x,self.data_y, hip = hipi).folds()       #Aplica kfold na grid de hiperparâmetros

    def find(self):     
        """Pega o modelo e faz LOOCV, kfold ou booststrap para cada"""
        lista = []              #Lista de acurácias
        for i in range(len(self.grid)):
            lista.append((self.cv_score(self.grid[i]),self.grid[i]))   
        minim = min(lista)[1]           #pega o melhor hiperparâmetro
        best = self.model(minim)
        
        return best.fit(self.data_x,self.data_y), minim

if __name__ == "__main__":
    class ConstantModel:
        """Classe de teste com y fixo"""

        def __init__(self, value=0):
            self.value = value

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), self.value)


    def test_find_selects_best_hyperparameter():
        X = np.arange(20).reshape(-1, 1)
        y = np.full(20, 2.0)

        search = FindHiperParam(
            ConstantModel,
            X,
            y,
            grid=[0, 2, 4]
        )

        model, best_hip = search.find()
        mse = np.mean((model.predict(X) - y) ** 2)

        assert best_hip == 2, "O melhor hiperparâmetro deveria ser 2"
        assert isinstance(model, ConstantModel), "Model não é uma instância de ConstantModel"
        assert mse == 0, "MSE deveria ser 0 para o melhor hiperparâmetro"

        print("Tudo OK")



    test_find_selects_best_hyperparameter()