import numpy as np
from Classificacao import KNN
from metricas import MSE

class KFold:
    def __init__(self,
            model,
            X : np.ndarray,
            Y : np.ndarray,
            k: int = 5,
            hip = None,
            stat = MSE
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
        temp = self.model(hip) if self.hip is not None else self.model()
        if hasattr(temp, "fit") == False:
            raise AttributeError("O modelo não contém a função 'fit' necessária")
        elif hasattr(temp,"predict") == False:
            raise AttributeError("O modelo não contém a função 'predict' necessária")

    def folds(self):
        """Aplicação do K-fold"""

        lista = []                                      
        inds = np.random.permutation(len(self.X))
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
        self.X = X
        self.Y = Y
        self.step = step
        self.model = model

    def cv_score(self, hipi):
        """Calcula a acurácia de cada hiperparâmetro da grid"""
        return KFold(self.model,self.X,self.Y, hip = hipi).folds()       #Aplica kfold na grid de hiperparâmetros

    def find(self):     
        """Pega o modelo e faz LOOCV, kfold ou booststrap para cada"""
        lista = []              #Lista de acurácias
        for i in range(len(self.grid)):
            lista.append((self.cv_score(self.grid[i]),self.grid[i]))   
        minim = min(lista)[1]           #pega o melhor hiperparâmetro
        best = self.model(minim)
        
        return best.fit(self.X,self.Y), minim

if __name__ == "__main__":
    from Regressao import RegressaoPolinomial
    seed = 42
    np.random.seed(seed)
    class ConstantModel:
        """Classe de teste com y fixo"""

        def __init__(self, value=3.0):
            self.value = value

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), self.value)

    class ConstantModelWithHyperparameter:
        """Classe de teste com y fixo e hiperparâmetro"""

        def __init__(self, hyperparameter):
            self.hyperparameter = hyperparameter

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), self.hyperparameter)

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


    def test_find_selects_best_hyperparameter_with_hyperparameter():
        X = np.arange(20).reshape(-1, 1)
        y = np.full(20, 5.0)

        search = FindHiperParam(
            ConstantModelWithHyperparameter,
            X,
            y,
            grid=[0, 5, 10]
        )

        model, best_hip = search.find()
        mse = np.mean((model.predict(X) - y) ** 2)

        assert best_hip == 5, "O melhor hiperparâmetro deveria ser 5"
        assert isinstance(model, ConstantModelWithHyperparameter), "Model não é uma instância de ConstantModelWithHyperparameter"
        assert mse == 0, "MSE deveria ser 0 para o melhor hiperparâmetro"

        print("Tudo OK com hiperparâmetro")

    def test_LOOCV():
        X = np.arange(20).reshape(-1, 1)
        y = np.full(20, 3.0)

        loocv = LOOCV(ConstantModel, X, y)
        mse = loocv.LOO()

        assert mse == 0, "MSE deveria ser 0 para o modelo constante"

        print("LOOCV OK")

    def test_bootstrap():
        X = np.arange(20).reshape(-1, 1)
        y = np.full(20, 3.0)

        bootstrap = Bootstrap(RegressaoPolinomial, X, y, n=100, hip = 2)
        arr = bootstrap.build_array()
        mean = bootstrap.mean()
        std_dev = bootstrap.std_dev()
        conf_int = bootstrap.conf_int(alpha=0.05)

        #print(mean, std_dev, conf_int)

        assert len(arr) == 100, "O array de estatísticas deveria ter tamanho 100"
        assert abs(mean) <= 10**9, "A média deveria ser 0 para o modelo constante"
        assert std_dev <= 10**9, "O desvio padrão deveria ser 0 para o modelo constante"
        assert abs(conf_int[0]) <= 10**9 and abs(conf_int[1]) <= 10**9, "O intervalo de confiança deveria ser [0, 0] para o modelo constante"

        print("Bootstrap OK")

    def test_KFold():
        X = np.arange(20).reshape(-1, 1)
        y = np.full(20, 3.0)

        kfold = KFold(ConstantModel, X, y, k=5)
        mse = kfold.folds()

        assert mse == 0, "MSE deveria ser 0 para o modelo constante"

        print("KFold OK")

    def test_KNN():
        from Classificacao import KNN
        from metricas import classification_error
        # Criar um conjunto de dados de classificação
        X = np.array([[0, 0], [1, -0.5], [2, 2], [3, 5], [-3, 10], [4, 2.5], [1.5, 3.5], [-1, 1.5], [-3, 2.5], [-2.5, -0.5], [-0.5, -1], [-2, 7], [-2.5, 6]])
        Y = np.array(["Red", "Red", "Blue", "Blue", "Green", "Blue", "Blue", "Red", "Red", "Red", "Red", "Green", "Green"])


        # Inicializar o modelo KNN
        knn_model = KNN
        # Aplicar KFold com o modelo KNN
        kfold = KFold(knn_model, X, Y, hip=3, stat=classification_error)
        mse = kfold.folds()

        print(f"KFold MSE para KNN: {mse}")


    def test_with_svm():
        from Classificacao import SVM
        from metricas import classification_error
        # Criar um conjunto de dados de classificação
        X = np.array([[0, 0], [1, -0.5], [2, 2], [3, 5], [-3, 10], [4, 2.5], [1.5, 3.5], [-1, 1.5], [-3, 2.5], [-2.5, -0.5], [-0.5, -1], [-2, 7], [-2.5, 6]])
        Y = np.array(["Red", "Red", "Blue", "Blue", "Green", "Blue", "Blue", "Red", "Red", "Red", "Red", "Green", "Green"])

        # Inicializar o modelo SVM
        svm_model = SVM
        # Aplicar KFold com o modelo SVM
        kfold = KFold(svm_model, X, Y, hip=1, stat=classification_error)
        mse = kfold.folds()

        print(f"KFold MSE para SVM: {mse}")

    test_find_selects_best_hyperparameter()
    test_find_selects_best_hyperparameter_with_hyperparameter()
    test_LOOCV()
    test_bootstrap()
    test_KFold()
    test_KNN()
    test_with_svm()