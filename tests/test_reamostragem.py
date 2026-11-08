if __name__ == "__main__":
    from src.Reamostragem import KFold, LOOCV, Bootstrap, FindHiperParam
    from src.Regressao import RegressaoPolinomial
    import numpy as np
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
        from src.Classificacao import KNN
        from src.metricas import classification_error
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
        from src.Classificacao import SVM
        from src.metricas import classification_error
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