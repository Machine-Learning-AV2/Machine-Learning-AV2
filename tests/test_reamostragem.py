if __name__ == "__main__":
    from src.Reamostragem import KFold, LOOCV, Bootstrap, FindHiperParam
    from src.Regressao import RegressaoPolinomial
    import numpy as np
    seed = 46
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

    def test_with_logistic_regression():
        from src.Classificacao import Logistic_Regression
        from src.metricas import classification_error
        # Criar um conjunto de dados de classificação
        X = np.array([[0, 0], [1, -0.5], [2, 2], [3, 5], [-3, 10], [4, 2.5], [1.5, 3.5], [-1, 1.5], [-3, 2.5], [-2.5, -0.5], [-0.5, -1], [-2, 7], [-2.5, 6]])
        Y = np.array(["Red", "Red", "Blue", "Blue", "Green", "Blue", "Blue", "Red", "Red", "Red", "Red", "Green", "Green"])

        # Inicializar o modelo Logistic Regression
        logreg_model = Logistic_Regression
        # Aplicar KFold com o modelo Logistic Regression
        kfold = KFold(logreg_model, X, Y, hip=1, stat=classification_error)
        mse = kfold.folds()

        print(f"KFold MSE para Logistic Regression: {mse}")

    def test_with_LDA():
        from src.Classificacao import LDA
        from src.metricas import classification_error
        # Criar um conjunto de dados de classificação
        X = np.array([[0, 0], [1, -0.5], [2, 2], [3, 5], [-3, 10], [4, 2.5], [1.5, 3.5], [-1, 1.5], [-3, 2.5], [-2.5, -0.5], [-0.5, -1], [-2, 7], [-2.5, 6]])
        Y = np.array(["Red", "Red", "Blue", "Blue", "Green", "Blue", "Blue", "Red", "Red", "Red", "Red", "Green", "Green"])

        # Inicializar o modelo LDA
        lda_model = LDA
        # Aplicar KFold com o modelo LDA
        kfold = KFold(lda_model, X, Y, stat=classification_error)
        mse = kfold.folds()

        print(f"KFold MSE para LDA: {mse}")

    def test_with_QDA():
        from src.Classificacao import QDA
        from src.metricas import classification_error
        # Criar um conjunto de dados de classificação

        import numpy as np




        X,Y = generate_QDA_dataset()
        # Inicializar o modelo QDA
        qda_model = QDA
        # Aplicar KFold com o modelo QDA
        kfold = KFold(qda_model, X, Y,k=3, stat=classification_error)
        mse = kfold.folds()

        print(f"KFold MSE para QDA: {mse}")

    def generate_QDA_dataset():
        np.random.seed(42)

        n = 30

        # Three classes with different means and covariance matrices
        X_red = np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[2, 0.5], [0.5, 1]],
            size=n
        )

        X_blue = np.random.multivariate_normal(
            mean=[5, 5],
            cov=[[1, -0.3], [-0.3, 2]],
            size=n
        )

        X_green = np.random.multivariate_normal(
            mean=[0, 6],
            cov=[[1.5, 0.7], [0.7, 1]],
            size=n
        )

        X = np.vstack([X_red, X_blue, X_green])

        Y = np.array(
            ["Red"] * n +
            ["Blue"] * n +
            ["Green"] * n
        )

        # Shuffle the observations
        indices = np.random.permutation(len(Y))
        X = X[indices]
        Y = Y[indices]

        return X, Y

    def test_decisionTree():
        from src.decision_tree import DecisionTreeClassification, DecisionTreeRegression
        from src.Reamostragem import KFold
        from src.metricas import classification_error, MSE
        import numpy as np

        # Classification dataset
        X = np.array([
            [1.0, 1.0, 1.5],
            [1.5, 2.0, 1.0],
            [2.0, 1.0, 2.0],
            [7.0, 1.0, 7.0],
            [7.5, 2.0, 8.0],
            [8.0, 1.0, 7.5],
            [4.0, 7.0, 4.0],
            [4.5, 8.0, 5.0],
            [5.0, 7.0, 4.5]
        ])

        Y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        kfold_class = KFold(
            DecisionTreeClassification,
            X,
            Y,
            k=3,
            stat=classification_error
        )

        error_class = kfold_class.folds()
        print("Decision Tree Classification Error:", error_class)

        # Regression dataset
        X_reg = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 2.0],
            [1.0, 2.0, 2.0],
            [3.0, 2.0, 1.0],
            [2.0, 3.0, 1.0],
            [3.0, 3.0, 2.0]
        ])

        Y_reg = np.array([4.0, 5.0, 5.0, 7.0, 9.0, 10.0])

        kfold_reg = KFold(
            DecisionTreeRegression,
            X_reg,
            Y_reg,
            k=3,
            stat=MSE
        )

        error_reg = kfold_reg.folds()
        print("Decision Tree Regression MSE:", error_reg)

    test_find_selects_best_hyperparameter()
    test_find_selects_best_hyperparameter_with_hyperparameter()
    test_LOOCV()
    test_bootstrap()
    test_KFold()
    test_KNN()
    test_with_svm()
    test_with_logistic_regression()
    test_with_LDA()
    test_with_QDA()
    test_decisionTree()