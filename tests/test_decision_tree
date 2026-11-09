if __name__ == "__main__":

    # Teste de classificação
    X_train = np.array([
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

    y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    X_test = np.array([
        [1.8, 1.5, 1.5],
        [7.8, 1.5, 7.5],
        [4.5, 7.5, 4.5]
    ])

    y_test = np.array([0, 1, 2])

    modelo = DecisionTreeClassification()
    modelo.adjust(X_train, y_train)

    previsoes = modelo.predict(X_test)

    print("Classificação")
    print("Previsões:", previsoes)
    print("Classes reais:", y_test)
    print("Acurácia:", np.mean(previsoes == y_test))


    # Teste de regressão
    X_train_reg = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 2.0],
        [1.0, 2.0, 2.0],
        [3.0, 2.0, 1.0],
        [2.0, 3.0, 1.0],
        [3.0, 3.0, 2.0]
    ])

    y_train_reg = np.array([4.0, 5.0, 5.0, 7.0, 9.0, 10.0])

    X_test_reg = np.array([
        [1.5, 1.5, 1.5],
        [2.5, 2.0, 1.5]
    ])

    y_test_reg = np.array([4.5, 7.0])

    modelo_reg = DecisionTreeRegression()
    modelo_reg.adjust(X_train_reg, y_train_reg)

    previsoes_reg = modelo_reg.predict(X_test_reg)

    print("\nRegressão")
    print("Previsões:", previsoes_reg)
    print("Valores reais:", y_test_reg)
    print("Erro absoluto médio:",
          np.mean(np.abs(previsoes_reg - y_test_reg)))
