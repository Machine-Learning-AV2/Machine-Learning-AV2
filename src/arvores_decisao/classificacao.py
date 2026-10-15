import numpy as np

class ArvoresDecisaoClassificacao:
  def __init__(self,max_profundidade=5, min_amostras=2):
    self.max_profundidade = max_profundidade
    self.min_amostras = min_amostras
    self.raiz = None

  def _gini(self,y):
    if len(y) == 0:
      return 0.0
      
    _, contagens = np.unique(y, return_counts=True)
    probabilidades = contagens / len(y)

  def _melhor_divisao(self,X,y):
    n_amostras, n_caracteristicas = X.shape
    melhor_gini = self._gini(y)
    melhor_divisao = None

    for j in range(n_caracteristicas):
      valores = np.unique(X[:,j])
      limites = (valores[:-1]+valores[1:])/2
      
      for limite in limites:
        masacara_esquerda = X[:,j]<= limite
        y_esquerda = y[mascara_esquerda]
        y_direita = y[~mascara_esquerda]

        n_esquerda = len(y_esquerda)
        n_direita = len(y_direita)

        gini_esquerda = self._gini(y_esquerda)
        gini_direita = self._gini(y_direita)

        gini_ponderado = (
          (n_esquerda/n_amostras)*gini_esquerda
          +(n_direita/n_amostras)*gini_direita
        )

        if gini_ponderado<melhor_gini:
          melhor_gini = gini_ponderado
          melhor_divisao = (j,limite)

    return melhor_divisao

  def _construir_arvore(self, X, y, profunidade =0):
    classes, contagens = np.unique(y, return_counts=True)
    classe_majoritaria = classes[np.argmax(contagens)]

    if (len(classes)) ==1:
      or profunidade >=self.max_profunidade
      or len(y) < self.min_amostras):
      return {"folha":True, "Valor": classe_majoritaria}

    divisao = self._melhor_divisao(X,y)

    if divisao is None:
      return {"folha": True, "valor": classe_majoritaria}
    j,limite = divisao
    mascara = X[:,j] <= limite

    return {
      "folha": False,
      "carateristicas": j,
      "limite": limite,
      "esquerda": self._construir_arvore(
        X[mascara], y[mascara], profundidade+1
      ),
      "direita": self._cosntruir_arvore(
        X[~mascara], y[~mascara],profunidade+1
      )
    }

   def ajustar(self, X,y):
     X = np.asrray(X, dtype=float)
     y = np.asarray(y)

     if X.ndim!=2:
       raise ValueError("X deve ser uma matriz BI"

     if y.ndim!=1:
       raise ValueError("y deve ser um vetor UNI")

     if len(X)!= len(y):
       raise ValueError("X e y devem ter o mesmo número de exemplos")

     if len(y) == 0:
       raise ValueError("os dados de treinamento nao podem estar vazios")

     if X.shape[1] == 0:
       raise ValueError("X deve ter pelo menos uma caracteristica")

     self.raiz = self._construir_arvore(X,y)

     return self

  def _prever_amostra(self,x,no):
    if no["folha"]:
      return no["valor"]

    j = no["caracteristica"]

    if x[j]<=no["limite"]:
      return self._prever_amostra(x,no["esquerda"]

     return self._prever_amostra(x,no["direita"])


  def prever(self,X):
    if self.raiz is None:
      raise ValueError("A arvore de decisão precisa treinada antes de prever")

    X=np.asarray(X,dtype=float)

    if X.ndim!=2:
      raise ValueError("X deve ser uma matriz BI")

    previsoes = [
      self._prever_amostra(x,self.raiz)
      for x in X
    ]

  
 if __name__ == "__main__":
   X = np.array([1], [2], [3], [4])
   y = np.array("A", "A", "B", "B")

   modelo = ArvoreDecisaoClassificacao()
   modelo.ajustar(X,y)

   print("Classificação")
   print(modelo.prever(X))
