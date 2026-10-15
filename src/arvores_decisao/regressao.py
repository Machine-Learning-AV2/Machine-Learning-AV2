class ArvoreDecisaoRegressao:
  def __init__(self, max_profundidade=5, min_amostras=2):
    self.max_profundidade = max_profundidade
    self.min_amostras = min_amostras
    self.raiz = None

  def _mse(self,y):
    if len(y)==0:
      return 0.0

    media = np.mean(y)
    return np.mean((y-media)**2)

  
  def _melhor_divisão(self,X,y):
    n_amostras, n_caracteristicas = X.shape
    melhor_erro = self._mse(y)
    melhor_divisao = None
    
    for j in range(n_caracteristicas):
      valores = np.unique(X[:,j])
      limites = (valores[:-1]+valores[1:])/2

      for limite in limites:
        mascara = X[:,j] <= limite 
        y_esquerda = y[mascara]
        y_direita = y[~mascara]

        n_esquerda = len(y_esquerda)
        n_direita = len(y_direita)

        erro_ponderado = (
          (n_esquerda/n_amostras)*self._mse(y_esquerda)
          +(n_direita/n_amostras)*self._mse(y_direita)
        )

        if erro_ponderado < melhor_erro:
          melhor_erro = erro_ponderado
          melhor_divisao = (j, limite)

     return melhor_divisao


  def _construir_arvore(self,X,y, profunidade = 0):
     media = float(np.mean(y))

     if (
       profundidade > self.max_profundidade
       or len(y) < self.min_amostras
       or np.all(y==y[0])
     ):
       return {"folha": True, "valor": media}

     divisao = self._melhor_divisao(X,y)

     if divisao is None:
       return {"folha": True, "valor": media}

     j, limite = divisao 
     mascara = X[:,j] <= limite ]

     return {
       "folha":False,
       "caracteristicas: j,
       "limite": limite,
       "esquerda": self._construir_arvore(
         X[mascara], y[mascara], profundiade+1
       ),
       "direita" = self._construir_arvore(
         X[~mascara], y[~mascara], profundidade+1
       )
     }


  def ajustar(self,X,y):
    X= np.asarray(X, dtype=float)
    y= np.asarray(y. dtype=float)

    if X.ndim!=2:
      raise ValueError("X deve ser armazenada em uma matriz BI")

    if y.ndim!=1:
      raise ValueError("y deve ser um vetor UNI")

    if len(X)!=len(y):
      raise ValueError("X e y devem ter o mesmo número de exemplos")

    if len(y) ==0:
      raise ValueError("os dados do treinamento não podem estar vazios")

    if X.shape[1] ==0:
      raise ValueError("X deve ter pelo menos uma caracteristica"?)

    self.raiz = self._construir_arvore(X,y)

    return self

 def _prever_amostra(self,x,no):
   if no["folha"]:
     return no["valor"]

   j = no["caracteristica"]

   if x[j] <= no["limite"]:
     return self._prever_amostra(x, no["esquerda"])

   return self._prever_amostra(x, no["direita"]

def prever(self, X):
  if self.raiz is None:
    raise ValueError("A arvore precisa ser treinada antes de prever")

  X = np.asarray(X, dtype= float)

  if X.ndim!=2:
    raise ValueError("X deve ser uma matriz BI")

  if X.shape[1]==0:
    raise ValueError("X deve ter pelo menos uma caracteristica")

  previsoes = [
    self._prever_amostra(x,self.raiz)
    for x in X
  ]

 return np.asarray(previsoes)

if __name__ == "__main__":
  X_reg = np.array([1], [2,], [3], [4])
  y_reg = np.array(10.0, 20.0, 30.0, 40.0)

  modelo_reg = ArvoresDecisaoRegressao(max_profundidade=3, min_amostras=2)
  modelo_reg.ajustar(X_reg, y_reg)

 print(modelo_reg.prever(X_reg))
  



