Decision Trees — Classificação e Regressão

Implementação de árvores de decisão para problemas de classificação e regressão utilizando Python e NumPy.

O projeto implementa dois modelos:

* DecisionTreeClassification: árvore de decisão para classificação, utilizando o índice de Gini.
* DecisionTreeRegression: árvore de decisão para regressão, utilizando o erro quadrático médio (MSE).

Ambos os modelos utilizam divisões binárias e construção recursiva da árvore.


1. Dependências

O projeto utiliza:

* Python 3
* NumPy

Para instalar a dependência:

pip install numpy

Importação:

import numpy as np


2. Fundamentos matemáticos

2.1. Representação dos dados

Os métodos de treinamento recebem dois objetos:

Matriz de características (X)

É uma matriz bidimensional de dimensões (n \times p), em que:

* (n) representa o número de observações;
* (p) representa o número de características;
* (X_{ij}) representa o valor da característica (j) na observação (i).

Cada linha representa uma observação, enquanto cada coluna representa uma característica.

Vetor de respostas (y)

É um vetor unidimensional de tamanho (n), contendo a resposta associada a cada observação.

Na classificação, os valores representam classes. Na regressão, representam valores numéricos.

2.2. Divisões binárias

Cada nó interno da árvore contém uma regra de decisão baseada em uma única característica:

$$
X_j \leq t,
$$

em que:

* (j) é o índice da característica utilizada;
* (t) é o limite numérico escolhido pelo algoritmo.

A divisão produz dois subconjuntos:

$$
D_L = {(x_i,y_i):x_{ij}\leq t},
$$

$$
D_R = {(x_i,y_i):x_{ij}>t}.
$$

As observações que satisfazem a condição seguem para o ramo esquerdo. As demais seguem para o ramo direito.

2.3. Escolha dos limites candidatos

Para cada característica, o algoritmo identifica seus valores distintos e os organiza em ordem crescente:

$$
v_1 < v_2 < \cdots < v_k.
$$

Os limites candidatos são os pontos médios entre valores consecutivos:

$$
t_i = \frac{v_i+v_{i+1}}{2}.
$$

Dessa forma, são avaliados:

$$
t_1,t_2,\ldots,t_{k-1}.
$$

Essa estratégia considera todas as divisões distintas induzidas pela ordenação dos valores observados de uma característica numérica.



3. Árvore de decisão para classificação

3.1. Classe DecisionTreeClassification

Implementa uma árvore de decisão para problemas em que a variável resposta representa uma classe categórica.

O critério utilizado para escolher as divisões é o índice de Gini. O objetivo é produzir subconjuntos progressivamente mais homogêneos em relação às classes.

3.1.1. Construtor __init__

def __init__(self, max_depth=5, min_samples=2):

Inicializa o modelo e define os parâmetros responsáveis por controlar o crescimento da árvore.

Parâmetro	Padrão	Descrição
max_depth	5	Profundidade máxima permitida para a árvore.
min_samples	2	Número mínimo de observações necessário para tentar dividir um nó.

Atributos:

* self.max_depth: armazena a profundidade máxima.
* self.min_samples: armazena o limite mínimo de observações.
* self.root: armazena a raiz da árvore. Inicialmente, recebe None.
* self.n_caracteristicas: inicializado com None, mas não utilizado no restante da implementação.

3.1.2. Método _gini

def _gini(self, y):

Calcula a impureza de Gini de um conjunto de respostas.

Sua definição matemática é:

$$
G(D)=1-\sum_{c=1}^{K}p_c^2,
$$

em que:

* (K) é o número de classes distintas;
* (p_c) é a proporção de observações pertencentes à classe (c).

Quanto menor o índice, mais homogêneo é o conjunto.

Interpretação:

* (G(D)=0): todas as observações pertencem à mesma classe.
* Valores maiores indicam maior diversidade de classes.
* A impureza máxima ocorre quando todas as classes possuem a mesma proporção, sendo igual a:

$$
G_{\max}=1-\frac{1}{K}.
$$

Funcionamento:

1. Verifica se o vetor está vazio. Caso esteja, retorna 0.0.
2. Identifica as classes e suas frequências utilizando np.unique.
3. Calcula a proporção de cada classe.
4. Calcula o índice de Gini.

Parâmetro:

* y: vetor contendo as classes das observações.

Retorno: número real correspondente à impureza de Gini.

3.1.3. Método _best_division

def _best_division(self, X, y):

Procura a melhor divisão binária para um conjunto de observações.

O critério utilizado é a impureza de Gini ponderada dos subconjuntos resultantes.

Considere um conjunto (D), dividido em (D_L) e (D_R).

A impureza ponderada é:

$$
G_{\text{divisão}} =
\frac{|D_L|}{|D|}G(D_L)
+
\frac{|D_R|}{|D|}G(D_R).
$$

O algoritmo deseja minimizar essa quantidade.

Funcionamento:

1. Obtém o número de observações e características.
2. Inicializa a melhor impureza com o Gini do conjunto original.
3. Percorre todas as características.
4. Calcula os limites candidatos.
5. Para cada limite:
    * Separa as observações em dois subconjuntos.
    * Calcula o Gini de cada subconjunto.
    * Calcula a impureza ponderada.
6. Atualiza a melhor divisão caso encontre uma impureza estritamente menor.

Parâmetros:

* X: matriz de características.
* y: vetor de classes correspondente.

Retorno:

* Uma tupla (j, limit), contendo a característica e o limite escolhidos.
* None, caso nenhuma divisão produza redução estrita na impureza.

3.1.4. Método _build_tree

def _build_tree(self, X, y, depth=0):

Constrói a árvore recursivamente.

Primeiramente, identifica as classes presentes e suas frequências:

classes, counting = np.unique(y, return_counts=True)
class_majority = classes[np.argmax(counting)]

A classe majoritária é utilizada como previsão caso o nó se torne uma folha.

Condições de parada:

O algoritmo interrompe a construção quando pelo menos uma das condições seguintes é satisfeita:

1. Todas as observações pertencem à mesma classe:

len(classes) == 1

2. A profundidade máxima foi atingida:

depth >= self.max_depth

3. O número de observações é insuficiente:

len(y) < self.min_samples

Quando alguma condição é satisfeita, retorna:

{"folha": True, "valor": class_majority}

Caso contrário, procura a melhor divisão por meio de _best_division.

Se nenhuma divisão válida for encontrada, o nó também se torna uma folha.

Quando uma divisão é encontrada, os dados são separados e o método é chamado recursivamente para os dois subconjuntos.

A estrutura de um nó interno é:

{
    "folha": False,
    "caracteristica": j,
    "limite": limit,
    "esquerda": ...,
    "direita": ...
}

Parâmetros:

* X: matriz de características.
* y: vetor de classes.
* depth: profundidade atual do nó, iniciada em zero.

Retorno: dicionário que representa uma subárvore.

3.1.5. Método adjust

def adjust(self, X, y):

Treina o modelo utilizando os dados fornecidos.

Parâmetros:

* X: matriz bidimensional de características.
* y: vetor unidimensional de classes.

Validações:

Condição	Comportamento
X não é bidimensional	Lança ValueError.
y não é unidimensional	Lança ValueError.
Quantidades de observações diferentes	Lança ValueError.
y vazio	Lança ValueError.
Nenhuma característica em X	Lança ValueError.

A matriz X é convertida para float, enquanto y é convertido para um array NumPy sem imposição de tipo numérico.

Após as verificações, a árvore é construída:

self.root = self._build_tree(X, y)

Retorno: o próprio objeto (self), permitindo encadeamento de chamadas.

3.1.6. Método _predict_sample

def _predict_sample(self, x, no):

Realiza a previsão de uma única observação percorrendo a árvore.

Parâmetros:

* x: vetor contendo as características da observação.
* no: nó atual da árvore.

Funcionamento:

1. Verifica se o nó atual é uma folha.
2. Se for uma folha, retorna o valor armazenado.
3. Caso contrário, obtém a característica e o limite da divisão.
4. Verifica se:

$$
x_j \leq t.
$$

5. Se a condição for verdadeira, percorre o ramo esquerdo.
6. Caso contrário, percorre o ramo direito.

O processo termina quando uma folha é encontrada.

Retorno: classe prevista para a observação.

3.1.7. Método predict

def predict(self, X):

Realiza previsões para um conjunto de observações.

Parâmetro:

* X: matriz bidimensional contendo as observações que serão classificadas.

Funcionamento:

1. Verifica se o modelo já foi treinado.
2. Converte X para um array NumPy de tipo float.
3. Verifica se a matriz é bidimensional.
4. Percorre cada observação e aplica _predict_sample.
5. Armazena os resultados em um array NumPy.

Retorno: array unidimensional contendo as classes previstas.

Exceções:

* ValueError: quando a árvore ainda não foi treinada.
* ValueError: quando X não é bidimensional.

⸻

4. Árvore de decisão para regressão

4.1. Classe DecisionTreeRegression

Implementa uma árvore de decisão para problemas em que a variável resposta é numérica.

O critério utilizado para escolher as divisões é o erro quadrático médio (MSE).

O objetivo é produzir subconjuntos cujas respostas apresentem baixa variabilidade.

4.1.1. Construtor __init__

def __init__(self, max_depth=5, min_samples=2):

Inicializa o modelo de regressão.

Parâmetro	Padrão	Descrição
max_depth	5	Profundidade máxima permitida para a árvore.
min_samples	2	Número mínimo de observações necessário para tentar dividir um nó.

Atributos:

* self.max_depth: profundidade máxima.
* self.min_samples: número mínimo de observações para tentar uma divisão.
* self.root: raiz da árvore, inicialmente definida como None.

4.1.2. Método _mse

def _mse(self, y):

Calcula o erro quadrático médio em relação à média dos valores de resposta.

Sua definição é:

$$
MSE(D)=\frac{1}{n}
\sum_{i=1}^{n}(y_i-\bar y)^2,
$$

em que:

$$
\bar y=\frac{1}{n}\sum_{i=1}^{n}y_i.
$$

A média é utilizada porque é o valor constante que minimiza a soma dos erros quadráticos:

$$
\bar y =
\underset{a\in\mathbb R}{\arg\min}
\sum_{i=1}^{n}(y_i-a)^2.
$$

Assim, caso um conjunto seja representado por uma única previsão constante, a melhor escolha sob perda quadrática é sua média.

Funcionamento:

1. Verifica se o vetor está vazio. Caso esteja, retorna 0.0.
2. Calcula a média de y.
3. Calcula a média dos quadrados dos desvios em relação à média.

Parâmetro:

* y: vetor de valores numéricos.

Retorno: número real correspondente ao erro quadrático médio.

4.1.3. Método _best_division

def _best_division(self, X, y):

Procura a divisão que minimiza o MSE ponderado dos subconjuntos resultantes.

Para uma divisão em dois grupos:

$$
MSE_{\text{divisão}} =
\frac{|D_L|}{|D|}MSE(D_L)
+
\frac{|D_R|}{|D|}MSE(D_R).
$$

Funcionamento:

1. Obtém o número de observações e características.
2. Inicializa o melhor erro com o MSE do conjunto original.
3. Percorre todas as características.
4. Identifica os valores distintos de cada característica.
5. Calcula os pontos médios entre valores consecutivos.
6. Avalia cada limite:
    * Separa as observações em dois subconjuntos.
    * Calcula o MSE de cada subconjunto.
    * Calcula o MSE ponderado da divisão.
7. Atualiza a melhor divisão quando encontra um erro estritamente menor.

Parâmetros:

* X: matriz de características.
* y: vetor numérico de respostas.

Retorno:

* Uma tupla (j, limit), contendo a característica e o limite escolhidos.
* None, caso nenhuma divisão melhore estritamente o erro do conjunto original.

4.1.4. Método _build_tree

def _build_tree(self, X, y, depth=0):

Constrói a árvore de regressão recursivamente.

Diferentemente da classificação, o valor armazenado em cada folha é a média das respostas do conjunto associado ao nó:

mean = float(np.mean(y))

Essa escolha minimiza o erro quadrático dentro da folha.

Condições de parada:

1. A profundidade máxima foi atingida:

depth >= self.max_depth

2. O número de observações é insuficiente:

len(y) < self.min_samples

3. Todos os valores de resposta são iguais:

np.all(y == y[0])

Quando alguma condição é satisfeita, retorna:

{"folha": True, "valor": mean}

Caso contrário, procura a melhor divisão utilizando _best_division.

Se nenhuma divisão for encontrada, o nó também se torna uma folha.

Se existir uma divisão válida, os dados são separados e as duas subárvores são construídas recursivamente.

Parâmetros:

* X: matriz de características.
* y: vetor de respostas numéricas.
* depth: profundidade atual do nó, iniciada em zero.

Retorno: dicionário que representa uma subárvore.

4.1.5. Método adjust

def adjust(self, X, y):

Treina a árvore de regressão utilizando os dados fornecidos.

Parâmetros:

* X: matriz bidimensional de características.
* y: vetor unidimensional de respostas numéricas.

Validações:

Condição	Comportamento
X não é bidimensional	Lança ValueError.
y não é unidimensional	Lança ValueError.
Quantidades de observações diferentes	Lança ValueError.
y vazio	Lança ValueError.
Nenhuma característica em X	Lança ValueError.

Os dados são convertidos para arrays NumPy de tipo float.

Após as verificações, a árvore é construída:

self.root = self._build_tree(X, y)

Retorno: o próprio objeto (self).

4.1.6. Método _predict_sample

def _predict_sample(self, x, no):

Realiza a previsão de uma observação percorrendo a árvore de regressão.

Parâmetros:

* x: vetor de características da observação.
* no: nó atual da árvore.

Funcionamento:

1. Se o nó for uma folha, retorna o valor numérico armazenado.
2. Caso contrário, obtém a característica e o limite da divisão.
3. Verifica se:

$$
x_j \leq t.
$$

4. Se a condição for verdadeira, percorre o ramo esquerdo.
5. Caso contrário, percorre o ramo direito.

Retorno: número real correspondente à previsão.

4.1.7. Método predict

def predict(self, X):

Realiza previsões para um conjunto de observações.

Parâmetro:

* X: matriz bidimensional contendo as observações para as quais serão produzidas previsões.

Funcionamento:

1. Verifica se o modelo já foi treinado.
2. Converte X para um array NumPy de tipo float.
3. Verifica se a matriz é bidimensional.
4. Verifica se existe pelo menos uma característica.
5. Percorre as observações e aplica _predict_sample.
6. Converte as previsões em um array NumPy.

Retorno: array unidimensional contendo os valores previstos.

Exceções:

* ValueError: quando a árvore ainda não foi treinada.
* ValueError: quando X não é bidimensional.
* ValueError: quando X não possui características.

⸻

5. Funcionamento matemático geral

5.1. Impureza do nó e dos filhos

Considere um nó com impureza (I(D)).

Após uma divisão, surgem dois subconjuntos, (D_L) e (D_R).

A impureza resultante é:

$$
I_{\text{filhos}} =
\frac{|D_L|}{|D|}I(D_L)
+
\frac{|D_R|}{|D|}I(D_R).
$$

A redução de impureza é:

$$
\Delta I =
I(D)-I_{\text{filhos}}.
$$

Como o código compara a impureza ponderada dos filhos com a impureza do pai, uma divisão somente é aceita quando:

$$
\Delta I>0.
$$

Em outras palavras, uma divisão precisa melhorar estritamente o critério utilizado.

5.2. Diferenças entre classificação e regressão

Característica	Classificação	Regressão
Critério	Gini	MSE
Objetivo	Reduzir a heterogeneidade das classes	Reduzir a variabilidade das respostas
Valor armazenado na folha	Classe majoritária	Média das respostas
Tipo de previsão	Categórica	Numérica

⸻

6. Estrutura interna da árvore

A árvore é armazenada utilizando dicionários Python.

Existem dois tipos de nós.

6.1. Nó folha

Uma folha representa o resultado final de uma observação que percorreu a árvore.

Na classificação:

{"folha": True, "valor": class_majority}

Na regressão:

{"folha": True, "valor": mean}

O campo valor contém a previsão associada à folha.

6.2. Nó interno

Um nó interno contém uma regra de divisão e referências para as duas subárvores:

{
    "folha": False,
    "caracteristica": j,
    "limite": limit,
    "esquerda": ...,
    "direita": ...
}

Os campos possuem os seguintes significados:

* folha: indica se o nó é terminal.
* caracteristica: índice da variável utilizada na divisão.
* limite: valor numérico utilizado como limiar.
* esquerda: subárvore das observações que satisfazem a condição.
* direita: subárvore das observações que não satisfazem a condição.

Os índices das características começam em zero.

⸻

7. Parâmetros de controle do crescimento

As duas classes utilizam os mesmos parâmetros.

7.1. max_depth

Controla a profundidade máxima da árvore.

A raiz possui profundidade zero. Cada divisão aumenta a profundidade dos nós filhos em uma unidade.

Para max_depth=5, por exemplo:

* A raiz está no nível 0.
* Seus filhos estão no nível 1.
* Os filhos seguintes estão no nível 2.
* A profundidade máxima dos nós é 5.

Os nós que atingem profundidade 5 tornam-se folhas.

Diminuir esse parâmetro restringe o crescimento da árvore e pode reduzir sua complexidade.

Aumentá-lo permite regras de decisão mais detalhadas, mas pode favorecer o ajuste excessivo aos dados de treinamento.

## 7.2. `min_samples`

Controla a quantidade mínima de observações necessária para que o algoritmo tente dividir um nó.

A condição implementada no código é:

$$
n_{\text{nó}} < \text{min\_samples}
$$

Portanto, uma divisão ainda pode ser tentada quando:

$$
n_{\text{nó}} = \text{min\_samples}
$$

**Importante:** esse parâmetro não impõe uma quantidade mínima de observações em cada folha. O código não verifica o tamanho dos filhos antes de aceitar uma divisão. Assim, uma folha pode terminar contendo apenas uma observação.

8. Exemplos de utilização

8.1. Classificação

O bloco principal do código cria um conjunto de dados com três características e três classes.

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

O modelo é instanciado e treinado:

modelo = DecisionTreeClassification()
modelo.adjust(X_train, y_train)

Em seguida, são fornecidas novas observações:

X_test = np.array([
    [1.8, 1.5, 1.5],
    [7.8, 1.5, 7.5],
    [4.5, 7.5, 4.5]
])
previsoes = modelo.predict(X_test)

As previsões são comparadas com as respostas reais:

y_test = np.array([0, 1, 2])
acuracia = np.mean(previsoes == y_test)

A acurácia é calculada por:

$$
\text{Acurácia} =
\frac{\text{Número de previsões corretas}}
{\text{Número total de previsões}}.
$$

Ela representa a proporção de observações classificadas corretamente.

8.2. Regressão

O conjunto de treinamento contém três características e uma resposta numérica:

X_train_reg = np.array([
    [1.0, 1.0, 1.0],
    [2.0, 1.0, 2.0],
    [1.0, 2.0, 2.0],
    [3.0, 2.0, 1.0],
    [2.0, 3.0, 1.0],
    [3.0, 3.0, 2.0]
])
y_train_reg = np.array([4.0, 5.0, 5.0, 7.0, 9.0, 10.0])

O modelo é criado e treinado:

modelo_reg = DecisionTreeRegression()
modelo_reg.adjust(X_train_reg, y_train_reg)

Novas observações são utilizadas para previsão:

X_test_reg = np.array([
    [1.5, 1.5, 1.5],
    [2.5, 2.0, 1.5]
])
previsoes_reg = modelo_reg.predict(X_test_reg)

O código compara as previsões com os valores reais utilizando o erro absoluto médio (MAE):

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}|\hat y_i-y_i|.
$$

Esse indicador representa o erro absoluto médio das previsões e é expresso na mesma unidade da variável resposta.


9. Limitações da implementação

A implementação apresenta algumas simplificações que devem ser consideradas antes de utilizá-la em aplicações mais gerais.

9.1. Características exclusivamente numéricas

A matriz X é convertida para o tipo float.

Consequentemente, as características devem ser representadas numericamente. Variáveis categóricas precisam ser codificadas previamente.

9.2. Ausência de tratamento de valores ausentes

O código não implementa uma estratégia específica para lidar com valores NaN.

A presença desses valores pode comprometer a geração dos limites candidatos e a separação das observações.

Recomenda-se tratar valores ausentes antes do treinamento e da previsão.

9.3. Ausência de poda

A árvore não realiza poda (pruning).

Depois que os critérios de parada são satisfeitos, a estrutura construída é mantida integralmente.

Não existe um procedimento posterior para remover divisões que tenham pouca capacidade de generalização.

9.4. Ausência de restrição ao tamanho dos filhos

O parâmetro min_samples controla apenas se o nó atual pode tentar uma divisão.

Não existe uma restrição que exija determinado número mínimo de observações nos dois filhos.

Isso permite divisões que produzem folhas muito pequenas.

9.5. Ausência de validação do número de características na previsão

Os métodos predict verificam se a entrada é bidimensional, mas não verificam se a quantidade de características corresponde à utilizada no treinamento.

Caso uma observação tenha menos características do que a árvore exige, a previsão pode produzir um erro de índice.

Caso possua características adicionais, elas podem simplesmente ser ignoradas.

Uma versão mais robusta deveria armazenar o número de características durante o treinamento e compará-lo com X.shape[1] durante a previsão.

9.6. Ausência de validação dos hiperparâmetros

Os construtores não verificam se max_depth e min_samples possuem valores válidos.

Por exemplo, valores negativos podem produzir comportamentos inesperados.

9.7. Critério de desempate na classificação

Quando duas ou mais classes possuem a mesma frequência em um nó, o método:

np.argmax(counting)

seleciona a primeira frequência máxima.

Como np.unique organiza os valores distintos, o desempate normalmente favorece a primeira classe na ordenação utilizada pelo NumPy.

Portanto, o resultado de um empate não é escolhido aleatoriamente.

9.8. Busca exaustiva por divisões

Para cada nó, o algoritmo percorre todas as características e avalia os limites candidatos gerados a partir dos valores distintos.

Cada candidato exige a construção de máscaras e o recálculo da impureza dos subconjuntos.

Essa estratégia é simples e adequada para fins didáticos, mas pode ser custosa para conjuntos grandes.

A implementação não utiliza estruturas otimizadas de ordenação incremental ou cálculo acumulado das estatísticas.

9.9. Ausência de pesos e aleatoriedade

O código não implementa:

* Pesos individuais para as observações.
* Pesos para as classes.
* Seleção aleatória de características.
* Seleção aleatória de observações.
* Floresta aleatória (Random Forest).
* Critérios alternativos de divisão.

Cada divisão é determinada de maneira determinística a partir dos dados e dos parâmetros, ressalvadas situações em que existam empates entre candidatos.


10. Complexidade computacional

Considere:

* (n): número de observações.
* (p): número de características.
* (d): profundidade máxima da árvore.

Em um nó com (m) observações, uma característica pode gerar até (m-1) limites candidatos.

Para cada candidato, o algoritmo constrói máscaras e calcula as impurezas dos subconjuntos, operações que podem exigir tempo proporcional a (m).

Portanto, uma estimativa para o custo da busca por divisões em um único nó é:

$$
O(pm^2).
$$

Na implementação atual, uma árvore muito desbalanceada pode apresentar custo acumulado de ordem:

$$
O(pn^3),
$$

em um cenário extremo no qual os tamanhos dos nós diminuem aproximadamente uma observação por nível.

Em árvores mais equilibradas, o custo total pode ficar próximo de:

$$
O(pn^2).
$$

Essas estimativas são referentes à estratégia de busca implementada e não devem ser confundidas com a complexidade de versões otimizadas de árvores de decisão.


11. Diferenças em relação a bibliotecas convencionais

A implementação apresenta uma API própria:

* adjust: realiza o treinamento.
* predict: realiza as previsões.

Em bibliotecas como Scikit-learn, é comum utilizar os métodos fit e predict.

Além disso, a implementação não disponibiliza diretamente:

* predict_proba para classificação.
* score para avaliação automática.
* Exportação gráfica da árvore.
* Importância das características.
* Poda da árvore.
* Validação cruzada.
* Seleção automática de hiperparâmetros.

Essas funcionalidades não fazem parte do código apresentado.


12. Referência rápida da API

DecisionTreeClassification

Método	Finalidade	Retorno
__init__(max_depth=5, min_samples=2)	Inicializa o classificador.	Nenhum retorno explícito.
_gini(y)	Calcula a impureza de Gini.	Número real.
_best_division(X, y)	Encontra a melhor divisão segundo o Gini.	Tupla (j, limite) ou None.
_build_tree(X, y, depth=0)	Constrói a árvore recursivamente.	Dicionário.
adjust(X, y)	Treina o modelo.	self.
_predict_sample(x, no)	Prevê a classe de uma observação.	Classe.
predict(X)	Prevê as classes de várias observações.	Array NumPy.

DecisionTreeRegression

Método	Finalidade	Retorno
__init__(max_depth=5, min_samples=2)	Inicializa o regressor.	Nenhum retorno explícito.
_mse(y)	Calcula o erro quadrático médio.	Número real.
_best_division(X, y)	Encontra a melhor divisão segundo o MSE.	Tupla (j, limite) ou None.
_build_tree(X, y, depth=0)	Constrói a árvore recursivamente.	Dicionário.
adjust(X, y)	Treina o modelo.	self.
_predict_sample(x, no)	Prevê o valor de uma observação.	Número real.
predict(X)	Prevê os valores de várias observações.	Array NumPy.


13. Execução do módulo

O trecho:

if __name__ == "__main__":

garante que os exemplos de classificação e regressão sejam executados somente quando o arquivo for executado diretamente.

Se o arquivo for importado por outro módulo, esse bloco não será executado automaticamente.

Exemplo:

from decision_tree import (
    DecisionTreeClassification,
    DecisionTreeRegression
)

A partir daí, as classes podem ser instanciadas e utilizadas normalmente.


14. Conclusão

O código implementa duas árvores de decisão binárias construídas recursivamente.

Ambas seguem o mesmo princípio:

1. Considerar as características disponíveis.
2. Gerar limites candidatos entre valores consecutivos.
3. Avaliar a qualidade de cada divisão.
4. Selecionar a divisão que mais reduz a impureza.
5. Construir duas subárvores recursivamente.
6. Interromper o processo quando uma condição de parada for atingida.
7. Utilizar o valor armazenado nas folhas para produzir previsões.

A diferença central está na função objetivo:

* Na classificação, utiliza-se o índice de Gini e a classe majoritária.
* Na regressão, utiliza-se o erro quadrático médio e a média das respostas.

A implementação constitui uma versão didática e funcional de árvores de decisão, permitindo estudar diretamente a relação entre critérios matemáticos de divisão, construção recursiva de estruturas e produção de previsões.
