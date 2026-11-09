# Machine Learning Library

Biblioteca de algoritmos e ferramentas de **Aprendizado de Máquina (Machine Learning)**, desenvolvida em Python como parte das atividades da disciplina de Aprendizado de Máquina, durante o período letivo 2.2 do Bacharelado em Matemática e Tecnologia da Inovação no IMPA Tech. O projeto reúne implementações de diferentes métodos e técnicas de aprendizado de máquina aprendidos durante o curso, organizados em módulos, buscando manter uma estrutura de fácil compreensão.

**Autores:** Marce Alves da Silva, Natan Gabriel Spohr, Pedro Paulo Gomes Paiva, Kenzo Nagamine, Lucas Caracas, Lucas da Silva Oliveira, Pedro Pereira Carvalho e Samuel Leite

---

## Sobre o Projeto

Esta biblioteca foi desenvolvida com o objetivo de implementar e explorar, na prática, diferentes conceitos e algoritmos fundamentais de Aprendizado de Máquina. Como parte dos requisitos da atividade, todo o projeto foi implementado em **Python**, utilizando apenas a biblioteca externa **NumPy** como dependência para operações numéricas e computacionais.

## Estrutura do Repositório

O repositório está organizado em três diretórios principais, cada um com uma finalidade específica:

```text
.
├── Documentation/
├── src/
└── tests/
```

### Documentation

Diretório destinado à documentação técnica da biblioteca. Cada módulo possui um documento próprio em formato Markdown, contendo descrições detalhadas de suas classes, métodos e funções. A documentação também apresenta exemplos de utilização para melhor compreensão de seu funcionamento sem a necessidade de consultar diretamente o código-fonte. **Recomenda-se começar por este diretório** para obter uma visão geral da biblioteca.

### src

Diretório que contém o código-fonte da biblioteca e todas as implementações dos algoritmos e ferramentas desenvolvidos. O código está organizado em nove módulos principais, agrupados de acordo com suas respectivas áreas de aplicação em Aprendizado de Máquina:

#### 1. Regression

Módulo dedicado aos algoritmos de regressão, voltados à modelagem de relações entre variáveis e à predição de valores contínuos.

#### 2. Classification

Reúne os algoritmos de classificação KNN, LDA, QDA, Regressão Logística e SVM, responsáveis por atribuir observações a categorias ou classes predefinidas.

#### 3. NeuralNetwork

Contém implementação de rede neural artificial, modelos computacionais inspirados em estruturas de processamento de informação por unidades interconectadas.

#### 4. UnsupervisedLearning

Contém a implementação do método KMeans e do PCA, responsáveis por identificar padrões nos dados sem a necessidade de classes ou valores previamente atribuídos.

#### 5. DecisionTree

Módulo dedicado a algoritmos baseados em árvores de decisão, tanto para regressão quanto para classificação, que utilizam estruturas de regras em hierarquia para realizar tarefas de predição e classificação.

#### 6. Resampling

Reúne técnicas de reamostragem de dados, utilizadas em diferentes contextos de análise, treinamento e avaliação de modelos de aprendizado de máquina.

#### 7. Metrics

Disponibiliza métricas para avaliação de desempenho de modelos, permitindo quantificar e comparar seus resultados de acordo com diferentes critérios.

#### 8. Functions

Contém funções auxiliares e operações matemáticas utilizadas como suporte às implementações dos demais módulos da biblioteca, como sigmoide e arco tangente.

#### 9. Preprocessing

Reúne ferramentas de pré-processamento de dados, destinadas à preparação e transformação dos conjuntos de dados antes de sua utilização pelos algoritmos.

### tests

Diretório destinado aos testes automatizados da biblioteca, desenvolvidos utilizando o framework **pytest**.

Os testes verificam o comportamento de diferentes métodos e componentes por meio de verificações sistemáticas, permitindo identificar inconsistências e validar o funcionamento das implementações.

Além de auxiliar no desenvolvimento e na manutenção do projeto, essa estrutura oferece uma maneira prática de verificar a integridade da biblioteca como um todo.

Para executar os testes, utilize o comando:

```bash
pytest tests/
```

## Instalação

A instalação da biblioteca pode ser realizada diretamente pelo gerenciador de pacotes pip, utilizando o seguinte comando:

```bash
pip install machinelearnig-mnpkllps
