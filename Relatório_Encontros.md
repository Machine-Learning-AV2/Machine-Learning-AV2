# Reunião dia 03/010/2026

Participantes:
  - Marce Alves (Redes Neurais) : Comparaceu
  - Natan Spohr (Classificação): Compareceu
  - Samuel Leite (Regressão): Compareceu
  - Kenzo Nagamine (Não Supervisionado): Compareceu
  - Lucas Caraccas (Reamostragem Seleção): Compareceu
  - Pedro Paulo (Métricas): Não Compareceu
  - Lucas Oliveira (Árvores de Decisão): Não Compareceu
  - Pedro Carvalho (Pré-Processamento): Não pode comparecer

No encontro foi definido que cada participante será responsável por um módulo.

Como o projeto parte 1 e parte 2 precisam estar prontos até dia 09/11 foi acordado que todos os métodos seriam finalizados até o dia 17/10 (2 semanas). Tendo em seguida mais uma semana (dia 24/10) para fazer a revisão dos módulo, juntamente da finalização dos casos de teste e a documentação.

Para otimização do processo de documentação e casos de teste foi recomendado que seja cada responsável por um módulo já realize uma pequena documentação e casos de teste. 
  * Para a documentação recomenda-se criar um arquivo markdown "módulo.md" e adicionar todas as documentações para as classes e métodos criados, incluindo os parâmetros utilizado, funcionalidades, etc.
  * Para os testes foi recomentado criar uma parte no final do codigo com if __ name __ == "__ main __", e adicionar os testes para cada módulo. Depois de finalizar por completo devemos redirecionar os testes para a pasta tests com o nome de arquivo "test_modulo.py".

Como observações: 
  * É extremamente recomendado que cada participante crie uma branch e se certifique que não esteja trabalhando em mesmo arquivos que outros colegas para evitar problemas de colisão
  * Após finalizar o processo realizar um pull request e esperar verificação de colisões feitas por Marce ou Natan.
  * Os módulos devevem ser criados dentro da pasta src, com os nomes dados por "modulo.py". __Atenção__: Não é nescessário criar uma nova pasta com o nome do seu módulo.
  * Cada módulo conterá as respectivas classes ou funcões nescessárias, não separando em arquivos diferentes.



Cada participante deve até dia 17:
  * Finalizar seu respectivo módulo;
  * Criar os casos de teste para seu módulo;
  * Criar uma documentação para seu módulo;
  
  
  


# Reunião dia 17/10/2026
Participantes:
- Marce Alves (Redes Neurais) : Compareceu
- Natan Spohr (Classificação): Compareceu
- Samuel Leite (Regressão): Não Compareceu
- Kenzo Nagamine (Não Supervisionado): Compareceu
- Lucas Caracas (Reamostragem Seleção): Compareceu
- Pedro Paulo (Métricas): Compareceu
- Lucas Oliveira (Árvores de Decisão): Não Compareceu
- Pedro Carvalho (Pré-Processamento): Não Compareceu

Reunião anotações:

1) Padronizar a forma de fazer o as implementações deixando em arquivo único .py, com as classes. 


2) Colocar comentários logo abaixo da classe explicando o funcionamento, para aparecer a explicação sempre que passar o cursor por cima do objeto no vs code. Exemplo:
def perda(x: np.ndarray, y:np.ndarray, p:float | str, l:list = [], subs:float = 1) -> float:

    """Retorna o valor da soma de uma função perda para um vetor de valores previstos e observados [..]
    ”””

3) Foi criado uma pasta “Documentation” para todos deixarem a documentação da sua respectiva parte, a documentação deve estar no formato markdown .md.

4) Foi definido que os nomes dos arquivos, atributos do objeto, classe, métodos e funções devem estar em inglês.
Nome de classe deve ser UpperCamelCase, exemplo: “MinhaClasse”  
Nome de função/método deve ser snake case, exemplo: “minha_funcao”

5) Padronizar o tipo de dados para array

6) Esperar todo mundo terminar as suas respectivas partes para no final juntar e fazer o relatório e a documentação geral.

7) Foi criando uma nova issue (SVM) para ser feito (classificação).

8) Colocar tolerância opcional nos teste, com o padrão de 10e-9

9) Sugestões de base de dados para analisar: ENEM (que possuimos vários relatórios e análises), Olimpíada

