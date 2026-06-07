# Diabetes Classification

Este projeto utiliza três algoritmos de Machine Learning para prever a ocorrência de diabetes com base em informações clínicas dos pacientes.

**Dataset:** diabetes.csv

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download

## Como Rodar

1. Dentro da pasta, criar e ativar o ambiente virtual
2. Instalar as dependências dentro do **requirements.txt**
3. Rodar o arquivo **classificador.py** para carregar os dados, balancear as classes, normalizar os atributos, treinar os modelos e exibir os resultados

## Modelos Utilizados

- **Random Forest** - Classificador baseado em múltiplas árvores de decisão com hiperparametrização utilizando RandomizedSearchCV
- **Support Vector Machine (SVM)** - Classificador baseado em margens máximas com hiperparametrização utilizando RandomizedSearchCV
- **Logistic Regression** - Classificador linear para problemas binários com hiperparametrização utilizando RandomizedSearchCV

## Avaliações

Para cada modelo são calculadas as seguintes métricas:

- **Acurácia**
- **Matriz de Confusão**
- **Especificidade**
- **Sensibilidade**

Ao final da execução é apresentada uma comparação entre os modelos e identificado automaticamente o modelo com melhor acurácia para utilização em produção.
