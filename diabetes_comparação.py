import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from collections import Counter


#Abrir o arquivo de dados e separar atributos e classe
dados = pd.read_csv('diabetes.csv', sep = ',')
dados_atributos = dados.drop(columns=['Outcome'])
dados_classe = dados['Outcome']


#segmentar os dados em dados para treinamento e dados para teste
atributos_train, atributos_teste, \
    classe_train,classe_test = train_test_split(
            dados_atributos,
            dados_classe,
            test_size=0.3,
            random_state=42
        )

#Balancear os dados de treino
resampler = SMOTE(random_state=42) 

atributos_train, classe_train = \
    resampler.fit_resample(
    atributos_train,
    classe_train
)

print('#### FREQUENCIA DAS CLASSES APÓS O BALANCEAMENTO ###')
class_count = Counter(classe_train)
print(class_count)

    
scaler = StandardScaler()
atributos_train = scaler.fit_transform(atributos_train)
atributos_teste = scaler.transform(atributos_teste)


#-----------------
#--RANDOM FOREST--
#-----------------

print("--RANDOM FOREST--")

n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
max_features = ['sqrt', 'log2']

#criar a grade de valores RF
rf_grid={
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split':min_samples_split,
    'max_depth': max_depth, 
    'max_features': max_features
}

rf = RandomForestClassifier(random_state=42)
rf_hyperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=5,
    cv=3,
    verbose=2,
    n_jobs=1,
    random_state=42
)
rf_hyperparameters.fit(atributos_train, classe_train)

#Mostrar o resultado da hiperparametrização RF
print('Melhores parametros:')
pprint(rf_hyperparameters.best_params_)

#Recuperar melhor modelo RF
melhor_rf = rf_hyperparameters.best_estimator_

predicoes_rf = melhor_rf.predict(
    atributos_teste
)

#Calcular Acuracia RF
acuracia_rf = accuracy_score(
    classe_test,
    predicoes_rf
)
print("Acuracia: ", acuracia_rf)


#Matriz de confusão RF
cm_rf = confusion_matrix(
    classe_test,
    predicoes_rf
)   

print(cm_rf)

tn, fp, fn, tp = cm_rf.ravel()

especificidade_rf = tn/(tn+fp)
sensibilidade_rf = tp/(tp+fn)

print("Especificidade_rf:", especificidade_rf)
print("Sensibilidade_rf:", sensibilidade_rf)


#-------------------
#--------SVM--------
#-------------------

print("--------SVM--------")

svm_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm = SVC(random_state=42)

svm_hyperparameters = RandomizedSearchCV(
    estimator=svm,
    param_distributions=svm_grid,
    n_iter=5,
    cv=3,
    verbose=2,
    n_jobs=1,
    random_state=42
)
svm_hyperparameters.fit(atributos_train, classe_train)

#Mostrar o resultado da hiperparametrização SVM
print('Melhores parametros:')
pprint(svm_hyperparameters.best_params_)

melhor_svm = svm_hyperparameters.best_estimator_
predicoes_svm = melhor_svm.predict(atributos_teste)

#Calcular Acuracia SVM
acuracia_svm = accuracy_score(
    classe_test,
    predicoes_svm
)
print('Acuracia:', acuracia_svm)

#Matriz de confusão SVM
cm_svm = confusion_matrix(
    classe_test,
    predicoes_svm
)
print(cm_svm)

tn, fp, fn, tp = cm_svm.ravel()

especificidade_svm = tn / (tn + fp)
sensibilidade_svm = tp / (tp + fn)

print('Especificidade:', especificidade_svm)
print('Sensibilidade:', sensibilidade_svm)


#-----------------------
#--LOGISTIC REGRESSION--
#-----------------------

print("--LOGISTIC REGRESSION--")

lr_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

lr = LogisticRegression(max_iter=1000, random_state=42)

lr_hyperparameters = RandomizedSearchCV(
    estimator=lr,
    param_distributions=lr_grid,
    n_iter=5,
    cv=3,
    verbose=2,
    n_jobs=1,
    random_state=42
)

lr_hyperparameters.fit(atributos_train, classe_train)

#Mostrar o resultado da hiperparametrização LR
print('Melhores parametros:')
pprint(lr_hyperparameters.best_params_)

melhor_lr = lr_hyperparameters.best_estimator_
predicoes_lr = melhor_lr.predict(atributos_teste)

#Calcular Acuracia LR
acuracia_lr = accuracy_score(
    classe_test,
    predicoes_lr
)
print('Acuracia:', acuracia_lr)

#Matriz de confusão LR
cm_lr = confusion_matrix(
    classe_test,
    predicoes_lr
)
print(cm_lr)

tn, fp, fn, tp = cm_lr.ravel()

especificidade_lr = tn / (tn + fp)
sensibilidade_lr = tp / (tp + fn)

print('Especificidade:', especificidade_lr)
print('Sensibilidade:', sensibilidade_lr)


#----Comparação Final----

print('\n----Comparacao Final----')

print('Random Forest:', acuracia_rf)
print('SVM:', acuracia_svm)
print('Logistic Regression:', acuracia_lr)

resultados = {
    'Random Forest': acuracia_rf,
    'SVM': acuracia_svm,
    'Logistic Regression': acuracia_lr
}

melhor_modelo = max(
    resultados,
    key=resultados.get
)

print('\nMELHOR MODELO:')
print(melhor_modelo)
print('Acuracia:', resultados[melhor_modelo])