# Importação das bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Carregando o conjunto de dados
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Dividindo o conjunto de dados em 75% para treino e 25% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Normalizando o conjunto de dados
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Treinando o modelo SVM
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# Predizendo os resultados para o conjunto de teste
y_pred = classifier.predict(X_test)

# Calculando a Matriz Confusão e as demais métricas de avalição
cm = confusion_matrix(y_pred, y_test)

vp=cm[1][1] 
fp=cm[1][0]
fn=cm[0][1]
vn=cm[0][0]
acu = (vp+vn)/(vp+vn+fp+fn)
sensi = vp/(vp+fn)
especi = vn/(fp+vn)
preci = vp/(vp+fp)
fscore = 2*(preci*sensi)/(preci+sensi)
#Imprimindo os valores
print('Matriz Confusão: \n', cm)
print('Verdadeiros Positivo: ',vp)
print('Verdadeiros Negativo: ',vn)
print('Falsos Positivo: ',fp)
print('Falsos Negativo: ',fn)
print(f'Acurácia: {acu:.3f}')
print(f'Fscore: {fscore:.3f}')
print(f'Precisão: {preci:.3f}')
print(f'Especificidade: {especi:.3f}')
print(f'Sensibilidade: {sensi:.3f}')