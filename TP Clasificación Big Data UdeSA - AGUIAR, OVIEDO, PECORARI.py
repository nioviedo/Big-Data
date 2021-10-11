# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:20:13 2020

@author: Ovi
"""
#==================================================#
#                                                  #
#               Trabajo Práctico 1                 #
#                                                  #
#                                                  #
#==================================================#

import os 
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
#pip install heatmapz
from heatmap import heatmap, corrplot
import seaborn as sns

os.getcwd()  
os.chdir('C:\\Users\\Ovi\\Desktop\\Big Data\\TP_Big_Data')



# --- PARTE I: ANALIZANDO LA BASE --- #



#2.a) Importamos la base de datos

data = pd.ExcelFile('usu_individual_T120.xlsx')
df = data.parse('usu_pers_T12020')

# Nos quedamos con los datos del GCBA (REGION = 1)

df_gcba = df[df.REGION == 1]
df_gcba['REGION'].mean() #chequiamos que efectivamente se hallan eliminado los demás regiones, la media tiene que dar 1 para que así sea


#2.b) Eliminamos variables que no tienen sentido, es decir, eliminamos las edades que sean menores o iguales a cero y los ingresos que sean negativos (las condiciones están al revés porque expresamos los valores que quedemos que se queden)

df_gcba1 = df_gcba[df_gcba.CH06 > 0]
df_gcba1 = df_gcba1[df_gcba1.IPCF >= 0]      
df_gcba1 = df_gcba1[df_gcba1.ITF >= 0]


#2.c) Grafico de barras mostrando la composición por género. Primero creo un nuevo data frame para renombrar las variables y que sea más sencillo de graficar, luego grafico.

género = df_gcba1['CH04'].replace(1, 'Varón').replace(2, 'Mujer')

fig = género.value_counts().plot(kind='bar', figsize=(7, 6), rot=0, color={'r','b'})
plt.xlabel("Género", labelpad=14)
plt.ylabel("Cantidad de personas", labelpad=14)
plt.title("EPH Gran Buenos Aires - 1er trimestre 2020", y=1.02);


#2.d) Graficamos la matriz de correlación de las variables pedidas

df_corr = df_gcba1[["CH04", "CH07","CH08","NIVEL_ED","ESTADO","CAT_INAC","IPCF"]]

corr = df_corr.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


#2.e) En la EPH las categorías están empleadas de la siguiente forma: Desocupados Estado = 2 & Inactivo = 3

df_gcba1["ESTADO"].value_counts() #contamos la cantidad de valores que hay dentro de la columna ESTADO
a = [df_gcba1["ESTADO"].value_counts()] #guardamos estos valores en "a" y luego nos quedamos solo con los valores que requerimos (desocupados e inactivos)

desocupados = (a[-1][2])
inactivos = (a[-1][3])

print(f'Hay {desocupados} desocupados y {inactivos} inactivos')

# Calculamos la media de ingreso per cápita familiar (IPCF) según estado 

df_gcba1.groupby('ESTADO', as_index=False)['IPCF'].mean()


#2.f) En este punto se realizarán una serie de operaciones para agregar columnas de adulto equivalente a partir de la base de datos provista

#Llamamos al nuevo archivo donde se encuentra la información de adulto equivalente
adultoeq = pd.ExcelFile('tabla_adulto_equiv.xlsx')
dfadultoeq = adultoeq.parse('Tabla de adulo equivalente')

#Eliminamos la última columna que no sirve y luego limpiamos las filas que están vacías.
dfadultoeq = dfadultoeq.drop(['Unnamed: 3'], axis=1)
dfadultoeq = dfadultoeq.dropna()

#Terminamos de formatear la tabla de equivalencia
dfadultoeq.columns=['Edad', 'Mujeres', 'Varones']
dfadultoeq = dfadultoeq.drop(2,axis=0)
dfadultoeq = dfadultoeq.set_index('Edad')

#Agregamos la columna de adulto equivalente a nuestro data set principal
df_gcba1['adulto_equiv'] = np.nan

#Generamos una nueva columna que contenga una etiqueta de acuerdo con el valor de la edad
df_gcba1['edadcat'] = np.nan
age = df_gcba1.get('CH06')
age.astype('float64').dtypes

#Etiquetador

idx = list(df_gcba1.index.values.tolist())
i=0
j = idx[i]

for elt in age:
    
    j = idx[i]
    k = elt
    
    if elt < 1:
       df_gcba1.at[{j}, 'edadcat'] = 'Menor de 1 año'
       i= i + 1
   
    elif elt == 1:
       df_gcba1.at[{j}, 'edadcat'] = '1año'
       i= i + 1
   
    elif elt == 2:
       df_gcba1.at[{j}, 'edadcat'] = '2 años'
       i= i + 1
   
    elif elt > 2 and elt < 18:
       df_gcba1.at[{j}, 'edadcat'] = f'{k} años'
       i= i + 1
   
    elif elt >= 18 and elt < 30:
       df_gcba1.at[{j}, 'edadcat'] = '18 a 29 años'
       i= i + 1

    elif elt >= 30 and elt < 46:
       df_gcba1.at[{j}, 'edadcat'] = '30 a 45 años'
       i= i + 1

    elif elt >= 46 and elt < 61:
       df_gcba1.at[{j}, 'edadcat'] = '46 a 60 años'
       i= i + 1

    elif elt >= 61 and elt < 75:
       df_gcba1.at[{j}, 'edadcat'] = '61 a 75 años'
       i= i + 1

    elif elt >= 75:
       df_gcba1.at[{j}, 'edadcat'] = 'más de 75 años'
       i= i + 1
       
    else:
        df_gcba1.at[{j}, 'edadcat'] = 'nan'
        print('Missing value')
        i= i + 1 #Esta última condición sirve para controlar que no haya que dado un missing value en la edad

#Controlamos visualmente el resultado del loop
control = df_gcba1[["CH06", "edadcat"]]

#Ahora establecemos la equivalencia respecto de un adulto para cada observación

i=0
j = idx[i]

for i in range(8304):
    j = idx[i]
    if float(df_gcba1.at[j, 'CH04']) == 1:
        m = dfadultoeq.loc[df_gcba1.edadcat, "Varones"].values
        float(m[i])
        df_gcba1.at[j, 'adulto_equiv'] = float(m[i])
        i = i + 1
    else:
        m = dfadultoeq.loc[df_gcba1.edadcat, "Mujeres"].values
        float(m[i])
        df_gcba1.at[j, 'adulto_equiv'] = float(m[i])
        i = i + 1

#Controlamos visualmente que haya corrido bien el loop

control = df_gcba1[["CH06", "edadcat", "CH04", "adulto_equiv"]]

#Agrupamos a las observaciones por hogar

grupo = df_gcba1.groupby(['CODUSU'])
print(df_gcba1.groupby(['CODUSU']).size())

#Calculamos la sumatoria de los valores de adulto equivalente de cada miembro, por hogar

valencia = grupo['adulto_equiv'].agg(np.sum)
print(grupo['adulto_equiv'].agg(np.sum))
valencia = pd.DataFrame(valencia)

#Asignamos el correspondiente valor a cada una de las observaciones en nuestro dataset principal
df_gcba1['ad_equiv_hogar'] = np.nan

i=0
j = idx[i]
n = df_gcba1.at[j, 'CODUSU']

o = valencia.at[n, 'adulto_equiv']
valencia.loc['TQRMNOVWRHJKMNCDEIIAD00626757',:]

while i in range(8304):
    j = idx[i]
    n = df_gcba1.at[j, 'CODUSU']
    o = valencia.at[n, 'adulto_equiv']
    df_gcba1.at[j, 'ad_equiv_hogar'] = o
    i = i + 1

#Controlamos
control = df_gcba1[["CODUSU", "ad_equiv_hogar", "adulto_equiv"]]    
     
#3) Se crean dos nuevas bases, la primera contiene las observaciones que no respondieron a la pregunta sobre su ITF, mientras que la segunda corresponde a aquellos que si respondieron.

norespondieron = df_gcba1[df_gcba1['ITF'] == 0]
noresp = len(norespondieron)
print(f'Hay {noresp} personas que no reportan su ingreso total familiar (ITF)')

respondieron = df_gcba1[df_gcba1['ITF'] != 0]
resp = len(respondieron)
print(f'Hay {resp} personas que reportaron su ingreso total familiar (ITF)')


#4) Agregamos una nueva columna llamada ingreso_necesario que sea igual al producto de la Canasta Básica Total para un adulto equivalente en el Gran Buenos Aires ($13.286) y el valor por ad_equiv_hogar

respondieron['ingreso_necesario'] = (13286 * respondieron['ad_equiv_hogar']) 


#5) Creamos una lista de condiciones para determinar si los individuos son pobres o no. Si su ingreso es mayor o igual al necesario, entonces se les asigna
# un 0 y no son pobres, mientras que por el contrario si el ingreso es inferior al necesario se les asigna un 1 y se los considera pobres.

# Se establecen las condiciones
conditions = [
    (respondieron['ITF'] < respondieron['ingreso_necesario']),
    (respondieron['ITF'] >= respondieron['ingreso_necesario'])
    ]

# Se establecen los valores que serán asignados para cada condición previamente determinada
values = ['1', '0']

# Se crea una nueva columna y se usa np.select para asignarle los valores a dicha columna en función de las condiciones y los valores establecidos
respondieron['pobre'] = np.select(conditions, values)

# Para contar la cantidad de pobres identificados se guardan los valores contados en "b" y luego nos quedamos con la fila de 1, es decir, con la cantidad de pobres

b = respondieron['pobre'].value_counts() 
pobres = (b[1])
print(f'Hay {pobres} pobres en la muestra analizada')



# --- PARTE II: CLASIFICACIÓN --- #


#1) Procedemos a eliminar las columnas relacionadas a ingresos y a adulto equivalente. Por eso eliminamos desde la primer columna de ingresos (P21) hasta la de ingreso_necesario. Dejamos la columna pobre porque será utilizada más adelante 

respondieron = respondieron.drop(respondieron.loc[:, 'P21':'ingreso_necesario'].columns, axis = 1) 
norespondieron = norespondieron.drop(norespondieron.loc[:, 'P21':'ad_equiv_hogar'].columns, axis = 1) #se elimina hasta esta columna porque para esta base no hay columnas de ingreso_necesario ni pobre


#2) Importamos la librería a utilizar. Usamos la función train_test_split especificando que queremos el 30% como test, dejando el restante 70% como entrenamiento y le asignamos la semilla random_state = 101 como pedido.

from sklearn.model_selection import train_test_split

train, test = train_test_split(respondieron, test_size = 0.3, random_state = 101)


#3) Establecemos la variable "pobre" como variable dependiente (Y) y el resto de las variables independientes entran dentro de la matriz X, para la constantes se agrega una columna de 1
import statsmodels.api as sm

ytrain = train['pobre']
ytrain = ytrain.apply(pd.to_numeric, errors='coerce')

ytest = test['pobre']
ytest = ytest.apply(pd.to_numeric, errors='coerce')

#Eliminamos la variable dependiente de las matrices de variables independientes
xtrain = train.drop('pobre',axis=1)
xtest = test.drop('pobre',axis=1)

#Purgamos las matrices, eliminando predictores que son strings y columnas con missing values
xtrain = xtrain.drop('CODUSU',axis=1)
xtrain = xtrain.drop('CH05',axis=1)
xtrain = xtrain.dropna(axis='columns')
xtrain= xtrain.apply(pd.to_numeric, errors='coerce')

xtest = xtest.drop('CODUSU',axis=1)
xtest = xtest.drop('CH05',axis=1)
xtest = xtest.dropna(axis='columns')
xtest= xtest.apply(pd.to_numeric, errors='coerce')

#Notar que las matrices de variables independientes ya tienen una columna de 1 (TRIMESTRE). Por ese motivo, no la agregaremos. La línea de código a implementar si esta no estuviera sería:
#x=sm.add_constant(train)

#4) Implementamos distintos modelos

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 

#Logit

logit_model=sm.Logit(ytrain.astype(float),xtrain.astype(float))
result=logit_model.fit()
print(result.summary())
print(result.summary().as_latex()) #para imprimir la salida en latex

y_pred = result.predict(xtest)
y_pred=np.where(y_pred>0.5, 1, y_pred)
y_pred=np.where(y_pred<=0.5, 0, y_pred)

confusion_matrix = confusion_matrix(ytest, y_pred)
print('Confusion Matrix :')
print(confusion_matrix)
print('Accuracy Score :',accuracy_score(ytest, y_pred))

# Recordar que en Python la matriz de confusión tiene en las filas los valores ciertos y en las columnas los valores predichos

auc = roc_auc_score(ytest, y_pred)
print('AUC: %.2f' % auc)

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()  

fpr, tpr, thresholds = roc_curve(ytest, y_pred)
plot_roc_curve(fpr, tpr)

#LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.metrics import confusion_matrix

clf = LinearDiscriminantAnalysis()
clf.fit(xtrain, ytrain)
resultslda=clf.predict(xtest)
y_pred_lda=pd.Series(resultslda.tolist())

cm = confusion_matrix(ytest, y_pred_lda)
print(cm)
print('Accuracy Score :',accuracy_score(ytest, y_pred_lda))

auc_lda = roc_auc_score(ytest, y_pred_lda)
print('AUC: %.2f' % auc_lda)
fpr, tpr, thresholds = roc_curve(ytest, y_pred_lda)
plot_roc_curve(fpr, tpr)

#K vecinos cercanos

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(xtrain, ytrain) 
y_pred = knn.predict(xtest)

cm2 = confusion_matrix(ytest, y_pred)
print(cm2)
print('Accuracy Score :',accuracy_score(ytest, y_pred))

auc_knn = roc_auc_score(ytest, y_pred)

print('AUC: %.2f' % auc_knn)
fpr, tpr, thresholds = roc_curve(ytest, y_pred)
plot_roc_curve(fpr, tpr)

#6) Usamos k vecinos cercanos en base a AUC

#Sobre norespondieron aplicamos el mismo tratamiento de depuración que a las matrices de variables independientes de respondieron
norespondieron = norespondieron.drop('CODUSU',axis=1)
norespondieron = norespondieron.drop('CH05',axis=1)
norespondieron = norespondieron.dropna(axis='columns')
norespondieron = norespondieron.apply(pd.to_numeric, errors='coerce')

pobre_pred = knn.predict(norespondieron)
np.mean(pobre_pred)

print(f'La proporción de personas que no respondieron que predecimos que son pobres es {np.mean(pobre_pred)}')

#7) Usamos un modelo logístico regularizado 
from sklearn import linear_model

model = linear_model.LogisticRegression(penalty='l1', solver= 'saga', max_iter= 10000).fit(xtrain, ytrain)
p_pred = model.predict(xtest)
auc_log2 = roc_auc_score(ytest, p_pred)
print('AUC: %.2f' % auc_log2)

#Código hecho en base a Python 3.7.6