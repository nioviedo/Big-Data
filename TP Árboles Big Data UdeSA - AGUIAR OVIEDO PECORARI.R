#----------------BIG DATA UDESA 2020-----------------#
#----------------Trabajo práctico 3------------------#
#Por Lautaro Aguiar, Natalia Pecorari y Nicolás Oviedo

# Limpiamos el entorno, fijamos el directorio de trabajo e invocamos las librerías que necesitamos
rm(list=ls())
gc()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(readxl)
library(xlsx)
library(ggplot2)
library(dplyr)
library(tidyr)
library(xtable)

###---------------###  
###----PARTE I----###
###---------------###

#1. Abriendo la base----

dataset <- read_xlsx("usu_individual_T120.xlsx")#, header = TRUE, sep = ",")

#2. Contamos niños----

#Filtramos por edad, purgando los mayores a 10 años para poder contar los niños
#Interpretamos que la edad -1 representa a bebés de menos de 1 año, por eso los mantenemos
dataset2 <- dataset[dataset$CH06<=10,]

agregar <- aggregate(dataset2$CH06,by = list(dataset2$CODUSU), FUN = "length") #Agrupamos por hogar y contamos la cantidad de niños
colnames(agregar) <- c('CODUSU', 'ninios')
dataset<-merge(dataset, agregar, by.x="CODUSU", by.y="CODUSU", all.x=TRUE)

#Hacemos lo mismo pero para niños varones, lo vamos a usar luego
dataset3 <- dataset[dataset$CH06<=10 & dataset$CH04==1,] #Niños varones
agregar2 <- aggregate(dataset3$CH06,by = list(dataset3$CODUSU), FUN = "length") #Agrupamos por hogar y contamos la cantidad de niños
colnames(agregar2) <- c('CODUSU', 'ninios_varones')
dataset<-merge(dataset, agregar2, by.x="CODUSU", by.y="CODUSU", all.x=TRUE)

#Hacemos lo mismo pero para niñas, lo vamos a usar luego
dataset4 <- dataset[dataset$CH06<=10 & dataset$CH04==2,] #Niñas
agregar3 <- aggregate(dataset4$CH06,by = list(dataset4$CODUSU), FUN = "length") #Agrupamos por hogar y contamos la cantidad de niños
colnames(agregar3) <- c('CODUSU', 'ninias')
dataset<-merge(dataset, agregar3, by.x="CODUSU", by.y="CODUSU", all.x=TRUE)

#Reemplazamos NA por 0 en las tres columnas que generamos
dataset <- dataset %>% replace_na(list(ninios = 0, ninias = 0, ninios_varones =0))

#Control de integridad 
b <- dataset$ninios - dataset$ninios_varones - dataset$ninias
summary(b)

rm(agregar, agregar2, agregar3, dataset2, dataset3, dataset4, b) #Descartamos auxiliares

#3. Filtrado----
#Descartamos a los menores de 16 años y a quienes no respondieron la encuesta individual
dataset <- dataset[dataset$H15==1 & dataset$CH06>=16,]

#4. Trabaja----

#Estado: 1=ocupado , 2= desocupado, 3=inactivo

dataset$trabaja=ifelse(dataset$ESTADO==1,1,0)

a<-dataset[c("CODUSU", "ESTADO", "trabaja")] #Controlamos que se haya generado bien la binaria
rm(a) #Descartamos este dataset auxiliar

#5. Partimos el dataset por sexo----

dataset_varones <- dataset[dataset$CH04==1,]
dataset_mujeres <- dataset[dataset$CH04==2,]

#6. Tratamiento de missing values----

sum(is.na(dataset$P47T))
dataset%>% filter(P47T==-9) %>% count(P47T)

#Analizamos los missing values
df<-dataset[dataset$P47T==-9,]
df%>% filter(P47T==-9) %>% count(CH12)
df%>% filter(P47T==-9) %>% count(PP04C)

#Decidimos eliminar los missing values porque no hay evidencia para rechazar nuestra hipotesis de que se dan porque la gente es pudorosa en declarar su ingreso y el pudor es ortogonal al nivel de ingreso
dataset <- dataset[dataset$P47T!=-9,]

#Análogamente se eliminaran de la muestra las observaciones que posean valores 0 para las variables ITF e IPCF (y se verifica que no hay NAs), que corresponden a missing values de dichas preguntas
sum(is.na(dataset$IPCF))
dataset2%>% filter(IPCF==0) %>% count(IPCF)

#Se eliminan los missing values
dataset <- dataset[dataset$IPCF!=0,]

#corroboramos que las observaciones que poseían 0 en IPCF corresponden a las que poseen 0 en ITF y que no hay NA
sum(is.na(dataset$ITF))
dataset%>% filter(ITF==0) %>% count(ITF)

#Volvemos a partir la base por sexo para que no queden missing values en P47T en las particiones
dataset_varones <- dataset[dataset$CH04==1,]
dataset_mujeres <- dataset[dataset$CH04==2,]

#7. Análisis de variables por sexo----

#7.1.Ingreso====
#Varones
mean(dataset_varones$IPCF[dataset_varones$trabaja==1])
mean(dataset_varones$IPCF[dataset_varones$trabaja==0])

#Mujeres
mean(dataset_mujeres$IPCF[dataset_mujeres$trabaja==1])
mean(dataset_mujeres$IPCF[dataset_mujeres$trabaja==0])

#Generamos un dataset auxiliar para graficar los histogramas
dataset5 <- dataset
dataset5$CH04 <- factor(dataset5$CH04, labels = c("Varón", "Mujer"))
colnames(dataset5$CH04) <- c('Sexo')
dataset5 <- dataset5 %>% rename(Sexo = CH04)

#Histograma 1
{ggplot(dataset5[dataset5$trabaja==1,], aes(x = IPCF)) +
  geom_histogram(aes(color = Sexo, fill = Sexo), 
                 position = "identity", bins = 15, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#bb0080")) +
  scale_fill_manual(values = c("#00AFBB", "#bb0080")) + scale_y_continuous(name = "Cantidad") + scale_x_continuous(limits = c(0,80000), name="Ingreso Per Cápita Familiar (IPCF)") + geom_vline(aes(xintercept = median(dataset_mujeres$IPCF[dataset_mujeres$trabaja==1])),col='red',size=1) + geom_vline(aes(xintercept = median(dataset_varones$IPCF[dataset_varones$trabaja==1])),col='blue',size=1)}

#Histograma 2
{ggplot(dataset5[dataset5$trabaja==0,], aes(x = IPCF)) +
  geom_histogram(aes(color = Sexo, fill = Sexo), 
                 position = "identity", bins = 15, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#bb0080")) +
  scale_fill_manual(values = c("#00AFBB", "#bb0080")) + scale_y_continuous(name = "Cantidad") + scale_x_continuous(limits = c(0,80000), name="Ingreso Per Cápita Familiar (IPCF)") + geom_vline(aes(xintercept = median(dataset_mujeres$IPCF[dataset_mujeres$trabaja==0])),col='red',size=1) + geom_vline(aes(xintercept = median(dataset_varones$IPCF[dataset_varones$trabaja==0])),col='blue',size=1)}

#7.2 Edades promedio=================================
##Varones
mean(dataset_varones$CH06[dataset_varones$trabaja==1])
mean(dataset_varones$CH06[dataset_varones$trabaja==0])

##Mujeres
mean(dataset_mujeres$CH06[dataset_mujeres$trabaja==1])
mean(dataset_mujeres$CH06[dataset_mujeres$trabaja==0])

##Histograma 3
{ggplot(dataset5[dataset5$trabaja==1,], aes(x = CH06)) +
  geom_histogram(aes(color = Sexo, fill = Sexo), 
                 position = "identity", bins = 15, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#bb0080")) +
  scale_fill_manual(values = c("#00AFBB", "#bb0080")) + scale_y_continuous(name = "Cantidad") + scale_x_continuous(limits = c(0,100), name="Edad") + geom_vline(aes(xintercept = median(dataset_mujeres$CH06[dataset_mujeres$trabaja==1])),col='red',size=1) + geom_vline(aes(xintercept = median(dataset_varones$CH06[dataset_varones$trabaja==1])),col='blue',size=1)}

##Histograma 4
{ggplot(dataset5[dataset5$trabaja==0,], aes(x = CH06)) +
  geom_histogram(aes(color = Sexo, fill = Sexo), 
                 position = "identity", bins = 15, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#bb0080")) +
  scale_fill_manual(values = c("#00AFBB", "#bb0080")) + scale_y_continuous(name = "Cantidad") + scale_x_continuous(limits = c(0,100), name="Edad") + geom_vline(aes(xintercept = median(dataset_mujeres$CH06[dataset_mujeres$trabaja==0])),col='red',size=1) + geom_vline(aes(xintercept = median(dataset_varones$CH06[dataset_varones$trabaja==0])),col='blue',size=1)}

#7.3 Promedio de niños por hogar ====
{mean(dataset_varones$ninios[dataset_varones$trabaja==1])
a11<-mean(dataset_varones$ninios_varones[dataset_varones$trabaja==1])
a12<-mean(dataset_varones$ninias[dataset_varones$trabaja==1])

mean(dataset_varones$ninios[dataset_varones$trabaja==0])
a21<-mean(dataset_varones$ninios_varones[dataset_varones$trabaja==0])
a22<-mean(dataset_varones$ninias[dataset_varones$trabaja==0])

mean(dataset_mujeres$ninios[dataset_mujeres$trabaja==1])
a31<-mean(dataset_mujeres$ninios_varones[dataset_mujeres$trabaja==1])
a32<-mean(dataset_mujeres$ninias[dataset_mujeres$trabaja==1])

mean(dataset_mujeres$ninios[dataset_mujeres$trabaja==0])
a41<-mean(dataset_mujeres$ninios_varones[dataset_mujeres$trabaja==0])
a42<-mean(dataset_mujeres$ninias[dataset_mujeres$trabaja==0])}

C <- matrix(data=c(a11,a12,a21,a22,a31,a32,a41,a42),nrow = 4, ncol = 2, byrow = TRUE)

rownames(C) <- list('Varón que trabaja','Varón desocupado','Mujer que trabaja','Mujer desocupada')
colnames(C) <- list('Promedio Niños en hogar', 'Promedio Niñas en hogar')

x = xtable(C)
print(x)

#7.4 Jubilados y amas de casa =====

jub<-dataset_varones %>% filter(trabaja==0) %>% count(CAT_INAC) %>%  filter(CAT_INAC==1)
ama<-dataset_varones %>% filter(trabaja==0) %>% count(CAT_INAC) %>%  filter(CAT_INAC==4)
n<-as.numeric(dataset_varones %>% filter(trabaja==0) %>% summarise(n=n()))


jub2<-dataset_mujeres %>% filter(trabaja==0) %>% count(CAT_INAC) %>%  filter(CAT_INAC==1) 
ama2<-dataset_mujeres %>% filter(trabaja==0) %>% count(CAT_INAC) %>%  filter(CAT_INAC==4)
m<-as.numeric(dataset_mujeres %>% filter(trabaja==0) %>% summarise(n=n()))

D <- matrix(data=c(jub$n/n, ama$n/n, jub2$n/m, ama2$n/m),nrow = 2, ncol = 2, byrow = TRUE)

rownames(D) <- list('Varón','Mujer')
colnames(D) <- list('Jubilado', 'Ama de casa')

x = xtable(D)
print(x)

###---------------###  
###----PARTE II######
###---------------###

#Depuramos el entorno y llamamos nuevas librerías
library(corrplot)
#install.packages("tree")
library(tree)
#install.packages("rpart")
library(rpart)
#install.packages("rattle")
library(rattle)
#install.packages("rpart.plot")
library(rpart.plot)
#install.packages("RGtk2")
library(RGtk2)
#install.packages("corrr")
library(corrr)

rm(ama, ama2, C, D, dataset5, jub, jub2, x, a11, a12, a21, a22, a31, a32, a41, a42, n, m, df)

#1. Prediciendo participación laboral-----
#Convierto la base en numerica para analizar correlaciones
sapply(dataset, is.numeric)
datasetnum <- dataset[, sapply(dataset, is.numeric)]

#Analizo correlaciones:
dataset.corr=cor(datasetnum[,])
cor<-datasetnum %>% correlate() %>% focus(trabaja)
cor
correlaciones <- filter(cor, trabaja >= 0.7 | trabaja <= -0.7)
correlaciones

Trabaja=ifelse(datasetnum$trabaja==1,"Si","No")
datasetnum=data.frame(datasetnum,Trabaja)

#Elimino variables que se usan para construir trabaja y aquellas que no tienen variabilidad:
datasetnum <- dplyr::select(datasetnum,-trabaja, -CAT_OCUP, - P21, -DECOCUR, -IDECOCUR,-RDECOCUR,-ADECOCUR, -DECINDR,-IDECINDR,-RDECINDR,-ADECINDR, -PONDIH, -COMPONENTE, -V2_M, -PONDII, -PONDIIO, -ESTADO, -PP02H, -PP02I, -PP02C1, -PP02C2, -PP02C3, -PP02C4, -PP02C5, -PP02C6, -PP02C7, -PP02C8, -PP02E, -PP02H, -PP02I, -CAT_INAC, -P21,-PP03C,-PP03D,-PP3E_TOT,-PP3F_TOT,-PP03G,-PP03H,-PP03I,-PP03J,-INTENSI,-PP04A,-PP04B_COD,-PP04B1,-PP04B2,-PP04B3_MES, -PP04B3_DIA, -PP04B3_ANO,-PP04C,-PP04C99,-PP04G,-PP05B2_MES,-PP05B2_ANO,-PP05B2_DIA,-PP05C_1,-PP05C_2,-PP05C_3,-PP05E,-PP05F,-PP05H,-PP06A,-PP06C,-PP06D,-PP06E,-PP06H,-PP07A,-PP07C,-PP07D,-PP07E,-PP07F1,-PP07F2,-PP07F3,-PP07F4,-PP07F5,-PP07G1,-PP07G2,-PP07G3,-PP07G4,-PP07G_59,-PP07H,-PP07I,-PP07J,-PP07K,-PP08D1,-PP08D4,-PP08F1,-PP08F2,-PP08J1,-PP08J2,-PP08J3,-PP09A,-PP09B,-PP09C,-PP10A,-PP10C,-PP10D,-PP10E,-PP11A, -PP11B_COD,-PP11B1,-PP11B2_MES,-PP11B2_ANO,-PP11B2_DIA,-PP11C,-PP11C99,-PP11D_COD,-PP11G_ANO,-PP11G_MES,-PP11G_DIA,-PP11L1,-PP11L,-PP11M,-PP11N,-PP11O,-PP11P,-PP11Q,-PP11R,-PP11S,-PP11T,-P21,-TOT_P12,-P47T,-ANO4, -T_VI, -TRIMESTRE, -H15)

###Vuelvo a crear las bases según sexo
datasetnum_mujeres <-datasetnum[datasetnum$CH04==2,]
datasetnum_varones <-datasetnum[datasetnum$CH04==1,]

#Arbol mujeres
tree.mujeres = rpart(Trabaja~.,datasetnum_mujeres,method = "class")
pdf("tree.mujeres.pdf",height=10,width=10,paper="special")
fancyRpartPlot(tree.mujeres,sub="")
dev.off()

#Arbol varones
tree.varones = rpart(Trabaja~.,datasetnum_varones,method = "class")
pdf("tree.varones.pdf",height=10,width=10,paper="special")
fancyRpartPlot(tree.varones,sub="")
dev.off()

#Arbol para toda la base
tree.datasetnum = rpart(Trabaja~.,datasetnum,method = "class")
pdf("tree.dataset.pdf",height=10,width=10,paper="special")
fancyRpartPlot(tree.datasetnum,sub="")
dev.off()

#4 Predecir para la base de prueba----
set.seed(123) #Fijamos semilla para que sean reproducibles los resultados

#Creo muestras de entrenamiento y test para mujeres
train.mujeres=sample(1:nrow(datasetnum_mujeres), 200)
test.mujeres=datasetnum_mujeres[-train.mujeres,]
Trabaja.test.mujeres=datasetnum_mujeres$Trabaja[-train.mujeres]

tree.pred.mujeres=predict(tree.mujeres,test.mujeres,type="class")

#Matriz de confusión mujeres
x<-table(tree.pred.mujeres,Trabaja.test.mujeres)
X <- xtable(x)
print(X)

#Creo muestras de entrenamiento y test para varones
train.varones=sample(1:nrow(datasetnum_varones), 200)
test.varones=datasetnum_varones[-train.varones,]
Trabaja.test.varones=datasetnum_varones$Trabaja[-train.varones]

tree.pred.varones=predict(tree.varones,test.varones,type="class")

#Matriz de confusión varones
x <- table(tree.pred.varones,Trabaja.test.varones)
X <- xtable(x)
print(X)

#Lo mismo pero para toda la base:

train.dataset=sample(1:nrow(datasetnum), 200)
test.dataset=datasetnum[-train.dataset,]
Trabaja.test.dataset=datasetnum$Trabaja[-train.dataset]

tree.pred=predict(tree.datasetnum,test.dataset,type="class")

table(tree.pred,Trabaja.test.dataset)

#5 Modelo alternativo -----
#Cargo paquetes a utilizar
library(MASS)
library(ISLR)
library(class)
library(Matrix) 
library(foreach)
library(glmnet)

#Creo la dummy trabaja (1,0) a partir de Trabaja (Sí,No)
datasetnum$trabaja=ifelse(datasetnum$Trabaja=="Si",1,0)
datasetnum <- dplyr::select(datasetnum,-Trabaja)

#Identifico columnas con missing values
list_na <- colnames(datasetnum)[ apply(datasetnum, 2, anyNA)]
list_na
#Las elimino 
datasetnum <- dplyr::select(datasetnum,-CH14,-CH15_COD,-CH16_COD,-PP04D_COD,-DECIFR,-IDECIFR,-RDECIFR,-ADECIFR,-DECCFR,-IDECCFR,-RDECCFR,-ADECCFR)
#Chequeo que no queden missing values
sum(is.na(datasetnum))

x=model.matrix(trabaja~.,datasetnum)[,-1]
y=datasetnum$trabaja

#--Ridge--#

set.seed(123) #Fijamos semilla para que sean reproducibles los resultados

cv.out.ridge=cv.glmnet(x,y,alpha=0)

# Gráfico del ECM y ln de lambda
plot(cv.out.ridge)

bestlam.ridge=cv.out.ridge$lambda.min
bestlam.ridge
# El lambda que resulta en el menor error de cross validation es 0.09.

# El ECM para lambda=0.09 es:
ridge.pred=predict(cv.out.ridge,s=bestlam.ridge,newx=x)
mean((ridge.pred-y)^2)
# igual a 0.1848127

# Estimamos el modelo de regresion de ridge con lambda=0.09
ridge_coef = predict(cv.out.ridge,type="coefficients",s=bestlam.ridge)[1:35,]

#Clasifico
ridge.pred.clas <- ifelse(ridge.pred > 0.5,1,0)

#Matriz de confusión de Ridge
x <- table(y,ridge.pred.clas)
X <- xtable(x)
print(X)

#6 Participación laboral-----
#Arbol para toda la base
tree.datasetnum = rpart(trabaja~.,datasetnum,method = "class")
pdf("tree.dataset.pdf",height=10,width=10,paper="special")
fancyRpartPlot(tree.datasetnum,sub="")
dev.off()