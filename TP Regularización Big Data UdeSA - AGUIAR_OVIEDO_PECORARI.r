#----------------BIG DATA UDESA 2020-----------------#
#----------------Trabajo pr�ctico 2------------------#
#Por Lautaro Aguiar, Natalia Pecorari y Nicol�s Oviedo


# Limpiamos el entorno, fijamos el directorio de trabajo e invocamos las librer�as que necesitamos
rm(list=ls())
gc()
#dataDir <- "C:/Users/sistemas/Desktop/TP2 Big Data"
#dataDir <- "C:/Users/Ovi/Desktop/Big Data/TP2_Big_Data"
dataDir <- "C:/Users/Lautaro/Downloads/Downloads/UdeSA/Big Data/TP/TP2"
setwd(dataDir)

library(readxl)
library(xlsx)
library(ggplot2)
library(dplyr)
library(xtable)
library(scales)
library(stargazer)


### --- PARTE I --- ###


#2. Descargando la base

aprender <- read.csv2("Aprender-2018-primaria-6.csv")#, header = TRUE, sep = ",")
View(aprender)
diccionario <- read_xlsx("aprender2018-diccionario-primaria-6.xlsx")

#(a) Hallando missing values
apply(is.na(aprender),2,sum) #Cuenta missing values por columna
missing <- apply(is.na(aprender),2,sum) 
stargazer(missing)
#Las variables SIN missing values son sector, ambito, cod_provincia, isocioa
#Puntaje en mate = mpuntaje

c<-is.na(aprender$mpuntaje)
summary(c)
# Hay 28490 missing values en Puntaje en Matem�tica

#(b) Tratamos missing values
#Eliminamos los missing values porque asumimos que no faltan por un criterio que sesgue la muestra/no hay un mecanismo de seleccion en las faltan sino que son aleatorios y no nos preocupa en principio tanto la insesgadez. No ponemos 0 porque eso sesga la muestra. 

aprender <- na.omit(aprender)
apply(is.na(aprender),2,sum)

#(c) Exportamos el diccionario a Excel
diccionario <- diccionario[,-3:-4]
diccionario <- na.omit(diccionario)
colnames(diccionario) <- c("Variable", "Etiqueta")

diccionario <- as.data.frame(diccionario)

write.xlsx(diccionario, "Diccionario.xlsx", row.names = FALSE)

#(d) Renombramos columnas
diccionario <- read_excel("Diccionario.xlsx")

diccionario <- as.data.frame(diccionario)

h <- colnames(aprender)

names(aprender)[match(diccionario[,"Variable"], names(aprender))] = diccionario[,"Etiqueta"]

#3. Distribuci�n de puntaje por g�nero
# Sexo: 1 = Hombre
aprender1 <- aprender[aprender$Sexo > 0, ] #Nos quedamos con las observaciones cuyo g�nero es hombre o mujer
aprender1$Sexo <- factor(aprender1$Sexo, labels = c("Var�n", "Mujer"))

#Puntaje en Lengua
ggplot(aprender1, aes(x = `Puntaje en Lengua`)) +
  geom_histogram(aes(color = Sexo, fill = Sexo), 
                 position = "identity", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#bb0080")) +
  scale_fill_manual(values = c("#00AFBB", "#bb0080")) + scale_y_continuous(name = "Cantidad de alumnos")

#Puntaje en Matem�tica
ggplot(aprender1, aes(x = `Puntaje en Matem�tica`)) +
  geom_histogram(aes(color = Sexo, fill = Sexo), 
                 position = "identity", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#bb0080")) +
  scale_fill_manual(values = c("#00AFBB", "#bb0080")) + scale_y_continuous(name = "Cantidad de alumnos")

#4. Correlaci�n entre puntajes
cor(aprender$'Puntaje en Lengua', aprender$'Puntaje en Matem�tica')
#Correlaci�n = 0.6319223

#5. Gr�fico de torta
count(aprender,`Indice socioecon�mico del alumno`)
aprender1 <- aprender[aprender$`Indice socioecon�mico del alumno` > 0, ] #Nos quedamos con las observaciones cuyo �ndice socioecon�mico es mayor a 0, el resto no son interpretables
df<-count(aprender1,`Indice socioecon�mico del alumno`)

df$`Indice socioecon�mico del alumno` <- factor(df$`Indice socioecon�mico del alumno`, labels = c("Bajo", "Medio", "Alto"))

por=c(percent(df$n/sum(df$n)))
prop=c(round(df$n/sum(df$n)*100,2))

df<-cbind(df, por, prop)

df<- df %>%
  arrange(desc(`Indice socioecon�mico del alumno`)) %>%
  mutate(lab.ypos = cumsum(prop) - 0.5*(prop))
df

mycols <- c("#01B8AA", "#374649", "#8AD4EB", "#A66999")
ggplot(df, aes(x = "", y = prop, fill = `Indice socioecon�mico del alumno`)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  geom_text(aes(y = lab.ypos, label = prop), color = "white")+
  scale_fill_manual(values = mycols) +
  theme_void()

#6. Nivel socioecon�mico por provincia
aprender1$`N�mero de jurisdicci�n`
aprender1$`N�mero de jurisdicci�n` <- factor(aprender1$`N�mero de jurisdicci�n`) 

# Generamos la tabla
lvlprov<- aprender1 %>% group_by(`N�mero de jurisdicci�n`) %>% summarise(Alumnos = n(),cond1=sum(`Indice socioecon�mico del alumno`==1), cond2=sum(`Indice socioecon�mico del alumno`==2), cond3=sum(`Indice socioecon�mico del alumno`==3), por1=percent(cond1/n()), por2 = percent(cond2/n()), por3=percent(cond3/n()))
lvlprov

# Agregamos etiqueta provincial
diccionario_prov <- read_xlsx("aprender2018-diccionario-primaria-6.xlsx")
diccionario_prov <- diccionario_prov[528:551,-1:-2]
colnames(diccionario_prov) <- c("N�mero de jurisdicci�n", "Provincia")

lvlprov$`N�mero de jurisdicci�n` <- diccionario_prov$Provincia

# La compactamos
lvlprov<-lvlprov[,-3:-5]
names(lvlprov) = c("Provincia", "Alumnos", "Bajo", "Medio", "Alto")

# La imprimimos
x <- xtable(lvlprov, caption="Proporci�n de alumnos por cada nivel socioecon�mico por provincia. Excluye aquellas observaciones cuyo nivel socioecon�mico era menor a 0", auto=TRUE)
r <- print(x, include.rownames=FALSE)



### --- PARTE II --- ###


library(MASS)
library(ISLR)
library(class)
library(Matrix) 
library(foreach)
library(glmnet)
library(RColorBrewer)
library(boot)

rm(c,aprender1,df,x,diccionario,diccionario_prov,lvlprov) #Descartamos objetos que ya no usaremos para optimizar el uso de memoria de R

# Creamos una nueva base de datos llamada aprender2 y eliminamos la columna Nivel de desempe�o en Matem�tica

aprender2 <- aprender[aprender$`Indice socioecon�mico del alumno` > 0, ]
aprender2 <- aprender2[aprender2$Sexo > 0, ]
#D ecidimos para la parte 2 eliminar las observaciones cuyo �ndice econ�mico y cuyo sexo no ten�an una interpretaci�n aparente

aprender2$`Nivel de desempe�o en Matem�tica` <- NULL 

#2. Renombro como Y a la variable dependiente que quiero estimar, en este caso Puntaje en Matem�tica y a X como la matriz que incluye el resto de variables que voy a utilizar para estimarlo
x=model.matrix(`Puntaje en Matem�tica`~.,aprender2)[,-1]
y=aprender2$`Puntaje en Matem�tica`

#--RIDGE--#

set.seed(101) #Fijamos semilla para que sean reproducibles los resultados

# Estimo utilizando esta funci�n que a trav�s de 10 fold CV estima el modelo (alpha = 0 me dice que es RIDGE)
cv.out.ridge=cv.glmnet(x,y,alpha=0)

# Gr�fico del ECM y ln de lambda
plot(cv.out.ridge)

# Lambda �ptimo
bestlam.ridge=cv.out.ridge$lambda.min
bestlam.ridge
# El lambda que resulta en el menor error de cross validation es 6.465789.

# El ECM para lambda=6.465789. es:
ridge.pred=predict(cv.out.ridge,s=bestlam.ridge,newx=x)
mean((ridge.pred-y)^2)
# igual a 5761.638

# Estimamos el modelo de regresion de RIDGE con lambda=6.473327
ridge_coef = predict(cv.out.ridge,type="coefficients",s=bestlam.ridge)[1:123,]
# Ning�n coeficiente se iguala a cero porque Ridge no nos selecciona variables

# Identifico las variables a las que Ridge asigna coeficiente cero:
round(ridge_coef,2)
ridge_coef[ridge_coef == 0]  #efectivamente ninguna tiene coeficiente nulo

# Calculo L2 norm
sqrt(sum(ridge_coef^2))   #da 165.449

#--LASSO--#
set.seed(101) #Fijamos semilla para que sean reproducibles los resultados

# Estimo utilizando esta funci�n que a trav�s de 10 fold CV estima el modelo
cv.out.lasso=cv.glmnet(x,y,alpha=1)

# Gr�fico del ECM y ln de lambda
plot(cv.out.lasso)

# Lambda �ptimo
bestlam.lasso=cv.out.lasso$lambda.min
bestlam.lasso
# El lambda que resulta en el menor error de cross validation es 0.01974559.

# El ECM para lambda=0.01801242 es:
lasso.pred=predict(cv.out.lasso,s=bestlam.lasso,newx=x)
mean((lasso.pred-y)^2)
# igual a 5707.848

# Estimamos el modelo de regresion de LASSO con lambda=0.01801242
lasso_coef = predict(cv.out.lasso,type="coefficients",s=bestlam.lasso)[1:123,]

#3. Predictores descartados

# Identifico las variables a las que LASSO asigna coeficiente cero:
round(lasso_coef,2)
lasso_coef[lasso_coef == 0]
lasso_coef_zero <- data.frame(lasso_coef[lasso_coef == 0]) #Tabla con variables descartadas

l <- xtable(lasso_coef_zero, caption="Coeficientes que Lasso hizo cero", auto=TRUE)
p <- print(l, include.rownames=TRUE)

#4. L2

# Calculo L2 norm de LASSO
sqrt(sum(lasso_coef^2))  #Da 121.9278

#5. Scatter plot LASSO: valores predichos vs. valores ciertos 

df<-cbind.data.frame(aprender2$`Puntaje en Matem�tica`,lasso.pred)
colnames(df)[2]<-"valor predicho"
colnames(df)[1]<-"valor cierto"
plot(df, col="orange")
abline(0, 1)

#ggplot(df, aes(x="valor cierto", y="valor predicho", color="orange")) +
 # geom_point(size=2, shape=23) + geom_rug()

#6. Alumnos por debajo de la media (predicho) 

mean<-mean(lasso.pred)
mean   #la media de los valores predichos es 504.5798
sum(lasso.pred < mean)  #224659 alumnos est�n por debajo de la media seg�n el modelo  

# Alumnos por debajo de la media (cierto)
meanReal<-mean(aprender2$`Puntaje en Matem�tica`)
meanReal
sum(aprender2$`Puntaje en Matem�tica` < meanReal) #236381 alumnos est�n por debajo de la media seg�n datos reales