# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 18:30:35 2020

@author: Ovi
"""
#Big Data - Trabajo Práctico 2
#Natalia Pecorari, Lautaro Aguiar, Nicolás Oviedo

#Ejercicio 1

import csv
import tweepy

#Credenciales de Twitter
consumer_key = '2MlfpmBC4d8zzz1cZxM1DhxMm'
consumer_secret = 'QxcmeHECRZhGRZ0HqkNglxT5SOAg3y4jRnGaGw6j2fxOmM3f16'
access_key = "1158370347089113088-dTsbvEWhDnKBwSpSXG9MnHHIn7HKHl"
access_secret = "0QxtTRAfl6FLuIuBQxFUPQ3JxdnyBfjuGcZkDUowIYW7S"

#Buscamos los útimos tweets de alguien
def get_tweets(username):
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    number_of_tweets = 75

    #Ponemos los tweets en una lista
    tweets_for_csv = []
    for tweet in tweepy.Cursor(api.user_timeline, screen_name = username).items(number_of_tweets):
        #Armo una matriz con la información: 
        #                       usuario        fecha                 favs                texto
        tweets_for_csv.append([username, tweet.created_at, tweet.favorite_count, tweet.text.encode("utf-8")]) 

    # Lo paso a un archivo csv
    outfile = username + "_tweets.csv" # string variable
    print("Escribiendo en " + outfile)
      
    with open(outfile, 'w+') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(tweets_for_csv)
        
users = ['wsosaescudero']

for user in users:
    get_tweets(user)

#Ejercicio 2
    
from bs4 import BeautifulSoup, Comment
import requests

user_agent_phone = 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_1 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9B179 Safari/7534.48.3'

headers = { 'User-Agent': user_agent_phone}

handle = input('Nombre de la cuenta de Twitter: ')
temp = requests.get('https://twitter.com/'+handle, headers=headers)
bs = BeautifulSoup(temp.text,'lxml')

try:
    mat = bs.find_all('div',{'class':'statnum'})
    seg1 = mat[1].text
    seg2 = mat[2].text
    
    var = bs.find('div',{'class':'bio'}).text
    var = var.rstrip()
    var = var.lstrip()

    print('''\n\n{} sigue a {} cuentas.
          \n{} tiene {} seguidores.
          \nLa biografía de {} es {}'''.format(handle,seg1, handle, seg2, handle, var))

except:
    print('Nombre de cuenta no encontrado.')
    
#Ejercicio 3
    
import pandas as pd
import os
import urllib.request

dir='C:\\Users\\Ovi\\Desktop\\Big Data\\Tutorial2_Big_Data'
os.getcwd()  
os.chdir(dir)

file = 'fruits_vegsdf.xlsx' #Asigna el archivo mencionado a la variable file
sheetname="Sheet1" #Asigna la hoja 1 de ese archivo a la variable sheetname

os.chdir(dir) #Cambiamos el directorio de trabajo a lo asignado más arriba a dir
cwd = os.getcwd() #Chequeamos el directorio de trabajo y lo alojamos en cwd
cwd #Mostramos en la consola el directorio de trabajo

#Creamos dos listas vacías
ylist = []
wlist = []

xl = pd.ExcelFile(file) #Transforma el Excel en un data frame y lo guarda en la variable xl
df = xl.parse(sheetname) #Del Excel, toma la hoja 1, la transforma en un data frame y lo guarda en df

for index, row in df.iterrows(): #Iniciamos un loop para iterar fila por fila de la base de datos
    urlpage =  row['link'] #Toma la url de la columna link y la asigna a la variable urlpage
    
    try: 
        page = urllib.request.urlopen(urlpage) #Abre la url contenida en urlpage, transforma el resultado en un objeto de Python y lo asigna a la variable page
        soup = BeautifulSoup(page, 'html.parser') #Transforma el HTML contenido en page en una estructura de datos a la cual se le pueden aplicar los métodos de BS para explotar la información
           
        y = soup.find('div',{'class':'skuReference'}).text  #Busca el código de producto y lo guarda en el objeto y       
        w = soup.find('span',{'class':'brand'}).text #Busca la marca del producto y la guarda en la variable w         
    except: #Si no se encuentra marca o código de producto, insertar "None"
        y = 'None'
        w = 'None'
   #Ir agregando cada uno de los resultados, guardados en w,y, en las listas w e y   
    ylist.append(y)
    wlist.append(w)

df['Y'] = ylist #Insertamos en la base de datos de la cual tomamos las URLs una nueva columna llamada Y con los códigos de cada producto
df['W'] = wlist #Insertamos en la base de datos de la cual tomamos las URLs una nueva columna llamada W con las marcas de cada producto
df = df[['nombre', 'Y', 'W', 'precio', 'link']] #Reordenamos las columnas de la base de datos
writer = pd.ExcelWriter('fruit and vegs results.xlsx', engine='xlsxwriter') #Se crea un Excel vacío que se guarda en la variable writer
df.drop_duplicates(keep='first')  .to_excel(writer, sheet_name='Sheet1', index=False) #Se eliminan los duplicados de la base y se guarda su contenido en el Excel creado en la línea anterior
writer.save() #Se guardan todos los cambios