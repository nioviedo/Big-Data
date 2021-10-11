# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:22:23 2020

@author: sistemas
"""

#Big Data - Trabajo Práctico 1
#Natalia Pecorari, Lautaro Aguiar, Nicolás Oviedo

#Ejercicio 1

str1 = "Ejercicio 1"
print(str1)
import math
print(math.tan(1))

# Variables numéricas
x = 1
x += 1
print(x*2)
        
for num in range(1, 16):
    if num % 5 == 0 and num % 3 == 0: # si num es divisible por x
       print("Bob Esponja")
    elif num % 3 == 0: # si num es divisible por x
       print("Bob")
    elif num % 5 == 0: # si num es divisible por x
       print("Esponja")   
    else:
        print(num)

#Ejercicio 2
        
print("\nEjercicio 2")      
lista = ["a","b","c"]  
print(lista)
lista.insert(1,"d")
lista.remove("b")
print(lista)

X = [i for i in range(100)]  #Contiene los enteros desde el 0 al 99
print(X)

n = 10000
i = 0
x = 2**i
 
while x < n:
    print(x)
    i += 1
    x = 2**i

def funcion(a,b):
    if a % 2 == 0:
        print(a+2*b)
    elif a % 2 != 0:
        print(a+3*b)
        
a = int(input("Inserte a: "))
b = int(input("Inserte b: "))        

funcion(a,b)

#Ejercicio 3

print("/nEjercicio 3")
mascotas_autorizadas = ["perro","gato","hamster","loro"]

print("¡Bienvenido a calle imaginaria 1450! ¿Qué mascota tiene?")
pregunta = input("Inserte su respuesta: ")

if pregunta in mascotas_autorizadas:
     print("¡Bienvenido a su "+pregunta+" también!")
     
else:
     print("\nLo sentimos, su "+pregunta+" no es bienvenida")

#Ejercicio 4
     
print("/nEjercicio 4")

año= int(input("Inserte un año para saber si es bisiesto: "))

def año_bisiesto(año):
    if año % 4 == 0 and año % 100 != 0 or año % 400 == 0:
        print(f"El año {año} es bisiesto")
    else:
        print(f"El año {año} no es bisiesto")
        
año_bisiesto(año)
        