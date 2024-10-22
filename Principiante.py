# Ejercicio 1: Imprimir los números del 1 al 10.
# Solución:
for i in range(1, 11):
    print(i)

# Ejercicio 2: Imprimir los números pares del 1 al 20.
# Solución:
for i in range(2, 21, 2):
    print(i)

# Ejercicio 3: Encontrar la suma de los primeros 100 números naturales.
# Solución:
suma = sum(range(1, 101))
print(suma)

# Ejercicio 4: Verificar si un número es par o impar.
# Solución:
num = int(input("Introduce un número: "))
if num % 2 == 0:
    print(f"{num} es par")
else:
    print(f"{num} es impar")

# Ejercicio 5: Imprimir los primeros 10 números de la secuencia de Fibonacci.
# Solución:
a, b = 0, 1
for i in range(10):
    print(a)
    a, b = b, a + b

# Ejercicio 6: Calcular el factorial de un número dado.
# Solución:
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

num = int(input("Introduce un número: "))
print(factorial(num))

# Ejercicio 7: Encontrar el número más grande en una lista.
# Solución:
lista = [4, 7, 1, 9, 12, 3]
print(max(lista))

# Ejercicio 8: Invertir una cadena de texto.
# Solución:
cadena = "Hola mundo"
print(cadena[::-1])

# Ejercicio 9: Comprobar si una cadena de texto es un palíndromo.
# Solución:
cadena = input("Introduce una cadena: ").lower().replace(" ", "")
if cadena == cadena[::-1]:
    print("Es un palíndromo")
else:
    print("No es un palíndromo")

# Ejercicio 10: Encontrar el número menor en una lista.
# Solución:
lista = [10, 3, 5, 7, 2, 9]
print(min(lista))

# Ejercicio 11: Ordenar una lista de números en orden ascendente.
# Solución:
lista = [4, 7, 2, 8, 3]
lista.sort()
print(lista)

# Ejercicio 12: Calcular el promedio de una lista de números.
# Solución:
lista = [10, 20, 30, 40]
promedio = sum(lista) / len(lista)
print(promedio)

# Ejercicio 13: Contar las vocales en una cadena de texto.
# Solución:
cadena = "Programación en Python"
vocales = "aeiou"
contador = 0
for letra in cadena.lower():
    if letra in vocales:
        contador += 1
print(f"Vocales encontradas: {contador}")

# Ejercicio 14: Eliminar duplicados en una lista.
# Solución:
lista = [1, 2, 2, 3, 4, 4, 5]
lista_sin_duplicados = list(set(lista))
print(lista_sin_duplicados)

# Ejercicio 15: Crear una lista de los cuadrados de los números del 1 al 10.
# Solución:
cuadrados = [i**2 for i in range(1, 11)]
print(cuadrados)

# Ejercicio 16: Encontrar el segundo número más grande en una lista.
# Solución:
lista = [10, 20, 4, 45, 99]
lista.sort()
print(lista[-2])

# Ejercicio 17: Crear una función que verifique si un número es primo.
# Solución:
def es_primo(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

num = int(input("Introduce un número: "))
print(es_primo(num))

# Ejercicio 18: Crear una función que encuentre el máximo común divisor (MCD) de dos números.
# Solución:
def mcd(a, b):
    while b:
        a, b = b, a % b
    return a

print(mcd(56, 98))

# Ejercicio 19: Implementar el algoritmo de búsqueda lineal en una lista.
# Solución:
def busqueda_lineal(lista, valor):
    for i in range(len(lista)):
        if lista[i] == valor:
            return i
    return -1

lista = [1, 3, 5, 7, 9]
valor = 5
print(busqueda_lineal(lista, valor))

# Ejercicio 20: Implementar el algoritmo de búsqueda binaria en una lista ordenada.
# Solución:
def busqueda_binaria(lista, valor):
    inicio = 0
    fin = len(lista) - 1
    while inicio <= fin:
        medio = (inicio + fin) // 2
        if lista[medio] == valor:
            return medio
        elif lista[medio] < valor:
            inicio = medio + 1
        else:
            fin = medio - 1
    return -1

lista = [1, 3, 5, 7, 9]
valor = 7
print(busqueda_binaria(lista, valor))

# Ejercicio 21: Crear una función que invierta una lista.
# Solución:
def invertir_lista(lista):
    return lista[::-1]

lista = [1, 2, 3, 4]
print(invertir_lista(lista))

# Ejercicio 22: Contar cuántas veces aparece un número en una lista.
# Solución:
lista = [1, 2, 2, 3, 3, 3, 4]
num = 3
print(lista.count(num))

# Ejercicio 23: Crear una función que devuelva la suma de los dígitos de un número.
# Solución:
def suma_digitos(n):
    suma = 0
    while n:
        suma += n % 10
        n //= 10
    return suma

print(suma_digitos(1234))

# Ejercicio 24: Calcular el área de un círculo dado su radio.
# Solución:
import math
def area_circulo(radio):
    return math.pi * radio ** 2

print(area_circulo(5))

# Ejercicio 25: Crear una función que convierta una cadena a mayúsculas sin usar la función `upper()`.
# Solución:
def convertir_mayusculas(cadena):
    resultado = ""
    for letra in cadena:
        if 'a' <= letra <= 'z':
            resultado += chr(ord(letra) - 32)
        else:
            resultado += letra
    return resultado

print(convertir_mayusculas("hola mundo"))
