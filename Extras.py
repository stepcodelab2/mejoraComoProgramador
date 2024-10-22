# Ejercicio 1: Contar el número de vocales en una cadena.
# Solución:
def contar_vocales(cadena):
    return sum(1 for letra in cadena.lower() if letra in 'aeiou')

cadena = "Hola Mundo"
print("Número de vocales:", contar_vocales(cadena))

# Ejercicio 2: Verificar si un número es primo.
# Solución:
def es_primo(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

n = 29
print(f"¿Es {n} primo?:", es_primo(n))

# Ejercicio 3: Invertir una cadena.
# Solución:
def invertir_cadena(cadena):
    return cadena[::-1]

cadena = "Python"
print("Cadena invertida:", invertir_cadena(cadena))

# Ejercicio 4: Calcular la suma de los primeros n números naturales.
# Solución:
def suma_n_naturales(n):
    return n * (n + 1) // 2

n = 10
print("Suma de los primeros", n, "números naturales:", suma_n_naturales(n))

# Ejercicio 5: Generar la tabla de multiplicar de un número.
# Solución:
def tabla_multiplicar(num):
    return [num * i for i in range(1, 11)]

num = 5
print("Tabla de multiplicar de", num, ":", tabla_multiplicar(num))

# Ejercicio 6: Comprobar si una cadena es un palíndromo.
# Solución:
def es_palindromo(cadena):
    return cadena == cadena[::-1]

cadena = "anilina"
print(f"¿Es '{cadena}' un palíndromo?:", es_palindromo(cadena))

# Ejercicio 7: Obtener el máximo y el mínimo de una lista.
# Solución:
def max_min_lista(lista):
    return max(lista), min(lista)

lista = [3, 5, 1, 8, 2]
maximo, minimo = max_min_lista(lista)
print("Máximo:", maximo, "Mínimo:", minimo)

# Ejercicio 8: Eliminar duplicados de una lista.
# Solución:
def eliminar_duplicados(lista):
    return list(set(lista))

lista = [1, 2, 2, 3, 4, 4, 5]
print("Lista sin duplicados:", eliminar_duplicados(lista))

# Ejercicio 9: Contar las palabras en una cadena.
# Solución:
def contar_palabras(cadena):
    return len(cadena.split())

cadena = "Este es un ejemplo de cadena"
print("Número de palabras:", contar_palabras(cadena))

# Ejercicio 10: Generar una lista de números Fibonacci hasta n.
# Solución:
def fibonacci_hasta_n(n):
    fib = [0, 1]
    while fib[-1] + fib[-2] <= n:
        fib.append(fib[-1] + fib[-2])
    return fib[:-1]

n = 100
print("Números Fibonacci hasta", n, ":", fibonacci_hasta_n(n))

# Ejercicio 11: Buscar el elemento más frecuente en una lista.
# Solución:
from collections import Counter

def elemento_frecuente(lista):
    conteo = Counter(lista)
    return conteo.most_common(1)[0]

lista = [1, 2, 2, 3, 3, 3, 4]
elemento, frecuencia = elemento_frecuente(lista)
print("Elemento más frecuente:", elemento, "Frecuencia:", frecuencia)

# Ejercicio 12: Encontrar la suma de dígitos de un número.
# Solución:
def suma_digitos(n):
    return sum(int(digito) for digito in str(n))

n = 12345
print("Suma de dígitos de", n, ":", suma_digitos(n))

# Ejercicio 13: Comprobar si dos cadenas son anagramas.
# Solución:
def son_anagramas(cadena1, cadena2):
    return sorted(cadena1) == sorted(cadena2)

cadena1 = "listen"
cadena2 = "silent"
print(f"¿'{cadena1}' y '{cadena2}' son anagramas?:", son_anagramas(cadena1, cadena2))

# Ejercicio 14: Encontrar el segundo mayor número en una lista.
# Solución:
def segundo_mayor(lista):
    return sorted(set(lista))[-2]

lista = [1, 2, 3, 4, 5]
print("Segundo mayor número:", segundo_mayor(lista))

# Ejercicio 15: Rotar una lista a la derecha n posiciones.
# Solución:
def rotar_lista(lista, n):
    n = n % len(lista)
    return lista[-n:] + lista[:-n]

lista = [1, 2, 3, 4, 5]
n = 2
print("Lista rotada:", rotar_lista(lista, n))

# Ejercicio 16: Obtener la media, mediana y moda de una lista.
# Solución:
def media_mediana_moda(lista):
    media = sum(lista) / len(lista)
    mediana = sorted(lista)[len(lista) // 2]
    moda = Counter(lista).most_common(1)[0][0]
    return media, mediana, moda

lista = [1, 2, 2, 3, 4, 5]
media, mediana, moda = media_mediana_moda(lista)
print("Media:", media, "Mediana:", mediana, "Moda:", moda)

# Ejercicio 17: Implementar la búsqueda de un elemento en una lista.
# Solución:
def busqueda_elemento(lista, x):
    return x in lista

lista = [1, 2, 3, 4, 5]
x = 3
print(f"¿Está {x} en la lista?:", busqueda_elemento(lista, x))

# Ejercicio 18: Calcular el factorial de un número de manera recursiva.
# Solución:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

n = 5
print("Factorial de", n, "es:", factorial(n))

# Ejercicio 19: Crear una función que devuelva el número de elementos únicos en una lista.
# Solución:
def elementos_unicos(lista):
    return len(set(lista))

lista = [1, 2, 2, 3, 4]
print("Número de elementos únicos:", elementos_unicos(lista))

# Ejercicio 20: Comprobar si una lista está ordenada.
# Solución:
def esta_ordenada(lista):
    return lista == sorted(lista)

lista = [1, 2, 3, 4, 5]
print("¿Está la lista ordenada?:", esta_ordenada(lista))

# Ejercicio 21: Implementar un algoritmo de búsqueda lineal.
# Solución:
def busqueda_lineal(lista, x):
    for i, elemento in enumerate(lista):
        if elemento == x:
            return i
    return -1

lista = [1, 2, 3, 4, 5]
x = 3
print("Índice de", x, "es:", busqueda_lineal(lista, x))

# Ejercicio 22: Crear una función que verifique si una cadena tiene caracteres únicos.
# Solución:
def tiene_caracteres_unicos(cadena):
    return len(cadena) == len(set(cadena))

cadena = "abcde"
print(f"¿'{cadena}' tiene caracteres únicos?:", tiene_caracteres_unicos(cadena))

# Ejercicio 23: Implementar un algoritmo de selección (selection sort).
# Solución:
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

arr = [64, 25, 12, 22, 11]
print("Arreglo ordenado (Selection Sort):", selection_sort(arr))

# Ejercicio 24: Implementar un algoritmo de ordenación por inserción (insertion sort).
# Solución:
def insertion_sort(arr):
    for i in range(1, len(arr)):
        clave = arr[i]
        j = i - 1
        while j >= 0 and clave < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = clave
    return arr

arr = [12, 11, 13, 5, 6]
print("Arreglo ordenado (Insertion Sort):", insertion_sort(arr))

# Ejercicio 25: Implementar el algoritmo de ordenación rápida (quick sort).
# Solución:
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivote = arr[len(arr) // 2]
    izquierda = [x for x in arr if x < pivote]
    medio = [x for x in arr if x == pivote]
