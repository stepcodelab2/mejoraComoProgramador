# Ejercicio 1: Crear una función que devuelva los divisores de un número.
# Solución:
def divisores(n):
    return [i for i in range(1, n+1) if n % i == 0]

print(divisores(28))

# Ejercicio 2: Implementar el algoritmo de ordenamiento burbuja.
# Solución:
def burbuja(lista):
    n = len(lista)
    for i in range(n):
        for j in range(0, n-i-1):
            if lista[j] > lista[j+1]:
                lista[j], lista[j+1] = lista[j+1], lista[j]
    return lista

print(burbuja([64, 34, 25, 12, 22, 11, 90]))

# Ejercicio 3: Encontrar el elemento más frecuente en una lista.
# Solución:
from collections import Counter
def mas_frecuente(lista):
    frecuencia = Counter(lista)
    return max(frecuencia, key=frecuencia.get)

print(mas_frecuente([1, 3, 3, 2, 1, 3, 4, 2, 1]))

# Ejercicio 4: Crear una función que devuelva los números primos hasta un límite dado.
# Solución:
def primos_hasta(n):
    primos = []
    for num in range(2, n+1):
        es_primo = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                es_primo = False
                break
        if es_primo:
            primos.append(num)
    return primos

print(primos_hasta(50))

# Ejercicio 5: Implementar el algoritmo de ordenamiento por inserción.
# Solución:
def insercion(lista):
    for i in range(1, len(lista)):
        key = lista[i]
        j = i - 1
        while j >= 0 and key < lista[j]:
            lista[j + 1] = lista[j]
            j -= 1
        lista[j + 1] = key
    return lista

print(insercion([12, 11, 13, 5, 6]))

# Ejercicio 6: Crear una función que convierta un número decimal a binario.
# Solución:
def decimal_a_binario(n):
    return bin(n)[2:]

print(decimal_a_binario(233))

# Ejercicio 7: Implementar una búsqueda binaria recursiva.
# Solución:
def busqueda_binaria_recursiva(lista, valor, inicio, fin):
    if inicio > fin:
        return -1
    medio = (inicio + fin) // 2
    if lista[medio] == valor:
        return medio
    elif lista[medio] < valor:
        return busqueda_binaria_recursiva(lista, valor, medio + 1, fin)
    else:
        return busqueda_binaria_recursiva(lista, valor, inicio, medio - 1)

lista = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(busqueda_binaria_recursiva(lista, 6, 0, len(lista)-1))

# Ejercicio 8: Encontrar el máximo común divisor (MCD) de una lista de números.
# Solución:
import math
def mcd_lista(lista):
    return math.gcd(*lista)

print(mcd_lista([24, 36, 48]))

# Ejercicio 9: Implementar el algoritmo de ordenamiento por selección.
# Solución:
def seleccion(lista):
    for i in range(len(lista)):
        minimo = i
        for j in range(i+1, len(lista)):
            if lista[j] < lista[minimo]:
                minimo = j
        lista[i], lista[minimo] = lista[minimo], lista[i]
    return lista

print(seleccion([29, 10, 14, 37, 14]))

# Ejercicio 10: Crear una función que verifique si dos cadenas son anagramas.
# Solución:
def son_anagramas(cadena1, cadena2):
    return sorted(cadena1) == sorted(cadena2)

print(son_anagramas("listen", "silent"))

# Ejercicio 11: Crear una función que devuelva el mínimo común múltiplo (MCM) de dos números.
# Solución:
def mcm(a, b):
    return abs(a*b) // math.gcd(a, b)

print(mcm(15, 20))

# Ejercicio 12: Implementar el algoritmo quicksort.
# Solución:
def quicksort(lista):
    if len(lista) <= 1:
        return lista
    pivote = lista[len(lista) // 2]
    izquierda = [x for x in lista if x < pivote]
    centro = [x for x in lista if x == pivote]
    derecha = [x for x in lista if x > pivote]
    return quicksort(izquierda) + centro + quicksort(derecha)

print(quicksort([3, 6, 8, 10, 1, 2, 1]))

# Ejercicio 13: Implementar el algoritmo de búsqueda en profundidad (DFS) para grafos.
# Solución:
def dfs(grafo, nodo, visitados=None):
    if visitados is None:
        visitados = set()
    visitados.add(nodo)
    print(nodo)
    for vecino in grafo[nodo]:
        if vecino not in visitados:
            dfs(grafo, vecino, visitados)

grafo = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
dfs(grafo, 'A')

# Ejercicio 14: Implementar el algoritmo de búsqueda en anchura (BFS) para grafos.
# Solución:
from collections import deque
def bfs(grafo, nodo):
    visitados = set()
    cola = deque([nodo])
    while cola:
        actual = cola.popleft()
        if actual not in visitados:
            print(actual)
            visitados.add(actual)
            cola.extend(grafo[actual])

bfs(grafo, 'A')

# Ejercicio 15: Crear una función que calcule la potencia de un número de forma recursiva.
# Solución:
def potencia(base, exponente):
    if exponente == 0:
        return 1
    else:
        return base * potencia(base, exponente - 1)

print(potencia(2, 3))

# Ejercicio 16: Crear una función que encuentre el k-ésimo elemento más pequeño en una lista.
# Solución:
def k_esimo_menor(lista, k):
    lista.sort()
    return lista[k-1]

print(k_esimo_menor([7, 10, 4, 3, 20, 15], 3))

# Ejercicio 17: Implementar una cola utilizando listas.
# Solución:
class Cola:
    def __init__(self):
        self.cola = []
    
    def encolar(self, valor):
        self.cola.append(valor)
    
    def desencolar(self):
        if len(self.cola) > 0:
            return self.cola.pop(0)
        else:
            return None

cola = Cola()
cola.encolar(1)
cola.encolar(2)
print(cola.desencolar())
print(cola.desencolar())

# Ejercicio 18: Crear una función que verifique si una matriz es simétrica.
# Solución:
def es_simetrica(matriz):
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            if matriz[i][j] != matriz[j][i]:
                return False
    return True

matriz = [[1, 2, 3], [2, 5, 6], [3, 6, 9]]
print(es_simetrica(matriz))

# Ejercicio 19: Implementar el algoritmo merge sort.
# Solución:
def merge_sort(lista):
    if len(lista) > 1:
        medio = len(lista) // 2
        izquierda = lista[:medio]
        derecha = lista[medio:]

        merge_sort(izquierda)
        merge_sort(derecha)

        i = j = k = 0

        while i < len(izquierda) and j < len(derecha):
            if izquierda[i] < derecha[j]:
                lista[k] = izquierda[i]
                i += 1
            else:
                lista[k] = derecha[j]
                j += 1
            k += 1

        while i < len(izquierda):
            lista[k] = izquierda[i]
            i += 1
            k += 1

        while j < len(derecha):
            lista[k] = derecha[j]
            j += 1
            k += 1

print(merge_sort([12, 11, 13, 5, 6, 7]))

# Ejercicio 20: Crear una función que encuentre la subcadena más larga sin caracteres repetidos.
# Solución:
def subcadena_sin_repetidos(cadena):
    max_len = 0
    subcadena = ""
    for i in range(len(cadena)):
        subcadena_temp = ""
        for j in cadena[i:]:
            if j not in subcadena_temp:
                subcadena_temp += j
            else:
                break
        if len(subcadena_temp) > max_len:
            max_len = len(subcadena_temp)
            subcadena = subcadena_temp
    return subcadena

print(subcadena_sin_repetidos("abcabcbb"))

# Ejercicio 21: Implementar una pila utilizando listas.
# Solución:
class Pila:
    def __init__(self):
        self.pila = []
    
    def apilar(self, valor):
        self.pila.append(valor)
    
    def desapilar(self):
        if len(self.pila) > 0:
            return self.pila.pop()
        else:
            return None

pila = Pila()
pila.apilar(1)
pila.apilar(2)
print(pila.desapilar())
print(pila.desapilar())

# Ejercicio 22: Encontrar la subsecuencia común más larga entre dos cadenas.
# Solución:
def subsecuencia_comun_larga(X, Y):
    m = len(X)
    n = len(Y)
    dp = [[None]*(n+1) for i in range(m+1)]
  
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

print(subsecuencia_comun_larga("AGGTAB", "GXTXAYB"))

# Ejercicio 23: Crear una función que devuelva la matriz transpuesta.
# Solución:
def transpuesta(matriz):
    return [[matriz[j][i] for j in range(len(matriz))] for i in range(len(matriz[0]))]

matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(transpuesta(matriz))

# Ejercicio 24: Crear una función que encuentre el número de ocurrencias de una subcadena en una cadena.
# Solución:
def contar_subcadena(cadena, subcadena):
    return cadena.count(subcadena)

print(contar_subcadena("banana", "ana"))

# Ejercicio 25: Implementar una cola de prioridad usando listas.
# Solución:
import heapq
class ColaPrioridad:
    def __init__(self):
        self.cola = []
    
    def encolar(self, prioridad, valor):
        heapq.heappush(self.cola, (prioridad, valor))
    
    def desencolar(self):
        return heapq.heappop(self.cola)[1]

cola = ColaPrioridad()
cola.encolar(1, "Tarea 1")
cola.encolar(3, "Tarea 3")
cola.encolar(2, "Tarea 2")
print(cola.desencolar())
print(cola.desencolar())
