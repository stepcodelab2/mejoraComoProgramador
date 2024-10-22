# Ejercicio 1: Implementar el algoritmo de búsqueda A*.
# Solución:
from heapq import heappop, heappush

def astar(grafo, inicio, meta, heuristica):
    open_list = []
    heappush(open_list, (0, inicio))
    came_from = {inicio: None}
    g_score = {inicio: 0}

    while open_list:
        _, nodo_actual = heappop(open_list)

        if nodo_actual == meta:
            path = []
            while nodo_actual:
                path.append(nodo_actual)
                nodo_actual = came_from[nodo_actual]
            return path[::-1]

        for vecino, coste in grafo[nodo_actual]:
            tentative_g_score = g_score[nodo_actual] + coste
            if vecino not in g_score or tentative_g_score < g_score[vecino]:
                g_score[vecino] = tentative_g_score
                f_score = tentative_g_score + heuristica[vecino]
                heappush(open_list, (f_score, vecino))
                came_from[vecino] = nodo_actual

grafo = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 1), ('E', 5)],
    'C': [('E', 2)],
    'D': [('F', 2)],
    'E': [('F', 1)],
    'F': []
}
heuristica = {'A': 4, 'B': 2, 'C': 2, 'D': 6, 'E': 1, 'F': 0}
print(astar(grafo, 'A', 'F', heuristica))

# Ejercicio 2: Implementar el algoritmo de Dijkstra para encontrar el camino más corto.
# Solución:
def dijkstra(grafo, inicio):
    import sys
    distancias = {nodo: sys.maxsize for nodo in grafo}
    distancias[inicio] = 0
    pq = [(0, inicio)]
    
    while pq:
        distancia_actual, nodo_actual = heappop(pq)
        
        if distancia_actual > distancias[nodo_actual]:
            continue
        
        for vecino, peso in grafo[nodo_actual]:
            distancia = distancia_actual + peso
            if distancia < distancias[vecino]:
                distancias[vecino] = distancia
                heappush(pq, (distancia, vecino))
    
    return distancias

grafo = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}
print(dijkstra(grafo, 'A'))

# Ejercicio 3: Implementar un sistema de caché LRU (Least Recently Used).
# Solución:
class LRUCache:
    def __init__(self, capacidad):
        self.cache = {}
        self.orden = []
        self.capacidad = capacidad

    def obtener(self, clave):
        if clave in self.cache:
            self.orden.remove(clave)
            self.orden.append(clave)
            return self.cache[clave]
        return -1

    def establecer(self, clave, valor):
        if clave in self.cache:
            self.orden.remove(clave)
        elif len(self.cache) >= self.capacidad:
            clave_vieja = self.orden.pop(0)
            del self.cache[clave_vieja]
        self.cache[clave] = valor
        self.orden.append(clave)

cache = LRUCache(2)
cache.establecer(1, 1)
cache.establecer(2, 2)
print(cache.obtener(1))
cache.establecer(3, 3)
print(cache.obtener(2))

# Ejercicio 4: Implementar el algoritmo de Floyd-Warshall para encontrar caminos mínimos en un grafo.
# Solución:
def floyd_warshall(grafo):
    dist = list(map(lambda i: list(map(lambda j: j, i)), grafo))
    n = len(grafo)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

grafo = [[0, 3, float('inf'), 5],
         [2, 0, float('inf'), 4],
         [float('inf'), 1, 0, float('inf')],
         [float('inf'), float('inf'), 2, 0]]
print(floyd_warshall(grafo))

# Ejercicio 5: Implementar el algoritmo de Bellman-Ford para detección de ciclos negativos.
# Solución:
def bellman_ford(grafo, inicio):
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0

    for _ in range(len(grafo) - 1):
        for nodo in grafo:
            for vecino, peso in grafo[nodo]:
                if distancias[nodo] + peso < distancias[vecino]:
                    distancias[vecino] = distancias[nodo] + peso

    for nodo in grafo:
        for vecino, peso in grafo[nodo]:
            if distancias[nodo] + peso < distancias[vecino]:
                print("Ciclo negativo detectado")
                return

    return distancias

grafo = {
    'A': [('B', -1), ('C', 4)],
    'B': [('C', 3), ('D', 2), ('E', 2)],
    'C': [],
    'D': [('B', 1), ('C', 5)],
    'E': [('D', -3)]
}
print(bellman_ford(grafo, 'A'))

# Ejercicio 6: Implementar el algoritmo de Kruskal para encontrar el árbol de expansión mínima.
# Solución:
class Grafo:
    def __init__(self, vertices):
        self.vertices = vertices
        self.grafo = []

    def agregar_arista(self, u, v, w):
        self.grafo.append([u, v, w])

    def buscar(self, padre, i):
        if padre[i] == i:
            return i
        return self.buscar(padre, padre[i])

    def union(self, padre, rango, x, y):
        raiz_x = self.buscar(padre, x)
        raiz_y = self.buscar(padre, y)

        if rango[raiz_x] < rango[raiz_y]:
            padre[raiz_x] = raiz_y
        elif rango[raiz_x] > rango[raiz_y]:
            padre[raiz_y] = raiz_x
        else:
            padre[raiz_y] = raiz_x
            rango[raiz_x] += 1

    def kruskal(self):
        resultado = []
        i, e = 0, 0
        self.grafo = sorted(self.grafo, key=lambda item: item[2])
        padre = []
        rango = []

        for nodo in range(self.vertices):
            padre.append(nodo)
            rango.append(0)

        while e < self.vertices - 1:
            u, v, w = self.grafo[i]
            i += 1
            x = self.buscar(padre, u)
            y = self.buscar(padre, v)

            if x != y:
                e += 1
                resultado.append([u, v, w])
                self.union(padre, rango, x, y)

        return resultado

g = Grafo(4)
g.agregar_arista(0, 1, 10)
g.agregar_arista(0, 2, 6)
g.agregar_arista(0, 3, 5)
g.agregar_arista(1, 3, 15)
g.agregar_arista(2, 3, 4)

print(g.kruskal())

# Ejercicio 7: Implementar el algoritmo de Prim para el árbol de expansión mínima.
# Solución:
import sys

def prim(grafo):
    key = [sys.maxsize] * len(grafo)
    padre = [None] * len(grafo)
    key[0] = 0
    mstSet = [False] * len(grafo)
    padre[0] = -1

    for _ in range(len(grafo)):
        min_key = sys.maxsize
        u = -1

        for v in range(len(grafo)):
            if not mstSet[v] and key[v] < min_key:
                min_key = key[v]
                u = v

        mstSet[u] = True

        for v, peso in enumerate(grafo[u]):
            if 0 < peso < key[v] and not mstSet[v]:
                key[v] = peso
                padre[v] = u

    return padre

grafo = [[0, 2, 0, 6, 0],
         [2, 0, 3, 8, 5],
         [0, 3, 0, 0, 7],
         [6, 8, 0, 0, 9],
         [0, 5, 7, 9, 0]]

print(prim(grafo))

# Ejercicio 8: Implementar el algoritmo de Edmonds-Karp para encontrar el flujo máximo.
# Solución:
from collections import deque

def bfs(rg, s, t, parent):
    visitado = [False] * len(rg)
    cola = deque([s])
    visitado[s] = True

    while cola:
        u = cola.popleft()

        for v, capacidad in enumerate(rg[u]):
            if not visitado[v] and capacidad > 0:
                parent[v] = u
                if v == t:
                    return True
                cola.append(v)
                visitado[v] = True

    return False

def edmonds_karp(grafo, fuente, sumidero):
    rg = [fila[:] for fila in grafo]
    parent = [-1] * len(grafo)
    flujo_maximo = 0

    while bfs(rg, fuente, sumidero, parent):
        camino_flujo = float('Inf')
        v = sumidero

        while v != fuente:
            u = parent[v]
            camino_flujo = min(camino_flujo, rg[u][v])
            v = parent[v]

        flujo_maximo += camino_flujo
        v = sumidero

        while v != fuente:
            u = parent[v]
            rg[u][v] -= camino_flujo
            rg[v][u] += camino_flujo
            v = parent[v]

    return flujo_maximo

grafo = [[0, 16, 13, 0, 0, 0],
         [0, 0, 10, 12, 0, 0],
         [0, 4, 0, 0, 14, 0],
         [0, 0, 9, 0, 0, 20],
         [0, 0, 0, 7, 0, 4],
         [0, 0, 0, 0, 0, 0]]

print(edmonds_karp(grafo, 0, 5))

# Ejercicio 9: Implementar un algoritmo de backtracking para el problema de la n-reinas.
# Solución:
def es_seguro(tablero, fila, col):
    for i in range(col):
        if tablero[fila][i] == 1:
            return False

    for i, j in zip(range(fila, -1, -1), range(col, -1, -1)):
        if tablero[i][j] == 1:
            return False

    for i, j in zip(range(fila, len(tablero)), range(col, -1, -1)):
        if tablero[i][j] == 1:
            return False

    return True

def resolver_n_reinas(tablero, col):
    if col >= len(tablero):
        return True

    for i in range(len(tablero)):
        if es_seguro(tablero, i, col):
            tablero[i][col] = 1
            if resolver_n_reinas(tablero, col + 1):
                return True
            tablero[i][col] = 0

    return False

def imprimir_solucion(tablero):
    for fila in tablero:
        print(fila)

n = 4
tablero = [[0] * n for _ in range(n)]
if resolver_n_reinas(tablero, 0):
    imprimir_solucion(tablero)

# Ejercicio 10: Implementar el algoritmo de programación dinámica para el problema del corte de varillas.
# Solución:
def corte_varillas(precios, longitud):
    dp = [0] * (longitud + 1)

    for i in range(1, longitud + 1):
        max_valor = -float('inf')
        for j in range(i):
            max_valor = max(max_valor, precios[j] + dp[i-j-1])
        dp[i] = max_valor

    return dp[longitud]

precios = [1, 5, 8, 9, 10, 17, 17, 20]
longitud = 8
print(corte_varillas(precios, longitud))

# Ejercicio 11: Implementar el algoritmo de Ford-Fulkerson para flujo máximo.
# Solución:
def ford_fulkerson(grafo, fuente, sumidero):
    flujo_maximo = 0
    rg = [fila[:] for fila in grafo]

    def dfs(rg, s, t, parent):
        visitado = [False] * len(rg)
        stack = [s]
        visitado[s] = True

        while stack:
            u = stack.pop()
            for v, capacidad in enumerate(rg[u]):
                if not visitado[v] and capacidad > 0:
                    parent[v] = u
                    if v == t:
                        return True
                    stack.append(v)
                    visitado[v] = True
        return False

    parent = [-1] * len(grafo)
    while dfs(rg, fuente, sumidero, parent):
        camino_flujo = float('Inf')
        v = sumidero
        while v != fuente:
            u = parent[v]
            camino_flujo = min(camino_flujo, rg[u][v])
            v = parent[v]

        flujo_maximo += camino_flujo
        v = sumidero
        while v != fuente:
            u = parent[v]
            rg[u][v] -= camino_flujo
            rg[v][u] += camino_flujo
            v = parent[v]

    return flujo_maximo

grafo = [[0, 16, 13, 0, 0, 0],
         [0, 0, 10, 12, 0, 0],
         [0, 4, 0, 0, 14, 0],
         [0, 0, 9, 0, 0, 20],
         [0, 0, 0, 7, 0, 4],
         [0, 0, 0, 0, 0, 0]]

print(ford_fulkerson(grafo, 0, 5))

# Ejercicio 12: Implementar el algoritmo de programación dinámica para el problema de la mochila.
# Solución:
def mochila(pesos, valores, capacidad):
    n = len(valores)
    dp = [[0 for _ in range(capacidad + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacidad + 1):
            if pesos[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - pesos[i - 1]] + valores[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacidad]

pesos = [1, 2, 3, 8, 7, 4]
valores = [20, 5, 10, 40, 15, 25]
capacidad = 10
print(mochila(pesos, valores, capacidad))

# Ejercicio 13: Implementar un árbol AVL y sus operaciones de inserción y eliminación.
# Solución:
class Nodo:
    def __init__(self, clave):
        self.clave = clave
        self.izquierda = None
        self.derecha = None
        self.altura = 1

class AVL:
    def __init__(self):
        self.raiz = None

    def obtener_altura(self, nodo):
        return nodo.altura if nodo else 0

    def obtener_balance(self, nodo):
        return self.obtener_altura(nodo.izquierda) - self.obtener_altura(nodo.derecha)

    def rotar_derecha(self, y):
        x = y.izquierda
        T2 = x.derecha

        x.derecha = y
        y.izquierda = T2

        y.altura = 1 + max(self.obtener_altura(y.izquierda), self.obtener_altura(y.derecha))
        x.altura = 1 + max(self.obtener_altura(x.izquierda), self.obtener_altura(x.derecha))

        return x

    def rotar_izquierda(self, x):
        y = x.derecha
        T2 = y.izquierda

        y.izquierda = x
        x.derecha = T2

        x.altura = 1 + max(self.obtener_altura(x.izquierda), self.obtener_altura(x.derecha))
        y.altura = 1 + max(self.obtener_altura(y.izquierda), self.obtener_altura(y.derecha))

        return y

    def insertar(self, nodo, clave):
        if not nodo:
            return Nodo(clave)

        if clave < nodo.clave:
            nodo.izquierda = self.insertar(nodo.izquierda, clave)
        else:
            nodo.derecha = self.insertar(nodo.derecha, clave)

        nodo.altura = 1 + max(self.obtener_altura(nodo.izquierda), self.obtener_altura(nodo.derecha))

        balance = self.obtener_balance(nodo)

        if balance > 1:
            if clave < nodo.izquierda.clave:
                return self.rotar_derecha(nodo)
            else:
                nodo.izquierda = self.rotar_izquierda(nodo.izquierda)
                return self.rotar_derecha(nodo)

        if balance < -1:
            if clave > nodo.derecha.clave:
                return self.rotar_izquierda(nodo)
            else:
                nodo.derecha = self.rotar_derecha(nodo.derecha)
                return self.rotar_izquierda(nodo)

        return nodo

    def preorden(self, nodo):
        if nodo:
            print(nodo.clave, end=' ')
            self.preorden(nodo.izquierda)
            self.preorden(nodo.derecha)

avl = AVL()
valores = [10, 20, 30, 40, 50, 25]
for valor in valores:
    avl.raiz = avl.insertar(avl.raiz, valor)

print("Recorrido en preorden del árbol AVL:")
avl.preorden(avl.raiz)

# Ejercicio 14: Implementar un algoritmo de búsqueda de caminos más cortos usando Dijkstra.
# Solución:
import heapq

def dijkstra(grafo, fuente):
    n = len(grafo)
    distancias = [float('inf')] * n
    distancias[fuente] = 0
    pq = [(0, fuente)]

    while pq:
        distancia_actual, u = heapq.heappop(pq)

        for v, peso in enumerate(grafo[u]):
            if peso > 0:
                distancia = distancia_actual + peso
                if distancia < distancias[v]:
                    distancias[v] = distancia
                    heapq.heappush(pq, (distancia, v))

    return distancias

grafo = [[0, 7, 9, 0, 0, 14],
         [7, 0, 10, 15, 0, 0],
         [9, 10, 0, 11, 0, 2],
         [0, 15, 11, 0, 6, 0],
         [0, 0, 0, 6, 0, 9],
         [14, 0, 2, 0, 9, 0]]

print(dijkstra(grafo, 0))

# Ejercicio 15: Implementar un algoritmo de ordenación por montículo (heap sort).
# Solución:
def heapify(arr, n, i):
    mayor = i
    izquierda = 2 * i + 1
    derecha = 2 * i + 2

    if izquierda < n and arr[i] < arr[izquierda]:
        mayor = izquierda

    if derecha < n and arr[mayor] < arr[derecha]:
        mayor = derecha

    if mayor != i:
        arr[i], arr[mayor] = arr[mayor], arr[i]
        heapify(arr, n, mayor)

def heap_sort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

arr = [12, 11, 13, 5, 6, 7]
heap_sort(arr)
print("Arreglo ordenado:", arr)

# Ejercicio 16: Implementar el algoritmo de Kruskal para encontrar el árbol de expansión mínima.
# Solución:
class UnionFind:
    def __init__(self, n):
        self.padre = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.padre[u] != u:
            self.padre[u] = self.find(self.padre[u])
        return self.padre[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.padre[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.padre[root_u] = root_v
            else:
                self.padre[root_v] = root_u
                self.rank[root_u] += 1

def kruskal(nodos, aristas):
    aristas.sort(key=lambda x: x[2])
    uf = UnionFind(len(nodos))
    arbol_expansion = []

    for u, v, peso in aristas:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            arbol_expansion.append((u, v, peso))

    return arbol_expansion

nodos = [0, 1, 2, 3]
aristas = [(0, 1, 10), (0, 2, 6), (0, 3, 5),
           (1, 3, 15), (2, 3, 4)]

arbol = kruskal(nodos, aristas)
print("Árbol de expansión mínima:", arbol)

# Ejercicio 17: Implementar un algoritmo de búsqueda en profundidad (DFS).
# Solución:
def dfs(grafo, inicio, visitado=None):
    if visitado is None:
        visitado = set()
    visitado.add(inicio)
    print(inicio, end=' ')

    for vecino in grafo[inicio]:
        if vecino not in visitado:
            dfs(grafo, vecino, visitado)

grafo = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 4],
    3: [1],
    4: [1, 2]
}

print("Recorrido DFS:")
dfs(grafo, 0)

# Ejercicio 18: Implementar un algoritmo de búsqueda en anchura (BFS).
# Solución:
from collections import deque

def bfs(grafo, inicio):
    visitado = set()
    cola = deque([inicio])
    visitado.add(inicio)

    while cola:
        vertice = cola.popleft()
        print(vertice, end=' ')

        for vecino in grafo[vertice]:
            if vecino not in visitado:
                visitado.add(vecino)
                cola.append(vecino)

grafo = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 4],
    3: [1],
    4: [1, 2]
}

print("\nRecorrido BFS:")
bfs(grafo, 0)

# Ejercicio 19: Implementar un algoritmo de mezcla y ordenación (merge sort).
# Solución:
def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m
    L = arr[l:m + 1]
    R = arr[m + 1:r + 1]

    i = j = 0
    k = l

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def merge_sort(arr, l, r):
    if l < r:
        m = (l + r) // 2
        merge_sort(arr, l, m)
        merge_sort(arr, m + 1, r)
        merge(arr, l, m, r)

arr = [38, 27, 43, 3, 9, 82, 10]
merge_sort(arr, 0, len(arr) - 1)
print("Arreglo ordenado:", arr)

# Ejercicio 20: Implementar un algoritmo de programación dinámica para el problema de la suma de subconjuntos.
# Solución:
def suma_subconjuntos(nums, suma):
    n = len(nums)
    dp = [[False] * (suma + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = True

    for i in range(1, n + 1):
        for j in range(1, suma + 1):
            if nums[i - 1] <= j:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    return dp[n][suma]

nums = [3, 34, 4, 12, 5, 2]
suma = 9
print(suma_subconjuntos(nums, suma))

# Ejercicio 21: Implementar un algoritmo de búsqueda binaria.
# Solución:
def busqueda_binaria(arr, x):
    izquierda, derecha = 0, len(arr) - 1
    while izquierda <= derecha:
        medio = (izquierda + derecha) // 2
        if arr[medio] == x:
            return medio
        elif arr[medio] < x:
            izquierda = medio + 1
        else:
            derecha = medio - 1
    return -1

arr = [2, 3, 4, 10, 40]
x = 10
resultado = busqueda_binaria(arr, x)
if resultado != -1:
    print("Elemento encontrado en el índice:", resultado)
else:
    print("Elemento no encontrado.")

# Ejercicio 22: Implementar el algoritmo de búsqueda de la subsecuencia común más larga.
# Solución:
def lcs(x, y):
    m = len(x)
    n = len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

x = "AGGTAB"
y = "GXTXAYB"
print("Longitud de la subsecuencia común más larga:", lcs(x, y))

# Ejercicio 23: Implementar un algoritmo de programación dinámica para el problema de la mochila.
# Solución:
def knapsack(pesos, valores, capacidad):
    n = len(valores)
    dp = [[0 for _ in range(capacidad + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacidad + 1):
            if pesos[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], valores[i - 1] + dp[i - 1][w - pesos[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacidad]

valores = [60, 100, 120]
pesos = [10, 20, 30]
capacidad = 50
print("Valor máximo en la mochila:", knapsack(pesos, valores, capacidad))

# Ejercicio 24: Implementar un algoritmo de programación dinámica para el problema de Fibonacci.
# Solución:
def fibonacci(n):
    if n <= 1:
        return n
    fib = [0] * (n + 1)
    fib[1] = 1

    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]

    return fib[n]

n = 10
print("El número de Fibonacci en la posición", n, "es:", fibonacci(n))
