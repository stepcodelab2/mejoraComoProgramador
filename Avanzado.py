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
                came_from[vecino] = vecino_actual

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
    distancias = {nodo: float('inf') for nodo in grafo}
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