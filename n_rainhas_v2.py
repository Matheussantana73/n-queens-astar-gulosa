"""N-rainhas: A* e busca gulosa (versão 2).

Formulação (igual ao enunciado):
- Estado inicial: tabuleiro vazio.
- Sucessor: adicionar uma rainha em qualquer casa vazia.
- h = pares de rainhas que se atacam; custo de cada passo = 1.

Distinções para garantir solução válida:
- Estados com h > 0 são descartados (rainhas nunca são removidas).
- Gulosa: descarta estados em que alguma linha vazia ficou sem casa livre,
  e desempata (todos têm h = 0) pela linha mais restrita. Toda solução
  tem uma rainha nessa linha, então as casas de outras linhas são redundantes.
  Empates restantes são aleatórios (--semente), com reinício quando a
  tentativa retrocede demais (o limite dobra a cada reinício).
"""

import argparse
import heapq
import itertools
import random
import time

from n_rainhas import Resultado, inserir_ordenado, mostrar_resultado, novos_conflitos


def astar(n, limite_tempo, limite_fronteira):
    """A* com f = g + h, sucessores em todas as casas vazias e poda h > 0."""
    inicio = time.perf_counter()
    contador = itertools.count()
    fronteira = [(0, next(contador), ())]
    visitados = {()}
    expandidos = gerados = 0
    fronteira_maxima = 1

    def fim(status, solucao=None):
        return Resultado("astar", n, status, time.perf_counter() - inicio,
                         expandidos, gerados, len(visitados),
                         fronteira_maxima, solucao)

    while fronteira:
        _, _, estado = heapq.heappop(fronteira)
        g = len(estado)
        if g == n:
            return fim("solução encontrada", estado)

        expandidos += 1
        ocupadas = set(estado)
        for casa in range(n * n):
            if casa in ocupadas:
                continue
            if gerados % 1024 == 0 and time.perf_counter() - inicio >= limite_tempo:
                return fim("limite de tempo")
            gerados += 1
            h = novos_conflitos(estado, casa, n)
            if h > 0:
                continue
            novo = inserir_ordenado(estado, casa)
            if novo in visitados:
                continue
            if len(fronteira) >= limite_fronteira:
                return fim("limite de fronteira")
            visitados.add(novo)
            heapq.heappush(fronteira, (g + 1 + h, next(contador), novo))
            fronteira_maxima = max(fronteira_maxima, len(fronteira))

    return fim("espaço de busca esgotado")


def gulosa(n, limite_tempo, semente=0, limite_retrocessos=None):
    """Gulosa com f = h.

    Com a poda, todo estado tem h = 0 e o desempate por maior profundidade
    faz a fila se comportar como uma pilha. Por isso a fronteira é guardada
    como pilha de candidatos (mesma ordem de expansão, sem copiar estados).
    """
    inicio = time.perf_counter()
    rng = random.Random(semente)
    col = bytearray(n)
    d1 = bytearray(2 * n)       # linha - coluna + n
    d2 = bytearray(2 * n)       # linha + coluna
    livres = [n] * n            # casas livres em cada linha vazia
    linhas_vazias = set(range(n))
    rainhas = {}
    expandidos = gerados = 0
    fronteira_maxima = 0

    def fim(status, solucao=None):
        return Resultado("gulosa", n, status, time.perf_counter() - inicio,
                         expandidos, gerados, expandidos + 1,
                         fronteira_maxima, solucao)

    def casa_livre(r, c):
        return not (col[c] or d1[r - c + n] or d2[r + c])

    def afetadas(r0, c0, r):
        return {c for c in (c0, r - r0 + c0, r0 + c0 - r) if 0 <= c < n}

    def colocar(r0, c0):
        ok = True
        linhas_vazias.discard(r0)
        for r in linhas_vazias:
            for c in afetadas(r0, c0, r):
                if casa_livre(r, c):
                    livres[r] -= 1
            if livres[r] == 0:
                ok = False
        col[c0] = d1[r0 - c0 + n] = d2[r0 + c0] = 1
        rainhas[r0] = c0
        return ok

    def remover(r0, c0):
        col[c0] = d1[r0 - c0 + n] = d2[r0 + c0] = 0
        del rainhas[r0]
        for r in linhas_vazias:
            for c in afetadas(r0, c0, r):
                if casa_livre(r, c):
                    livres[r] += 1
        linhas_vazias.add(r0)

    def novo_quadro():
        r = min(linhas_vazias, key=livres.__getitem__)
        candidatas = [c for c in range(n) if casa_livre(r, c)]
        rng.shuffle(candidatas)
        return [r, candidatas, 0, False]

    def reiniciar():
        col[:] = bytes(n)
        d1[:] = bytes(2 * n)
        d2[:] = bytes(2 * n)
        livres[:] = [n] * n
        linhas_vazias.update(range(n))
        rainhas.clear()
        return [novo_quadro()]

    # Reinício aleatório: se a tentativa retroceder demais, recomeça com
    # outra ordem de desempate. Só desliga com limite_retrocessos=None.
    retrocessos = 0
    pilha = [novo_quadro()]
    expandidos = 1
    reinicios = 0
    while pilha:
        if gerados % 256 == 0 and time.perf_counter() - inicio >= limite_tempo:
            return fim("limite de tempo")
        if limite_retrocessos and retrocessos > limite_retrocessos:
            reinicios += 1
            retrocessos = 0
            limite_retrocessos *= 2  # garante completude (n = 2, 3 sem solução)
            pilha = reiniciar()
            expandidos += 1
            continue
        quadro = pilha[-1]
        r, candidatas, i, colocada = quadro
        if colocada:
            remover(r, candidatas[i - 1])
            quadro[3] = False
        if i >= len(candidatas):
            pilha.pop()
            retrocessos += 1
            continue
        quadro[2] += 1
        quadro[3] = True
        gerados += 1
        if not colocar(r, candidatas[i]):
            continue
        if len(rainhas) == n:
            solucao = tuple(sorted(l * n + c for l, c in rainhas.items()))
            return fim(f"solução encontrada ({reinicios} reinícios)", solucao)
        expandidos += 1
        pilha.append(novo_quadro())
        fronteira_maxima = max(fronteira_maxima,
                               sum(len(q[1]) - q[2] for q in pilha))

    return fim("espaço de busca esgotado")


def main():
    p = argparse.ArgumentParser(description="N-rainhas: A* e gulosa (v2).")
    p.add_argument("n", type=int)
    p.add_argument("--algoritmo", choices=["astar", "gulosa", "ambos"], default="ambos")
    p.add_argument("--tempo", type=float, default=60.0)
    p.add_argument("--max-fronteira", type=int, default=2_000_000)
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--retrocessos", type=int, default=50,
                   help="Retrocessos antes de reiniciar a gulosa (0 = nunca).")
    a = p.parse_args()
    if a.n < 1:
        p.error("n deve ser >= 1.")

    algoritmos = ["astar", "gulosa"] if a.algoritmo == "ambos" else [a.algoritmo]
    for alg in algoritmos:
        if alg == "astar":
            r = astar(a.n, a.tempo, a.max_fronteira)
        else:
            r = gulosa(a.n, a.tempo, a.semente, a.retrocessos or None)
        mostrar_resultado(r)


if __name__ == "__main__":
    main()