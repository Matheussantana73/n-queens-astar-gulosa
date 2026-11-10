import argparse
import heapq
import itertools
import time
from bisect import bisect_left
from dataclasses import dataclass


@dataclass
class Resultado:
    algoritmo: str
    n: int
    status: str
    tempo_s: float
    expandidos: int
    gerados: int
    descobertos: int
    fronteira_maxima: int
    solucao: tuple | None


def novos_conflitos(estado, casa, n):
    """Conta os ataques entre a nova rainha e as já existentes."""
    linha, coluna = divmod(casa, n)
    conflitos = 0

    for outra in estado:
        outra_linha, outra_coluna = divmod(outra, n)

        if (
            linha == outra_linha
            or coluna == outra_coluna
            or abs(linha - outra_linha) == abs(coluna - outra_coluna)
        ):
            conflitos += 1

    return conflitos


def inserir_ordenado(estado, casa):
    """Representação canônica: a ordem de inserção não muda o estado."""
    indice = bisect_left(estado, casa)
    return estado[:indice] + (casa,) + estado[indice:]


def validar_solucao(estado, n):
    if estado is None or len(estado) != n or len(set(estado)) != n:
        return False

    if any(casa < 0 or casa >= n * n for casa in estado):
        return False

    linhas = set()
    colunas = set()
    diagonais_principais = set()
    diagonais_secundarias = set()

    for casa in estado:
        linha, coluna = divmod(casa, n)

        if (
            linha in linhas
            or coluna in colunas
            or linha - coluna in diagonais_principais
            or linha + coluna in diagonais_secundarias
        ):
            return False

        linhas.add(linha)
        colunas.add(coluna)
        diagonais_principais.add(linha - coluna)
        diagonais_secundarias.add(linha + coluna)

    return True


def buscar(
    n,
    algoritmo,
    limite_tempo=10.0,
    limite_gerados=100_000,
    limite_fronteira=50_000,
    podar_conflitos=False,
    uma_por_linha=False,
):
    if n < 1:
        raise ValueError("A quantidade de rainhas deve ser positiva.")

    if algoritmo not in ("astar", "gulosa"):
        raise ValueError("Algoritmo inválido.")

    inicio = time.perf_counter()
    contador = itertools.count()

    # Item da heap:
    # (prioridade, desempate por profundidade, identificador, estado, h)
    fronteira = [(0, 0, next(contador), (), 0)]
    visitados = {()}

    expandidos = 0
    gerados = 0
    fronteira_maxima = 1

    def finalizar(status, solucao=None):
        return Resultado(
            algoritmo=algoritmo,
            n=n,
            status=status,
            tempo_s=time.perf_counter() - inicio,
            expandidos=expandidos,
            gerados=gerados,
            descobertos=len(visitados),
            fronteira_maxima=fronteira_maxima,
            solucao=solucao,
        )

    while fronteira:
        if time.perf_counter() - inicio >= limite_tempo:
            return finalizar("limite de tempo")

        _, _, _, estado, h = heapq.heappop(fronteira)
        profundidade = len(estado)

        if profundidade == n and h == 0:
            return finalizar("solução encontrada", estado)

        # Estados completos com conflitos não têm sucessores.
        if profundidade == n:
            continue

        expandidos += 1
        ocupadas = set(estado)

        # Variante: só a próxima linha vazia (reduz n² sucessores para n).
        casas = (
            range(profundidade * n, profundidade * n + n)
            if uma_por_linha
            else range(n * n)
        )

        for casa in casas:
            if time.perf_counter() - inicio >= limite_tempo:
                return finalizar("limite de tempo")

            if casa in ocupadas:
                continue

            if gerados >= limite_gerados:
                return finalizar("limite de sucessores gerados")

            gerados += 1
            novo_h = h + novos_conflitos(estado, casa, n)

            # Como apenas adicionamos rainhas, conflitos existentes
            # jamais poderão ser removidos nesta formulação.
            if podar_conflitos and novo_h > 0:
                continue

            novo_estado = inserir_ordenado(estado, casa)

            if novo_estado in visitados:
                continue

            if len(fronteira) >= limite_fronteira:
                return finalizar("limite de fronteira")

            novo_g = profundidade + 1
            prioridade = novo_g + novo_h if algoritmo == "astar" else novo_h

            visitados.add(novo_estado)

            # Em prioridades iguais, preferimos maior profundidade.
            # Persistindo o empate, usamos a ordem de inserção.
            heapq.heappush(
                fronteira,
                (
                    prioridade,
                    -novo_g,
                    next(contador),
                    novo_estado,
                    novo_h,
                ),
            )

            fronteira_maxima = max(fronteira_maxima, len(fronteira))

    return finalizar("espaço de busca esgotado")


def mostrar_resultado(resultado):
    print(f"\nAlgoritmo: {resultado.algoritmo}")
    print(f"Rainhas: {resultado.n}")
    print(f"Status: {resultado.status}")
    print(f"Tempo observado: {resultado.tempo_s:.6f} s")
    print(f"Estados expandidos: {resultado.expandidos}")
    print(f"Sucessores gerados: {resultado.gerados}")
    print(f"Estados únicos descobertos: {resultado.descobertos}")
    print(f"Maior fronteira: {resultado.fronteira_maxima}")

    if resultado.solucao is None:
        return

    if not validar_solucao(resultado.solucao, resultado.n):
        raise RuntimeError("A solução retornada falhou na validação.")

    print("Solução validada: sim")

    if resultado.n <= 20:
        ocupadas = set(resultado.solucao)

        for linha in range(resultado.n):
            print(
                " ".join(
                    "Q" if linha * resultado.n + coluna in ocupadas else "."
                    for coluna in range(resultado.n)
                )
            )
    else:
        # Saída compacta: coluna da rainha de cada linha, começando em 1.
        colunas = [0] * resultado.n

        for casa in resultado.solucao:
            linha, coluna = divmod(casa, resultado.n)
            colunas[linha] = coluna + 1

        print("Coluna por linha, com índices iniciando em 1:")
        print(colunas)


def main():
    parser = argparse.ArgumentParser(
        description="Problema das n-rainhas com A* e busca gulosa."
    )
    parser.add_argument("n", type=int, help="Quantidade de rainhas.")
    parser.add_argument(
        "--algoritmo",
        choices=["astar", "gulosa", "ambos"],
        default="ambos",
    )
    parser.add_argument("--tempo", type=float, default=10.0)
    parser.add_argument("--max-gerados", type=int, default=100_000)
    parser.add_argument("--max-fronteira", type=int, default=50_000)
    parser.add_argument(
        "--podar-conflitos",
        action="store_true",
        help="Descarta estados com ataques; usar como variante identificada.",
    )

    parser.add_argument(
        "--uma-por-linha",
        action="store_true",
        help="Variante: a k-ésima rainha só entra na linha k (identificar no relatório).",
    )

    args = parser.parse_args()

    if args.n < 1:
        parser.error("n deve ser maior ou igual a 1.")

    if (
        args.tempo <= 0
        or args.max_gerados <= 0
        or args.max_fronteira <= 0
    ):
        parser.error("Os limites precisam ser positivos.")

    algoritmos = (
        ["astar", "gulosa"]
        if args.algoritmo == "ambos"
        else [args.algoritmo]
    )

    print("Uma rainha por linha:", "sim" if args.uma_por_linha else "não")
    print(
        "Modo:",
        "com poda de conflitos"
        if args.podar_conflitos
        else "sem poda de conflitos",
    )

    for algoritmo in algoritmos:
        resultado = buscar(
            n=args.n,
            algoritmo=algoritmo,
            limite_tempo=args.tempo,
            limite_gerados=args.max_gerados,
            limite_fronteira=args.max_fronteira,
            podar_conflitos=args.podar_conflitos,
            uma_por_linha=args.uma_por_linha,
        )
        mostrar_resultado(resultado)


if __name__ == "__main__":
    main()