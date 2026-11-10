import csv
import platform
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent))

from n_rainhas import buscar, validar_solucao
import n_rainhas_v2

TAMANHOS = (8, 80, 800)
SEMENTES = range(5)
LIMITE_TEMPO = 60.0


def literal(algoritmo):
    return lambda n, semente: buscar(n, algoritmo)


def astar_poda(n, semente):
    return n_rainhas_v2.astar(n, LIMITE_TEMPO, 2_000_000)


def gulosa_poda(n, semente):
    return n_rainhas_v2.gulosa(n, LIMITE_TEMPO, semente, 50)


CONFIGURACOES = [
    ("A* literal", literal("astar"), False),
    ("Gulosa literal", literal("gulosa"), False),
    ("A* com poda", astar_poda, False),
    ("Gulosa com poda", gulosa_poda, True),
]


def executar():
    linhas = []

    for nome, funcao, usa_semente in CONFIGURACOES:
        for n in TAMANHOS:
            for semente in SEMENTES if usa_semente else [0]:
                r = funcao(n, semente)
                resolveu = r.solucao is not None

                if resolveu and not validar_solucao(r.solucao, n):
                    raise RuntimeError(f"Solução inválida: {nome}, n={n}")

                linhas.append({
                    "configuracao": nome,
                    "n": n,
                    "semente": semente,
                    "status": r.status,
                    "resolveu": resolveu,
                    "tempo_s": r.tempo_s,
                    "expandidos": r.expandidos,
                    "gerados": r.gerados,
                    "fronteira_maxima": r.fronteira_maxima,
                })
                print(nome, n, semente, r.status,
                      f"{r.tempo_s:.4f}s", r.expandidos, r.gerados,
                      flush=True)

    with open(PASTA / "resultados.csv", "w", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)

    return linhas


def resumir(linhas):
    resumo = {}

    for nome, _, _ in CONFIGURACOES:
        for n in TAMANHOS:
            grupo = [l for l in linhas if l["configuracao"] == nome and l["n"] == n]
            resumo[nome, n] = {
                "tempo_s": statistics.median(l["tempo_s"] for l in grupo),
                "expandidos": statistics.median(l["expandidos"] for l in grupo),
                "resolveu": all(l["resolveu"] for l in grupo),
            }

    return resumo


def grafico(resumo, campo, rotulo, arquivo):
    fig, eixo = plt.subplots(figsize=(9.5, 4.5))
    largura = 0.2
    posicoes = range(len(TAMANHOS))

    for i, (nome, _, _) in enumerate(CONFIGURACOES):
        valores = [max(resumo[nome, n][campo], 1e-6) for n in TAMANHOS]
        xs = [p + (i - 1.5) * largura for p in posicoes]
        barras = eixo.bar(xs, valores, largura, label=nome)

        for barra, n in zip(barras, TAMANHOS):
            if not resumo[nome, n]["resolveu"]:
                barra.set_hatch("//")
                barra.set_alpha(0.6)
                eixo.annotate("X", (barra.get_x() + barra.get_width() / 2,
                                    barra.get_height()),
                              ha="center", va="bottom", fontsize=9,
                              color="red", fontweight="bold")

    eixo.set_yscale("log")
    eixo.set_xticks(list(posicoes))
    eixo.set_xticklabels([f"{n} rainhas" for n in TAMANHOS])
    eixo.set_ylabel(rotulo + " (escala log)")
    eixo.set_title(rotulo + " por algoritmo (X = parou no limite, sem solução)")
    eixo.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    eixo.grid(axis="y", which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA / arquivo, dpi=150)
    plt.close(fig)


def main():
    print(platform.python_version(), platform.system(), platform.machine())
    linhas = executar()
    resumo = resumir(linhas)

    print("\nResumo (mediana):")
    for (nome, n), r in resumo.items():
        print(f"{nome:16} n={n:4} resolveu={r['resolveu']!s:5} "
              f"tempo={r['tempo_s']:.6f}s expandidos={r['expandidos']}")

    grafico(resumo, "expandidos", "Estados expandidos (passos)", "grafico_passos.png")
    grafico(resumo, "tempo_s", "Tempo (s)", "grafico_tempo.png")


if __name__ == "__main__":
    main()
