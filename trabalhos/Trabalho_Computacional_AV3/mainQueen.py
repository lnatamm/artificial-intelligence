from algoritmos.otimizacao_discreto import SimulatedAnnealing8Queens
import time

inicio = time.time()
print("Iniciando busca com Têmpera Simulada para o problema das 8 Rainhas...")
solucoes_encontradas = set()
total_tentativas = 0

while len(solucoes_encontradas) < 92:
    sa = SimulatedAnnealing8Queens(max_it=10000, t=100, decay=0.99)
    sa.busca()

    if sa.f_opt == 28:
        solucao = tuple(sa.x_opt)
        solucao = tuple(map(int, solucao))
        if solucao not in solucoes_encontradas:
            solucoes_encontradas.add(solucao)
            print(f"Solução {len(solucoes_encontradas)} encontrada: {solucao}")
        total_tentativas += 1
fim = time.time()
print(f"Total de soluções distintas encontradas: {len(solucoes_encontradas)}")
print(f"Tentativas totais: {total_tentativas}")
print(f"Tempo total de execução: {fim - inicio:.2f} segundos")
sa = SimulatedAnnealing8Queens(plot=False)
sa.busca()

# sa = SimulatedAnnealing8Queens(plot=True)
# sa.busca()