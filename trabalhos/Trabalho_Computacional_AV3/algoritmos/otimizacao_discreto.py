import numpy as np
import matplotlib.pyplot as plt
class SimulatedAnnealing:
    def __init__(self,max_it,trocas,f,t,cidades,origem):
        self.trocas = trocas
        self.max_it = max_it
        self.f = f
        self.t = t
        self.origem = origem        
        self.cidades = np.vstack((origem,cidades))                
        self.x_opt = np.random.permutation(self.cidades.shape[0]-1)+1
        self.x_opt = np.concatenate((np.array([0]),self.x_opt))
        self.f_opt = self.f(cidades=self.cidades,caminho=self.x_opt)
        self.avaliados = []
        
        #não faz parte do global random search (somente é gráfico)
        self.ax = plt.subplot()
        self.linhas = []
        self.ax.scatter(self.cidades[:,0],self.cidades[:,1])
        self.plot_opt()
        
        
        pass
    
    def plot_opt(self,cor='k'):
        if len(self.linhas)!=0:
            for linha in self.linhas:
                linha[0].remove()
            self.linhas = []
        for i in range(len(self.x_opt)):
            p1 = self.cidades[self.x_opt[i]]
            p2 = self.cidades[self.x_opt[(i+1)%len(self.x_opt)]]
            if i == 0:
                l = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='g')
            elif i == len(self.x_opt)-1:
                l = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='cyan')
            else:
                l = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c=cor)
            self.linhas.append(l)    
                
            # plt.pause(.5)
    def perturb(self):
        idx1,idx2 = (np.random.permutation(self.cidades.shape[0]-1)+1)[:2]
        x_cand = np.copy(self.x_opt)
        x_cand[idx1],x_cand[idx2] = x_cand[idx2],x_cand[idx1]
        # indices1 = (np.random.permutation(self.cidades.shape[0]-1)+1)[:self.trocas]
        # indices2 = np.random.permutation(indices1)
        # x_cand = np.copy(self.x_opt)
        # x_cand[indices2] = self.x_opt[indices1]
        return x_cand
    def busca(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            # while np.sum(np.equal(self.x_opt,x_cand))==0:
            #     x_cand = self.perturb()
                
            f_cand = self.f(cidades=self.cidades,caminho=x_cand)
            
            if f_cand < self.f_opt or np.random.random() < np.exp(-((f_cand - self.f_opt) / self.t)):
                self.t *= 0.99
                self.x_opt = x_cand
                self.f_opt = f_cand
                self.avaliados.append(self.f_opt)
                self.plot_opt()   
                plt.pause(.5)    
            it+=1
        self.plot_opt(cor='pink')
        plt.figure(2)
        plt.plot(self.avaliados)
        plt.title('Melhorias ao longo da busca')
        plt.xlabel('Iterações')       
        plt.show()

class SimulatedAnnealing8Queens:
    def __init__(self, max_it=10000, t=100, decay=0.95, plot=False):
        self.max_it = max_it
        self.t = t
        self.decay = decay
        self.plot = plot
        self.x_opt = np.random.randint(1, 9, size=8)
        self.f_opt = self.f(self.x_opt)
        self.avaliados = []
        if self.plot:
            self.fig, self.ax = plt.subplots()
            plt.ion()  # Ativa modo interativo para atualização em tempo real (ChatGPT fez a parte de plotagem))

    # Método para verificar quantos conflitos existem
    def h(self, x):
        conflitos = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                    conflitos += 1
        return conflitos

    # Função de aptidão: número de pares de rainhas que não se atacam (Maximizar)
    def f(self, x):
        return 28 - self.h(x)

    # Gera uma nova permutação
    def perturb(self):
        x_cand = np.copy(self.x_opt)
        i = np.random.randint(0, 8)
        x_cand[i] = np.random.randint(1, 9)
        return x_cand

    # Método para plotar o tabuleiro com as rainhas (Feito pelo ChatGPT)
    def plot_tabuleiro(self, x):
        self.ax.clear()
        # Desenha o tabuleiro quadriculado
        for i in range(8):
            for j in range(8):
                cor = 'white' if (i + j) % 2 == 0 else 'gray'
                self.ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, color=cor))

        # Posiciona as rainhas
        for col in range(8):
            lin = 8 - x[col]  # converter linha para o sistema do matplotlib
            self.ax.text(col + 0.5, lin + 0.5, '♛', fontsize=24, ha='center', va='center', color='k')

        self.ax.set_xlim(0, 8)
        self.ax.set_ylim(0, 8)
        self.ax.set_xticks(range(8))
        self.ax.set_yticks(range(8))
        self.ax.set_xticklabels(range(1, 9))
        self.ax.set_yticklabels(reversed(range(1, 9)))
        self.ax.set_title(f"Solução atual: {x} | Aptidão: {self.f(x)}")
        self.ax.set_aspect('equal')
        plt.pause(0.001)

    def busca(self):
        iteracao = 0
        while iteracao < self.max_it and self.f_opt < 28:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)

            if f_cand > self.f_opt or np.random.rand() < np.exp(-(self.f_opt - f_cand) / self.t):
                self.x_opt = x_cand
                self.f_opt = f_cand
                self.avaliados.append(self.f_opt)

            self.t *= self.decay
            iteracao += 1

        print(f"Melhor solução: {self.x_opt}, Aptidão: {self.f_opt}")
        if self.plot:
            self.plot_tabuleiro(self.x_opt)
        
            plt.ioff()  # Desativa modo interativo
            plt.show()

            # Plot evolução da aptidão
            plt.figure()
            plt.plot(self.avaliados)
            plt.xlabel("Iterações")
            plt.ylabel("f(x)")
            plt.title("Evolução da solução - 8 Rainhas")
            plt.grid(True)
            plt.show()

