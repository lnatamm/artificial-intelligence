import numpy as np
import matplotlib.pyplot as plt


class LocalRandomSearch:
    def __init__(self,max_it,sigma,func,restricoes,tolerancia_maxima=5,max=False):
        self.max_it = max_it        
        self.f = func
        self.sigma = sigma
        self.restr_l = restricoes[:,0]
        self.restr_u = restricoes[:,1]
        self.tolerancia_maxima = tolerancia_maxima
        self.max = max
        self.x_opt = np.random.uniform(low=self.restr_l,high=self.restr_u)
        self.f_opt = self.f(*self.x_opt)
        
        
        # self.ax.scatter(*self.x_opt,resposta,c='r')
        # plt.pause(.1)
        
    def perturb(self):
        n = np.random.normal(loc=0,scale=self.sigma,size=self.x_opt.shape)
        x_cand = self.x_opt + n
        for i in range(len(x_cand)):
            if x_cand[i] < self.restr_l[i]:
                x_cand[i] = self.restr_l[i]
            if x_cand[i] > self.restr_u[i]:
                x_cand[i] = self.restr_u[i]
        return x_cand
    
    def compare(self,f_cand):
        if self.max:
            return f_cand > self.f_opt
        else:
            return f_cand < self.f_opt

    def search(self):
        it = 0
        tolerancia = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(*x_cand)
            
            # self.ax.scatter(*x_cand,f_cand,c='purple',alpha=.3)
            if self.compare(f_cand):
                self.x_opt = x_cand
                self.f_opt = f_cand
                tolerancia = 0
                # self.ax.scatter(*self.x_opt,self.f_opt,c='r')
                # plt.pause(.1)
            else:
                tolerancia += 1
                if tolerancia > self.tolerancia_maxima:
                    return self.x_opt, self.f_opt
            it+=1
        # self.ax.scatter(*self.x_opt,self.f_opt,c='g',marker='*')
        return self.x_opt, self.f_opt
        
class GlobalRandomSearch:
    def __init__(self,max_it,func,restricoes,tolerancia_maxima=5,max=False):
        self.max_it = max_it        
        self.f = func
        self.restr_l = restricoes[:,0]
        self.restr_u = restricoes[:,1]
        self.tolerancia_maxima = tolerancia_maxima
        self.max = max
        self.x_opt = np.random.uniform(low=self.restr_l,high=self.restr_u)
        self.f_opt = self.f(*self.x_opt)
        
        # x_axis = np.linspace(np.min(restricoes),np.max(restricoes),500)
        # X,Y = np.meshgrid(x_axis,x_axis)
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(projection='3d')
        # self.ax.plot_surface(X,Y,self.f(X,Y),rstride=20,cstride=20,cmap='gray',alpha=.4)
        # resposta = self.f(*self.x_opt) #splat operator
        # self.ax.scatter(*self.x_opt,resposta,c='r')
        # plt.pause(.1)
    
    def perturb(self):
        return np.random.uniform(low=self.restr_l,high=self.restr_u)
    
    def compare(self,f_cand):
        if self.max:
            return f_cand > self.f_opt
        else:
            return f_cand < self.f_opt
        
    def search(self):
        it = 0
        tolerancia = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(*x_cand)
            
            # self.ax.scatter(*x_cand,f_cand,c='purple',alpha=.3)
            if self.compare(f_cand):
                self.x_opt = x_cand
                self.f_opt = f_cand
                tolerancia = 0
                # self.ax.scatter(*self.x_opt,self.f_opt,c='r')
                # plt.pause(.1)
            else:
                tolerancia += 1
                if tolerancia > self.tolerancia_maxima:
                    return self.x_opt, self.f_opt
            it+=1
        return self.x_opt, self.f_opt
        # self.ax.scatter(*self.x_opt,self.f_opt,c='g',marker='*')
        # plt.show()
        
class HillClimbing:
    def __init__(self,max_it,max_viz,epsilon,func,restricoes,tolerancia_maxima=15,max=False):
        self.max_it = max_it
        self.max_viz = max_viz
        self.e = epsilon
        self.f = func
        self.restr_l = restricoes[:,0]
        self.restr_u = restricoes[:,1]
        self.tolerancia_maxima = tolerancia_maxima
        self.max = max
        self.x_opt = restricoes[:,0]
        self.f_opt = self.f(*self.x_opt)
        
        
        #fig:
        x_axis = np.linspace(np.min(restricoes),np.max(restricoes),500)
        X,Y = np.meshgrid(x_axis,x_axis)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.plot_surface(X,Y,self.f(X,Y),rstride=20,cstride=20,cmap='gray',alpha=.4,edgecolors='k')
        resposta = self.f(*self.x_opt) #splat operator
        self.ax.scatter(*self.x_opt,resposta,c='r')
        plt.pause(.1)
        plt.show()
    def perturb(self):
        x_cand = np.random.uniform(self.x_opt-self.e,self.x_opt+self.e)
        for i in range(len(x_cand)):
            if x_cand[i] < self.restr_l[i]:
                x_cand[i] = self.restr_l[i]
            if x_cand[i] > self.restr_u[i]:
                x_cand[i] = self.restr_u[i]
        return x_cand
            
    def compare(self,f_cand):
        if self.max:
            return f_cand > self.f_opt
        else:
            return f_cand < self.f_opt
        
    def search(self):
        it = 0
        melhoria = True
        tolerancia = 0
        while it < self.max_it and melhoria:
            melhoria = False
            for j in range(self.max_viz):
                x_cand = self.perturb()
                f_cand = self.f(*x_cand)
                # self.ax.scatter(*x_cand,f_cand,c='purple',alpha=.3)
                if self.compare(f_cand):
                    self.x_opt = x_cand
                    self.f_opt = f_cand
                    # self.ax.scatter(*self.x_opt,self.f_opt,c='r')
                    # plt.pause(.1)
                    melhoria = True
                    tolerancia = 0
                    # self.ax
                    break
            if not melhoria:
                tolerancia += 1
                if tolerancia > self.tolerancia_maxima:
                    return self.x_opt, self.f_opt
            it+=1
        return self.x_opt, self.f_opt
        # self.ax.scatter(*self.x_opt,self.f_opt,c='g',marker='*',s=150)
        # plt.show()

class SimulatedAnnealing:
    def __init__(self,max_it,N,t,func,restricoes):
        self.max_it = max_it        
        self.f = func
        self.N = N
        self.t = t
        self.restr_l = restricoes[:,0]
        self.restr_u = restricoes[:,1]
        self.x_opt = np.random.uniform(low=self.restr_l,high=self.restr_u)
        self.f_opt = self.f(*self.x_opt)
        
        x_axis = np.linspace(np.min(restricoes),np.max(restricoes),500)
        X,Y = np.meshgrid(x_axis,x_axis)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.plot_surface(X,Y,self.f(X,Y),rstride=20,cstride=20,cmap='gray',alpha=.4)
        resposta = self.f(*self.x_opt) #splat operator
        self.ax.scatter(*self.x_opt,resposta,c='r')
        plt.pause(.1)
        
    def perturb(self):
        n = np.random.normal(loc=0,scale=self.sigma,size=self.x_opt.shape)
        combinations = np.random.choice(self.N, size=self.N, replace=False)
        x_cand = self.x_opt + n[combinations]
        for i in range(len(x_cand)):
            if x_cand[i] < self.restr_l[i]:
                x_cand[i] = self.restr_l[i]
            if x_cand[i] > self.restr_u[i]:
                x_cand[i] = self.restr_u[i]
        return x_cand
        
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(*x_cand)
            
            # self.ax.scatter(*x_cand,f_cand,c='purple',alpha=.3)
            if f_cand > self.f_opt or np.random.rand() < np.exp((f_cand - self.f_opt) / self.t):
                self.t *= 0.999
                self.x_opt = x_cand
                self.f_opt = f_cand
                self.ax.scatter(*self.x_opt,self.f_opt,c='r')
                plt.pause(.1)
            it+=1
        self.ax.scatter(*self.x_opt,self.f_opt,c='g',marker='*')
        plt.show()