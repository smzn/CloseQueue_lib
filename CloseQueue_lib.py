import numpy as np
from numpy.linalg import solve 
import itertools #重複組合せを求める
from functools import lru_cache

class CloseQueue_lib:

    def __init__(self, N, K, p, mu, m):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.K = K #網内の客数
        self.p = p
        self.alpha = np.array([])#ノード到着率
        self.mu = mu
        self.rho = np.array([])
        self.pi = np.zeros((len(p),len(p)))
        self.m = m
        self.comb = self.fact(N + K -1) // (self.fact(N + K -1 - (N -1)) * self.fact(N -1))
        print('comb = {}'.format(self.comb))
        
    #閉鎖型ネットワークのノード到着率αを求める
    #https://mizunolab.sist.ac.jp/2019/09/bcmp-network1.html
    def getCloseTraffic(self):
        e = np.eye(len(self.p)-1) #次元を1つ小さくする
        pe = self.p[1:len(self.p), 1:len(self.p)].T - e #行と列を指定して次元を小さくする
        lmd = self.p[0, 1:len(self.p)] #0行1列からの値を右辺に用いる
        slv = solve(pe, lmd * (-1))
        self.alpha = np.insert(slv, 0, 1.0) #α1=1を追加
        return self.alpha

    #畳み込みで求める
    #http://www.ocw.titech.ac.jp/index.php?module=General&action=DownLoad&file=201316812-17-0-1.pdf&type=cal&JWC=201316812
    def calcConvolution(self, n, k, rho):
        #rho = alpha / mu
        if k == 0: #Gj(0) = 1 (j = 1,...,N)
            g = 1
        elif n == 0:#G(0,k) = f0(k)
            g = np.power(rho[n], k)
        else : 
            g = self.calcConvolution(n-1, k,rho) + rho[n] * self.calcConvolution(n, k-1,rho)
        return g

    # Closed Queueで窓口数が複数の場合の対応 : QN&MC P289 式(7.62) 2021/06/03追加したかったけどやめた
    
    def getGNK(self, rho):
        self.g = self.calcConvolution(self.N-1, self.K, rho) #ノード番号は1小さくしている
        self.gg = self.calcConvolution(self.N-1, self.K-1, rho) #ノード番号は1小さくしている
        
    def getThroughput(self):
        throughput = self.alpha * self.gg / self.g
        return throughput

    def getAvailability(self):
        avail = self.alpha / self.mu * self.gg / self.g
        return avail

    #http://www.ieice-hbkb.org/files/05/05gun_01hen_05.pdf
    #畳み込みを利用する場合、周辺分布までしか算出できない
    def getStationary(self, p, rho):
        pi = []
        for i in range(self.K+1):
            pi.append(np.power(rho[self.N-1], i) * self.calcConvolution(self.N-2,self.K-i, rho) / self.calcConvolution(self.N-1, self.K, rho))
        return pi

    def getLength(self, pi):
        l = np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.K+1):
                l[i] += j * pi[i][j]
        return l 

    def getStationaryDistribution(self):
        #定常分布を求めたいノードをノードN(最後のノード)に持ってくる
        #推移確率、α、μの順番を変更
        pi_all = np.zeros((self.N, self.K+1)) #全状態確率を入れるリスト
        for i in range(self.N):
            p = self.p.copy() #copyがないとobjectそのものになってしまう
            alpha = self.alpha.copy()
            mu = self.mu.copy()

            #行の交換
            p[len(p)-1] = self.p[i]
            p[i] = self.p[len(self.p)-1]
            pp = p.copy()
            #列の交換
            p[:,len(p)-1] = pp[:,i]
            p[:,i] = pp[:,len(self.p)-1]

            #αの交換
            alpha[len(p)-1] = self.alpha[i]
            alpha[i] = self.alpha[len(self.p)-1]

            #μの交換
            mu[len(p)-1] = self.mu[i]
            mu[i] = self.mu[len(self.p)-1]

            rho = alpha / mu
            pi = self.getStationary(p, rho)
            pi_all[i] = pi 
            
        return pi_all
    
    def getPi(self, k, rho):#定常分布pi(k1,k2,k3,...)を返す
        rho_p = 1
        for i in range(self.N):
            rho_p *= rho[i]**k[i]
        pi = rho_p / self.g
        return pi
            
    def fact(self, n):
        if n == 1:
            return 1
        return n * self.fact(n-1)
    
    def getCombi(self,N,K):
        #s = combinations_count(N, K)#重複組み合わせ
        l = [i for i in range(N)]
        p_list = list(itertools.combinations_with_replacement(l, K))
        combi = [[0 for i in range(N)] for j in range(len(p_list))]
        for ind, p in enumerate(p_list):
            for k in range(K):
                combi[ind][p[k]] += 1
        return combi
