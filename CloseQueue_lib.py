import numpy as np
from numpy.linalg import solve 

class CloseQueue_lib:

    def __init__(self, N, K, p, mu):
        self.N = N #網内にいる人数
        self.K = K #網内の拠点数
        self.p = p
        self.alpha = np.array([])#ノード到着率
        self.mu = mu
        self.rho = np.array([])
        #self.g = 1 #g = G(N,K) 
        #self.gg = 1 #gg = G(N-1,K)

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
    def calcConvolution(self, n, k):
        self.rho = self.alpha / self.mu
        #kは拠点番号だが、プログラム中の配列番号に合わせるため-1してある。
        if n == 0:
            g = 1
        elif k == 0:
            g = np.power(self.rho[0], n)
        else :
            g = self.calcConvolution(n, k-1) + self.rho[k] * self.calcConvolution(n-1, k)
        return g

    def getGNK(self):
        self.g = self.calcConvolution(self.N, self.K-1)
        self.gg = self.calcConvolution(self.N-1, self.K-1)
        
    def getThroughput(self):
        throughput = self.alpha * self.gg / self.g
        return throughput

    def getAvailability(self):
        avail = self.alpha / self.mu * self.gg / self.g
        return avail