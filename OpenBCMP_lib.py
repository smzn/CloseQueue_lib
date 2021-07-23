import numpy as np
from numpy.linalg import solve
import pandas as pd
import time

class OpenBCMP_lib:
    
    def __init__(self, N, R, p, lmd, mu, type_list):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.p = p
        self.lmd = lmd
        self.mu = mu #サービス率 (N×R)
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
   
    def getOpenBCMP(self):
        #(1)到着率の取得
        self.alpha = np.reshape(self.getTraffic(),[self.R,self.N]) #取得した値をR*Nの行列にしておく
        #(2)利用率の計算(rhoは拠点、クラスごと、rho_nodeは拠点でクラスをまとめたもの)
        self.rho, self.rho_node = self.getUtilization()
        #(3)系内人数の計算
        self.L = self.getLength()
        #(4)周辺分布の計算(計算する人数を与える)
        self.mp = self.getMarginalProbability(4)

    def getTraffic(self): #(1)到着率の計算
        ie = np.eye(self.N*self.R) #単位行列の生成
        lmd_pv = np.reshape(self.lmd, self.N*self.R) #行列からベクトルに変換
        lmd_pv *= -1 #右辺に移項
        return solve(self.p.T - ie, lmd_pv)#(P^T-E)x = -λ
    
    def getUtilization(self):#(2)利用率の計算
        rho = self.alpha / self.mu.T
        rho_node = np.sum(rho, axis=0)
        return rho, rho_node
    
    def getLength(self):#(3)系内人数の計算
        L = np.zeros((self.R, self.N))
        for i in range(self.R):
            for j in range(self.N):
                L[i,j] = self.rho[i,j] / (1 - self.rho_node[j])
        return L
    
    def getMarginalProbability(self, K):#(4)周辺分布の計算(Kは求める人数)
        mp = np.zeros((self.N, K))
        for i in range(self.N):
            for j in range(K):
                mp[i,j] = (1 - self.rho_node[i])*self.rho_node[i]**j
        return mp
    #周辺分布から定常分布を算出可能(QN&MC P306)
    #pi(3,2,1) = pi(1,3)*pi(2,2)*pi(3,1)で計算できる
    
    def getVisualize(self):
        print('到着率')
        print(self.alpha)
        print('')
        print('利用率(ノード&クラス毎)')
        print(self.rho)
        print('利用率(ノードで集約)')
        print(self.rho_node)
        print('')
        print('系内人数')
        print(self.L)
        print('')
        print('周辺分布')
        print(self.mp)
        
        
if __name__ == '__main__':
    
    N = 3 #与える
    R = 2 #与える
    p = np.array([[0,0.4,0.3,0,0,0],[0.6,0,0.4,0,0,0],[0.5,0.5,0,0,0,0],[0,0,0,0,0.3,0.6],[0,0,0,0.7,0,0.3],[0,0,0,0.4,0.6,0]])
    lmd = np.array([[1,0,0],[1,0,0]])#外部からの到着率
    p0 = np.array([[1,0,0],[1,0,0]]) #外部からの到着確率
    lmd_p = lmd * p0 #トラフィック方程式のλに相当するのがlmd_p
    mu = np.array([[8, 24],[12, 32],[16, 36]])
    type_list = [2, 4, 4]
    start = time.time()
    bcmp = OpenBCMP_lib(N, R, p, lmd_p, mu, type_list) 
    bcmp.getOpenBCMP()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    bcmp.getVisualize()