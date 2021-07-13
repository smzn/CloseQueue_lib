import numpy as np

class BCMP_lib:
    
    def __init__(self, N, R, K, mu, type_list, alpha):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        #self.p = p #推移確率行列 (N×N)
        self.mu = mu #サービス率 (N×R)
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.combi_list = []
        self.combi_list = self.getCombiList2([], self.K, self.R, 0, self.combi_list) #K, Rのときの組み合わせ
        NK = [k+1 for k in self.K]
        NK.insert(0, self.N)
        self.mp_set= np.zeros(NK) #周辺分布を格納
        self.exp = np.zeros((self.N, self.R)) #平均計内人数を格納
        self.tp = np.zeros((self.N, self.R)) #スループットを格納
        self.rho = [] #利用率を格納(各拠点ごと)
        #self.m = m #各拠点の窓口数
        self.alpha = alpha
        #課題
        #1. 推移確率行列から到着率の計算過程を入れる(推移確率は取り込みとランダム生成(エルゴード性を担保)の両方)
        #2. 窓口数を加味した計算(利用率やFSの部分)
        
    def getBCMP(self):
        for n in range(self.N): #P324 Table 8.3
            for i, k in enumerate(self.combi_list):
                g = self.getConvolution(n, k)
                #print('G{0}({1}) = {2}'.format(n,k,g))
                if n == self.N-1 and i == len(self.combi_list)-1: #combi_listの最後の要素のとき(注意：常に最後のリストで大丈夫か？)
                    self.GK = g
        
        self.getMarginalProbabilitySet() #[(0,0),(0,1),(0,2)],[(1,0),(1,1),(1,2)]
        
        self.getEXP()
        
        self.getThroughput()
        
        self.getUsageRate()
        
        return self.mp_set, self.exp, self.tp, self.rho
        
        
    def getMarginalProbabilitySet(self): #全ての周辺分布を求める
        for n in range(self.N):
            for k in self.combi_list:
                nk = []
                nk.append([n])
                for i in k: #ndarrayでリストで要素を指定するときには[[n],[k1],[k2]]のように[]をつけて、タプルにする
                    nk.append([i])
                mp = self.getMarginalProbability(n, k)
                self.mp_set[tuple(nk)] = mp
        
        
    #周辺分布を求める(拠点nに対して)
    #G(K)をgetConvolutionでN=n-1、Si=[K1,K2]となった時がGKなので最初にそれを取得してから(8.23)を計算する
    def getMarginalProbability(self, n, Si):
        mp = self.FS(n, Si) * self.getGi(n, np.array(K)-np.array(Si)) / self.GK
        return mp
    
    def getGi(self, n, Si): #P322 式(8.25,26,27) 引数nはG_N^(i)のiに対応する
        if sum(Si) == 0: #式(8.27)
            gi = 1
        else:
            gi = self.getConvolution(self.N-1, Si)
            combi_list = []
            combi_list = self.getCombiList2([], Si, R, 0, combi_list)
            for k in combi_list:
                if(sum(k) == 0):#j=0の場合は除く(8.26) 
                    continue
                else:
                    gi -= self.FS(n, k) * self.getGi(n, np.array(Si)-np.array(k))
        return gi
    
    def getConvolution(self, n, Si):#P321式(8.21)
        g = 0
        combi_list = []
        combi_list = self.getCombiList2([], Si, R, 0, combi_list)
        if n == 0:
            g = self.FS(n, Si)
        elif n >= 1:
            for k in combi_list:
                g += self.getConvolution(n-1, k) * self.FS(n, np.array(Si)-np.array(k)) 
        return g    
    
    # P323 Table 8.2 Fi(Si)の計算, P303式(7.82)
    #Siはノードiにおけるクラス別人数分布:(ノードiのクラス0の人数, ノードiのクラス1の人数)
    #Siは(0,0),(1,0),(0,1),(1,1),(0,2),(1,2) K1=1,K2=2なので
    def FS(self, n, Si):#sは状態分布, type_number =1(FCFS),2(PS),3(IS),4(LCFS-PR)
        f = 1
        if self.type_list[n] == 1:
            print('FCFS') #ここはまだ未実装
        else:
            for r in range(R):#type-3はこのループで終わり
                f *= 1 / self.fact(Si[r]) * (self.alpha[n,r] / self.mu[n,r])**Si[r]
            if self.type_list[n] == 2 or self.type_list[n] == 4:#type-2,4は累乗をかける    
                f *= self.fact(sum(Si))
        return f
    
    def fact(self, n):
        if n <= 1:
            return 1
        return n * self.fact(n-1)
        
    def getCombiList2(self, combi, K, R, idx, Klist):
        if len(combi) == R:
            Klist.append(combi.copy())
            return Klist
        for v in range(K[idx]+1):
            combi.append(v)
            Klist = self.getCombiList2(combi, K, R, idx+1, Klist)
            combi.pop()
        return Klist
    
    #平均系内人数を求める
    def getEXP(self):
        for n in range(self.N):
            for k in self.combi_list:
                nk = []
                nk.append([n])
                for i in k: #ndarrayでリストで要素を指定するときには[[n],[k1],[k2]]のように[]をつけて、タプルにする
                    nk.append([i])
                for r in range(self.R):
                    self.exp[n,r] += k[r] * self.mp_set[tuple(nk)]
        #return self.exp
    
    #スループット算出
    def getThroughput(self):
        for n in range(self.N):
            for r in range(self.R):
                r1 = np.zeros(self.R, dtype = int)
                r1[r] = 1
                self.tp[n,r] = self.alpha[n,r] * self.getConvolution(self.N-1, np.array(self.K) - r1) / self.GK
        #return self.tp
    
    def getUsageRate(self): #利用率算出 P322 式(8.29)lambda / (m * mu) 今回はm = 1 
        self.rho = self.tp / self.mu
        #return self.rho
        
if __name__ == '__main__':
    
    N = 4
    R = 2
    K1 = 1
    K2 = 2
    K = [K1, K2]
    mu = np.array([[1/1, 1/2],[1/4, 1/5],[1/8, 1/10],[1/12, 1/16]])
    type_list = [2, 4, 4, 3] #Node1:Type2プロセッサシェアリング（Processor Sharing: PS）, Node2:Type4後着順割込継続型（LCFS-PR), Node3:Type4後着順割込継続型（LCFS-PR), Node4:Type3無限サーバ（Infinite Server: IS）, その他Type1(FCFS) 
    alpha = np.array([[1, 1],[0.4, 0.4],[0.4, 0.3],[0.2, 0.3]])
    
    bcmp = BCMP_lib(N, R, K, mu, type_list, alpha)
    mp_set, exp, tp, rho = bcmp.getBCMP()
    print('周辺分布')
    print(mp_set)
    print('平均系内人数')
    print(exp)
    print('スループット')
    print(tp)
    print('利用率')
    print(rho)