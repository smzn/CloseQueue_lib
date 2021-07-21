import numpy as np
from numpy.linalg import solve
import pandas as pd
import time

class BCMP_lib:
    
    def __init__(self, N, R, K, mu, type_list):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        alp, self.p = self.getProbArrival() #推移確率の自動生成と到着率を求める
        self.saveCSVi(self.p, './tpr/tprNR_'+str(N)+'_'+str(R)+'.csv')#推移確率をcsvで保存しておく
        self.alpha = alp.T #転置して計算形式に合わせる
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
        #self.alpha = alpha
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
        
    #クラス数分推移確率行列を生成して、それぞれの到着率を返す関数
    def getProbArrival(self):
        pr = np.zeros((self.R*self.N, self.R*self.N))
        alpha = np.zeros((self.R, self.N))
        for r in range(self.R):
            class_number = 0
            while class_number != 1: #エルゴード性を持つか確認
                p = np.random.rand(self.N, self.N)
                for i, val in enumerate(np.sum(p, axis=1)): #正規化 axis=1で行和
                    p[i] /= val
                for i in range(self.N):#推移確率のマージ
                    for j in range(self.N):
                        pr[r*self.N+i,r*self.N+j] = p[i,j]
                equivalence, class_number = self.getEquivalence(0, 5, p)#0は閾値、5はステップ数
                if class_number == 1: #クラス数が1(エルゴード性を持つ)
                    break
            alpha_r = self.getCloseTraffic(p)
            for i, val in enumerate(alpha_r): #到着率を配列alphaに格納
                alpha[r,i] = val
                #print('r = {0}, i = {1}, val = {2}'.format(r,i,val))
        return alpha, pr
    
    def getCloseTraffic(self, p):
        e = np.eye(len(p)-1) #次元を1つ小さくする
        pe = p[1:len(p), 1:len(p)].T - e #行と列を指定して次元を小さくする
        lmd = p[0, 1:len(p)] #0行1列からの値を右辺に用いる
        slv = solve(pe, lmd * (-1))
        alpha = np.insert(slv, 0, 1.0) #α1=1を追加
        return alpha
    
    #同値類を求める関数
    def getEquivalence(self, th, roop, p):
        list_number = 0 #空のリストを最初から使用する

        #1. 空のリストを作成して、ノードを追加しておく(作成するのはノード数分)
        equivalence = [[] for i in range(len(p))] 
        
        #2. Comunicationか判定して、Commnicateの場合リストに登録
        for ix in range(roop):
            p = np.linalg.matrix_power(p.copy(), ix+1) #累乗
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    if(p[i][j] > th and p[j][i] > th): #Communicateの場合
                        #3. Communicateの場合登録するリストを選択
                        find = 0 #既存リストにあるか
                        for k in range(len(p)):
                            if i in equivalence[k]: #既存のk番目リストに発見(iで検索)
                                find = 1 #既存リストにあった
                                if j not in equivalence[k]: #jがリストにない場合登録
                                    equivalence[k].append(j)        
                                break
                            if j in equivalence[k]: #既存のk番目リストに発見(jで検索)
                                find = 1 #既存リストにあった
                                if i not in equivalence[k]:
                                    equivalence[k].append(i)        
                                break
                        if(find == 0):#既存リストにない
                            equivalence[list_number].append(i)
                            if(i != j):
                                equivalence[list_number].append(j)
                            list_number += 1

        #4. Communicateにならないノードを登録
        for i in range(len(p)):
            find = 0
            for j in range(len(p)):
                if i in equivalence[j]:
                    find = 1
                    break
            if find == 0:
                equivalence[list_number].append(i)
                list_number += 1

        #5. エルゴード性の確認(class数が1のとき)
        class_number = 0
        for i in range(len(p)):
            if len(equivalence[i]) > 0:
                class_number += 1

        return equivalence, class_number
    
    #データの保存
    def saveCSVi(self, df, fname):
        pdf = pd.DataFrame(df) #データフレームをpandasに変換
        pdf.to_csv(fname, index=True) #index=Falseでインデックスを出力しない
        
if __name__ == '__main__':
    
    N = 4 #与える
    R = 2 #与える
    K_total = 5 #与える
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full((N, R), 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    type_list = np.full(N, 2) #サービスタイプはPSとする
    #K1 = 1
    #K2 = 2
    #K = [K1, K2]
    #mu = np.array([[1/1, 1/2],[1/4, 1/5],[1/8, 1/10],[1/12, 1/16]])
    #type_list = [2, 4, 4, 3] #Node1:Type2プロセッサシェアリング（Processor Sharing: PS）, Node2:Type4後着順割込継続型（LCFS-PR), Node3:Type4後着順割込継続型（LCFS-PR), Node4:Type3無限サーバ（Infinite Server: IS）, その他Type1(FCFS) 
    #alpha = np.array([[1, 1],[0.4, 0.4],[0.4, 0.3],[0.2, 0.3]])
    
    #bcmp = BCMP_lib(N, R, K, mu, type_list, alpha)
    start = time.time()
    bcmp = BCMP_lib(N, R, K, mu, type_list) #この条件で推移確率を自動生成して、到着率をコンストラクタで求める
    mp_set, exp, tp, rho = bcmp.getBCMP()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    
    print('周辺分布')
    print(mp_set)
    print('平均系内人数')
    print(exp)
    print('スループット')
    print(tp)
    print('利用率')
    print(rho)