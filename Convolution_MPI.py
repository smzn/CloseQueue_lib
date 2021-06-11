import numpy as np
from numpy.linalg import solve 
import itertools #重複組合せを求める
import numpy as np
import time
from mpi4py import MPI

class Convolution_MPI:

    def __init__(self, N, K, p, mu, m, rank, size):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.K = K #網内の客数
        self.p = p
        self.alpha = np.array([])#ノード到着率
        self.mu = mu
        self.rho = np.array([])
        self.pi = np.zeros((len(p),len(p)))
        self.m = m
        self.comb = self.fact(N + K -1) // (self.fact(N + K -1 - (N -1)) * self.fact(N -1))
        self.rank = rank
        self.size = size
        print('rank = {0}, size = {1}'.format(self.rank, self.size))
        
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
    def calcConvolution(self, n, k, rho): #メモ化利用なし
        #rho = alpha / mu
        if k == 0: #Gj(0) = 1 (j = 1,...,N)
            g = 1
        elif n == 0:#G(0,k) = f0(k)
            g = np.power(rho[n], k)
        else : 
            g = self.calcConvolution(n-1, k,rho) + rho[n] * self.calcConvolution(n, k-1,rho)
        return g

    def initMemo(self, rho, memo):
        for i in range(self.N):
            memo[i][0] = 1
        for i in range(self.K+1): #(k = 0,1,2,...,K)
            memo[0][i] = np.power(rho[0], i)
        return memo
            
    #メモ化を利用して計算を早くする(畳み込みのインデックスを再度確認)
    def calcConvolution_memo(self, n, k, rho, memo):
        if memo[n][k] >= 0:
            g = memo[n][k]
        else : 
            memo[n][k] = self.calcConvolution_memo(n-1, k,rho,memo) + rho[n] * self.calcConvolution_memo(n, k-1,rho,memo)
            g = memo[n][k]
        return g
    
    # Closed Queueで窓口数が複数の場合の対応 : QN&MC P289 式(7.62) 2021/06/03追加したかったけどやめた
    
    def getGNK(self, rho):#メモ化利用なし
        self.g = self.calcConvolution(self.N-1, self.K, rho) #ノード番号は1小さくしている
        self.gg = self.calcConvolution(self.N-1, self.K-1, rho) #ノード番号は1小さくしている
        
    def getGNK_memo(self, rho):
        #メモ化の準備
        memo = np.ones((self.N, self.K+1))#メモ化で利用(n = 0,1,...,N-1, k = 0,1,2,...,K)
        memo *= -1
        memo = self.initMemo(rho, memo)
        self.g = self.calcConvolution_memo(self.N-1, self.K, rho, memo) #ノード番号は1小さくしている
        self.gg = self.calcConvolution_memo(self.N-1, self.K-1, rho, memo) #ノード番号は1小さくしている
        
    def getThroughput(self):
        throughput = self.alpha * self.gg / self.g
        return throughput

    def getAvailability(self):
        avail = self.alpha / self.mu * self.gg / self.g
        return avail

    #http://www.ieice-hbkb.org/files/05/05gun_01hen_05.pdf
    #畳み込みを利用する場合、周辺分布までしか算出できない
    def getStationary_memo(self, p, rho, memo): #memoを追加
        pi = []
        for i in range(self.K+1):
            pi.append(np.power(rho[self.N-1], i) * self.calcConvolution_memo(self.N-2,self.K-i, rho, memo) / self.calcConvolution_memo(self.N-1, self.K, rho, memo))
        return pi
    
    def getStationary(self, p, rho): #メモ化なし
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

    def getStationaryDistribution(self, memo_flag):#メモ化を追加
        #定常分布を求めたいノードをノードN(最後のノード)に持ってくる
        #推移確率、α、μの順番を変更
        #MPIで並列化する
        #pi_all = np.zeros((self.N, self.K+1)) #全状態確率を入れるリスト
        pi_all = np.zeros((self.N, self.K+1)) #並列計算用
        #for i in range(self.N):
        for i in range(self.rank, self.N, self.size):
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
            
            if memo_flag == 1:#メモ化実施
                #メモ化の準備
                memo = np.ones((self.N, self.K+1))#メモ化で利用(n = 0,1,...,N-1, k = 0,1,2,...,K)
                memo *= -1
                memo = self.initMemo(rho, memo)
                pi = self.getStationary_memo(p, rho, memo) #memoを追加
            else:
                pi = self.getStationary(p, rho) #memoなし
            
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

    def getG(self):
        print('G = {}'.format(self.g))
        
    def getGG(self):
        print('GG = {}'.format(self.gg))

        
if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #ガーデンパークモデル
    p = np.array([[0.0011003157101162463, 0.11024952952774816, 0.25380066059219375, 0.07343233522837377, 0.06562710435865776, 0.06991547626707886, 0.10609673473849808, 0.050680951566158115, 0.05416340360982934, 0.08843233248854096, 0.04622571708690421, 0.08027543882590069], [0.11224109903595923, 0.0011201920327369394, 0.22437474745112732, 0.0928160117262778, 0.06633208389032944, 0.07693487786139194, 0.0911065034259741, 0.05332880988786884, 0.056382385511375775, 0.09231541902287667, 0.04889709925335951, 0.08415077090072237], [0.12955652851817565, 0.11250332700140227, 0.0022466936586225217, 0.08550794194527646, 0.0776898265505006, 0.08511741760247529, 0.15651348601067064, 0.05643628307885522, 0.06111724492501443, 0.0963192863712463, 0.05063821571530522, 0.08635374862245535], [0.07665739572877639, 0.0951731533194184, 0.1748665012871232, 0.0011486402625580597, 0.07542601839408736, 0.10514011140095426, 0.08057724693680487, 0.0638584803679966, 0.06692754987640072, 0.10640262466933859, 0.057928360318264496, 0.09589391743827712], [0.06664181003065499, 0.06616251682383482, 0.15454719932124522, 0.07336991281425088, 0.001117328445981307, 0.10348905196528554, 0.08398618756577869, 0.07627412941786682, 0.09340052331976309, 0.11879790829042057, 0.06222474357179144, 0.09998868843312679], [0.0681419912175087, 0.07365285783380676, 0.16251497462228193, 0.0981619576858847, 0.09932815451211947, 0.0010724049589365796, 0.08165475365321273, 0.07076158082143474, 0.07888490198237387, 0.11033017852493987, 0.060002700564109945, 0.09549354362339066], [0.09668996596538912, 0.08155556536622668, 0.27942457521683006, 0.07034368131283429, 0.0753743734259102, 0.07635181447274138, 0.0010027593104024013, 0.051520265898476574, 0.05653585258439637, 0.08756788190780167, 0.04576475369666503, 0.07786851084232631], [0.049555419522444535, 0.0512192391518401, 0.10810312423511087, 0.059813367601586495, 0.07344464132470596, 0.07099085772581337, 0.05527708124300974, 0.0010758796931973325, 0.1237335345443852, 0.1766119256253121, 0.10286475980657848, 0.12731016952601582], [0.05390055332590845, 0.05511319101775745, 0.11914737822294351, 0.06380070548143595, 0.09153201937820726, 0.0805452004313421, 0.061735054057663746, 0.1259297376165996, 0.001094975973734647, 0.15076739806863848, 0.0805226919583598, 0.11591109446740913], [0.0515321453650312, 0.05284041535248883, 0.10995471190034087, 0.059395393320323064, 0.06817299833974819, 0.06596597171965457, 0.055992853305160135, 0.10525452414933562, 0.08828507837460474, 0.0025647465141094327, 0.13369713298865907, 0.2063440286705443], [0.046205696791962056, 0.04800868354393604, 0.09915707496628882, 0.05546721764192715, 0.06125075419973466, 0.06153768642090796, 0.050195363897146836, 0.10515552271976615, 0.08088032285832025, 0.2293331403860839, 0.001099839164884842, 0.1617086974090415], [0.054505112721089044, 0.05612256182278858, 0.11486010117757178, 0.062370452235229475, 0.06685623007833497, 0.06652533400798188, 0.0580145641621276, 0.0884038651827311, 0.0790846659133489, 0.24042482920081087, 0.10984393076216376, 0.0029883527358219083]])
    N = 12 #ノード数
    K = 10 #客数
    mu = [5, 5, 10, 5, 5, 5, 5, 5, 5, 10, 5, 10]
    m = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    start = time.time()
    
    qlib = Convolution_MPI(N, K, p, mu, m, rank, size)

    #閉鎖型 : ノード到着率を求める
    alpha = qlib.getCloseTraffic() #α1 = 1とする
    if rank == 0:
        print('各ノードへの到着率 : {0}'.format(alpha))

    rho = alpha / mu

    #メモ化用変数の初期化
    #qlib.initMemo(rho)

    #畳み込みの計算
    #qlib.getGNK(rho) #メモ化利用なし
    qlib.getGNK_memo(rho) #メモ化利用

    #Throughputの計算(ρ*μ)
    throughput = qlib.getThroughput()
    if rank == 0:
        print('ThroughPut : {0}'.format(throughput))

    #稼働率計算(Utilization)(1-π(0))
    avail = qlib.getAvailability()
    if rank == 0:
        print('Availability : {0}'.format(avail))

    #定常分布の計算(周辺分布) 
    #計算に時間がかかる
    memo_flag = 1 #1のときメモ化あり
    pi_marginal = qlib.getStationaryDistribution(memo_flag)

    #データの集約
    if rank == 0:
        for i in range(1, size):
            marginal =+ comm.recv(source=i, tag=11)
            pi_marginal = np.add(pi_marginal, marginal)
    else:
        comm.send(pi_marginal, dest=0, tag=11)
    if rank == 0:
        print('Marginal Stationary Distribution')
        print(pi_marginal)
        
    #平均系内人数
    l = qlib.getLength(pi_marginal)
    if rank == 0:
        print('Mean Length : {0}'.format(l))
        
    #定常分布(個々の定常分布)例 pi({1,1,1})
    #k = [2,0,1]
    combi_k = qlib.getCombi(N, K)
    for i in range(len(combi_k)):
        pi = qlib.getPi(combi_k[i],rho)
        ''' #定常分布の表示は時間がかかりすぎる
        if rank == 0:
            print('pi({0}) = {1}'.format(combi_k[i], pi))
        '''
        
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    
    #mpiexec -n 8 python3 Convolution_MPI.py