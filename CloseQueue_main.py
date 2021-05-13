import CloseQueue_lib as mdl
import numpy as np

#p = np.array([[0, 0.5, 0.5], [0, 0, 1.0], [1.0, 0, 0]]) #閉鎖型の例(ORの基礎 P155)
#QN&MC(P292 EX 7.5)
p = np.array([[0.6, 0.3, 0.1],[0.2, 0.3, 0.5],[0.4, 0.1, 0.5]])
N = 3
K = 3
mu = [0.8, 0.6, 0.4] #各ノードのサービス時間

qlib = mdl.CloseQueue_lib(N, K, p, mu)

#閉鎖型 : ノード到着率を求める
alpha = qlib.getCloseTraffic() #α1 = 1とする
print('各ノードへの到着率 : {0}'.format(alpha))

rho = alpha / mu
#畳み込みの計算
qlib.getGNK(rho)

#Throughputの計算(ρ*μ)
throughput = qlib.getThroughput()
print('ThroughPut : {0}'.format(throughput))

#稼働率計算(Utilization)(1-π(0))
avail = qlib.getAvailability()
print('Availability : {0}'.format(avail))

#定常分布の計算
pi = qlib.getStationaryDistribution()
print('Stationary Distribution')
print(pi)

#平均系内人数
l = qlib.getLength(pi)
print('Mean Length : {0}'.format(l))


#github https://github.com/smzn/CloseQueue_lib