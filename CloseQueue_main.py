import CloseQueue_lib as mdl
import numpy as np

N = 5
K = 3
p = np.array([[0, 0.5, 0.5], [0, 0, 1.0], [1.0, 0, 0]]) #閉鎖型の例
mu = [1,2,2] #各ノードのサービス時間
qlib = mdl.CloseQueue_lib(N, K, p, mu)

#閉鎖型 : ノード到着率を求める
alpha = qlib.getCloseTraffic() #α1 = 1とする
print('各ノードへの到着率 : {0}'.format(alpha))

#畳み込みの計算
qlib.getGNK()

#Throughputの計算
throughput = qlib.getThroughput()
print('ThroughPut : {0}'.format(throughput))

#稼働率計算
avail = qlib.getAvailability()
print('Availability : {0}'.format(avail))

#github https://github.com/smzn/CloseQueue_lib