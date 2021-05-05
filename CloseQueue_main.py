import CloseQueue_lib as mdl
import numpy as np

p = np.array([[0, 0.5, 0.5], [0, 0, 1.0], [1.0, 0, 0]]) #閉鎖型の例
qlib = mdl.CloseQueue_lib(p)

#閉鎖型の場合
alpha = qlib.getCloseTraffic() #α1 = 1とする
print('各ノードへの到着率 : {0}'.format(alpha))
