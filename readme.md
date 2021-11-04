# 说明


* 增加速度可以通过增加每一次推理的 batch size , 推理的线程的个数

* 直接增加 batch size 占满 GPU 就行，这样就不存在多个 model 之间竞争资源的问题了

*  


### 性能绑定

#### 锁核

`
cpu_num = 1                                                 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
`

#### 锁频

* 4.6 -> 2.2 

* refer : https://blog.csdn.net/weixin_32759777/article/details/118541117