# 说明


* 增加速度可以通过增加每一次推理的 batch size , 推理的线程的个数

* 直接增加 batch size 占满 GPU 就行，这样就不存在多个 model 之间竞争资源的问题了

*  
