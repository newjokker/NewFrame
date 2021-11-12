# saturn_lib
* 检测公用库，所有工程所用检测框架（faster-rcnn,r2cnn等）均包含在saturn_lib下
* 请确认是将可以公用的库置于本工程中，私有的代码请置于工程代码scripts文件夹下

# 结构
## detect_libs
* 抽象检测类，最终使用的抽象类

## detect_utils
* 检测工具，公用的检测工具代码


## faster_libs
* faster-rcnn所用的库文件

## r2cnn_libs
* r2cnn所用的库文件

# 改写规则
* a 如检测工程使用的X框架需要特定的库文件支持，那么就先在本工程根目录建立X_libs的文件夹，包含所有会用到的库
* b 在detect_libs中建立抽象类，调用X_libs中的库，实现通用函数，如detect，warmup等
* c 将本工程作为submodule加入检测工程（先创建自己的工程，然后在该工程下执行git submodule add ssh://git@192.168.3.108:2022/aigroup/saturn_lib.git lib）


# 如何git clone 含有子模块的项目
* 在执行 git clone 时加上 --recursive 参数。它会自动初始化并更新每一个子模块
* 如git clone --recursive https://github.com/example/example.git
* 项目已经克隆到了本地，执行下面的步骤：
    初始化本地子模块配置文件
    git submodule init
    更新项目，抓取子模块内容。
    git submodule update

# 同时更新子模块lib和工程代码
* 这种时候请首先更新lib子模块，再更新主工程

# 查看lib子模块状态发现Head分离的情况导致无法更新子模块代码到master分支
* 这种情况需要将子模块指向master分支
* git checkout master
* 更好的办法是直接在添加的时候就指定master分支
* git submodule add -b master [URL to Git repo]