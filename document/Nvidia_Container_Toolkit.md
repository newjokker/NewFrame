#### Nvidia Container Toolkit 安装

首先，要保证 Nvidia 显卡驱动已经安装。

安装成功后，接着安装 Nvidia Container Toolkit，该工具使 Docker 的容器能与主机的 Nvidia 显卡进行 interact。更多信息访问 Nvida Container Toolkit 的官方网站。安装前要求nvidia的驱动已安装，但不要求安装CUDA，同时，docker 版本是 19.03

```shell
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
```

上条命令如果输出：gpg: no valid OpenPGP data found.，可以多试几次，或者那么把命令分开执行

```shell
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

或者下面一行

curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu18.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

该命令成功后可以查看 `cat /etc/apt/sources.list.d/nvidia-docker.list`，会有如下内容:

```shell
deb https://nvidia.github.io/libnvidia-container/stable/ubuntu16.04/$(ARCH) /
#deb https://nvidia.github.io/libnvidia-container/experimental/ubuntu16.04/$(ARCH) /
deb https://nvidia.github.io/nvidia-container-runtime/stable/ubuntu16.04/$(ARCH) /
#deb https://nvidia.github.io/nvidia-container-runtime/experimental/ubuntu16.04/$(ARCH) /
deb https://nvidia.github.io/nvidia-docker/ubuntu16.04/$(ARCH) /
```

```shell
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker
sudo docker run --gpus all --rm nvidia/cuda:10.0-base nvidia-smi
```

