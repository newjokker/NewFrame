# docker hub 相关操作

* login
    * check the config
        * vim /etc/docker/daemon.json
        * {"insecure-registries":["192.168.3.108:7080"]}
    * docker login http://192.168.3.108:7080/
        * ldq
        * Txkj@2021

* tag
    * docker tag wuhan_ft:v2.1.2 192.168.3.108:7080/wuhan_ft/wuhan_ft:v2.1.2
    
* push
    * docker push 192.168.3.108:7080/wuhan_ft/wuhan_ft:v2.1.2
    
* pull 
    * docker pull 192.168.3.108:7080/wuhan_ft/wuhan_ft:v2.2.0


