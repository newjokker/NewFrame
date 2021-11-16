# DockerFile

* RUN
    * RUN echo helloword
    * RUN ['func', 'parameter_1', 'parameter_2']
    * &&, joint RUN with && 
    
* COPY 
    * /local/path/file /Image/path/file
    
* ADD
    * ADD file /images/path/file
    * ADD latest.tar /var/www/
    
* EXPOSE 
    * EXPOSE host
    
* CMD
    * use only once
    * CMD ['echo', 'hello docker file']
    
* WORKDIR
    * WORKDIR /path/to/workdir


# ----------------------------------------------------------------------------------------------------------------------

* RUN，dockerfile 最后一行使用 run 程序一直执行下去
    * RUN /v0.0.1/script/allflow_wuhan.py
    
* 

# ----------------------------------------------------------------------------------------------------------------------

* docker diff containerID
    * 相比于原始镜像，容器中改变了哪些文件

* docker cp 
    * docker cp containerID:file/path/cointer /file/path/local
    * cp 可以直接将宿主机复制到容器中
    
* ctrl + p , ctrl + q 
    * 退出容器的虚拟终端，docker 是在运行
    * 退出之后使用 docker exec 重新进入容器  

*   























 