# -*- coding: utf-8  -*-
# -*- author: jokker -*-



"""
* 总结 git 的常用命令


* git clone --recursive
    * --recursive 加载子 git 项目

* git init

* git add .

* git commit -m

* git remote -v

* git remote add region

* git fetch

* git pull origin master

* git push origin master

* git checkout

* git stash

* ---------------------------------

* git diff

* git merge


# 分支
* git branch
* git checkout branch_name

# 版本
* git tag
* git tag -l "v1.8.5*" [查找标签]
* git checkout -b branch_name tag_name
* git tag -a v1.4 -m "my version 1.4" [创建标签]

# 子模型
* git submodule add ssh://git@192.168.3.108:2022/aigroup/saturn_lib.git lib  [增加子模型，相当于是两个项目，最后的 lib 是重新给的名字]
* delete submodule
    * rm -rf 子模块目录 删除子模块目录及源码
    * vi .gitmodules 删除项目目录下.gitmodules文件中子模块相关条目
    * vi .git/config 删除配置项中子模块相关条目
    * rm .git/module/* 删除模块下的子模块目录，每个子模块对应一个目录，注意只删除对应的子模块目录即可

"""




