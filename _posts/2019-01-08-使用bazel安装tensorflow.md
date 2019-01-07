---
layout:     post
title:      使用bazel安装tensorflow
subtitle:   踩坑大型现场
date:       2019-01-08
author:     xhhszc
catalog: true
tags:
    - tensorflow
    - bazel
---

# 使用bazel安装tensorflow
------
为了提高CPU运行速度，使用SSE/AVX/FMA指令集，需要从source安装tensorflow，其中最简便的就是利用bazel安装tensorflow，但是安装过程也是充满了血泪。。。


##  1. 安装bazel

```shell
conda install bazel
```


##  2. 下载tensorflow代码包
在https://github.com/tensorflow/tensorflow中下载tf包 并上传到服务器中并解压
当然，服务器网速好的话也可以使用指令 $git clone https://github.com/tensorflow/tensorflow


##  3. 进入tensorflow-master文件夹

```shell
cd tensorflow #cd to the top-level directory created
```

##  4. 设置configure（请打起精神！）
这里比较重要，虽然都有提示，但还是很容易踩坑
```shell
./configure
```
**configure 的时候要选择一些东西是否支持，这里建议都选N，不然后面会报错，如果支持显卡，就在cuda的时候选择y,然后按照提示填写自己的cuda cudnn的版本**

整个flow如下(#字为我的注释)：
```bash
(python2bazel) hthong@node150:~/software/tensorflow-master$ ./configure
WARNING: detected http_proxy set in env, setting no_proxy for localhost.
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
INFO: Invocation ID: a54e5cb9-a34d-4848-ae0c-b17d9973d404
You have bazel 0.20.0- (@non-git) installed.
Please specify the location of python. [Default is /home/hthong/anaconda3/envs/python2bazel/bin/python]: 


Found possible Python library paths:
  /home/hthong/anaconda3/envs/python2bazel/lib/python2.7/site-packages
Please input the desired Python library path to use.  Default is [/home/hthong/anaconda3/envs/python2bazel/lib/python2.7/site-packages]

Do you wish to build TensorFlow with XLA JIT support? [Y/n]: Y  #这里可以设置为Y，可以提高tensorflow的运行效率
XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 10.0]: 


Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


Do you wish to build TensorFlow with TensorRT support? [y/N]: n
No TensorRT support will be enabled for TensorFlow.

Please specify the locally installed NCCL version you want to use. [Default is to use https://github.com/nvidia/nccl]: 


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. #去这个官网查自己GPU型号对应的计算能力系数，一般默认值都与官网一致
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.1,6.1,6.1,6.1,6.1,6.1,6.1,6.1]: 


Do you want to use clang as CUDA compiler? [y/N]: n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /home/hthong/anaconda3/envs/python2bazel/bin/gcc]: /usr/bin/gcc
#gcc要求是多版本的（64、32）的gcc，一般使用系统目录下的, 
#对于gcc 5或更高版本的说明：TensorFlow 网站上提供的二进制pip软件包是使用gcc4编译的，高版本gcc涉及一些setting，请自己翻官网


Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
    --config=gdr            # Build with GDR support.
    --config=verbs          # Build with libverbs support.
    --config=ngraph         # Build with Intel nGraph support.
    --config=dynamic_kernels    # (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
    --config=noaws          # Disable AWS S3 filesystem support.
    --config=nogcp          # Disable GCP support.
    --config=nohdfs         # Disable HDFS support.
    --config=noignite       # Disable Apacha Ignite support.
    --config=nokafka        # Disable Apache Kafka support.
    --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```

##  5.编译
编译之前，需安装：numpy，enum
根据是否使用GPU选择以下命令执行：
```bash
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package # CUP-only 

bazel build --config=opt --config=cuda --verbose_failures //tensorflow/tools/pip_package:build_pip_package # GPU support
```

这里我遇到了一个坑：
```bash
ERROR: missing input file '@pasta//:LICENSE'
ERROR: /home/shiki/tensorflow/tensorflow/tools/pip_package/BUILD:235:1: //tensorflow/tools/pip_package:build_pip_package: missing input file '@pasta//:LICENSE'
Target //tensorflow/tools/pip_package:build_pip_package failed to build
Use --verbose_failures to see the command lines of failed build steps.
ERROR: /home/shiki/tensorflow/tensorflow/tools/pip_package/BUILD:235:1 1 input file(s) do not exist
```
根据网友的智慧：https://github.com/tensorflow/tensorflow/issues/24722
删掉了（注释掉也行）tensorflow/tools/pip_package/BUILD文件中的第L172行代码（"@pasta//:LICENSE"），完美解决

##  6.生成whl
过了很久之后，第五步完成，生成了一个uild_pip_package脚本。然后我们就可以根据这个脚本生成whl文件了

The bazel build command builds a script named build_pip_package. Running this script as follows will build a .whl file within the /home/hthong/software/tensorflow_pkg directory
```bash
bazel-bin/tensorflow/tools/pip_package/build_pip_package /home/hthong/software/tensorflow_pkg
```


##  7.安装
终于可以安装了：
```shell
pip install /home/hthong/software/tensorflow_pkg/your_tensorflow_whlname.whl
```

提示成功之后就可以验证tensorflow是否安装成功了，记住退出当前tensorflow文件夹后再进行验证，不然会fail

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
你可以看到,没有warning了！！！tensorflow速度是不是变快了！！


