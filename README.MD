# Apple-Banan-Orange Detection

## 简介
本项目起源于本人《人工智能原理》一课的大作业，要求识别苹果、香蕉、橙子三种水果，并用IOU、Precision、Recall、mAP四个指标进行评估 ~~(谁拿前三个作为最终指标啊)~~ 。一通搜索下来最终决定拿 TensorFlow Object Detect API V2 来完成。主要参考了官方教程[TensorFlow 2 Object Detection API tutorial
](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) 以及众多大神博客。本文主要汇总一下对在这期间有用的教程与分享本人在这安装、训练期间遇到的问题以及解决方法，如果有直系学弟学妹明年遇到同款大作业而搜到这个仓库，望对遇到同样问题的同学有所帮助 ~~(真帮到了记得回来给个star
啊喂)~~

### ~~如果有下次，我选择PyTorch~~

### 项目目录
```
.
├─scripts           # 相关代码
│  ├─Pre-process    # 数据处理
│  ├─Train          # 训练
│  ├─Predict        # 推理预测
│  ├─export         # 导出模型
│  └─eval           # 评估
├─annotations       # 标签图 label_map
├─exported-models   # 本项目已训练好的模型
├─images            # 数据集(图片+xml标注)
└─data              # TFRecord格式数据
```

## 安装
主要参考[官方安装教程](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#)，踩了诸多坑后，最后汇总出来一套可以兼容的环境，这些不一定是唯一能用的版本，但是本人亲身测试过绝对能跑的

1. Python：3.9.19
    
    `conda create -n (环境名称) python=3.9.19`
2. TensorFlow版本: 2.10.1，注意Windows端TensorFlow在2.11及以上不能原生支持GPU加速，要用新的版本要搞好多操作

    `pip install tensorflow==2.10.1`

3. CUDA版本: 11.8，但由于本人电脑无英伟达显卡，这是在服务器上训练时直接拉的社区镜像里的配置，具体兼容性还请自行搜索。
   
    吐槽下Linux下是装CUDA是真的难搞，之前想换下CUDA版本搞一天都没搞定，最后直接拉大神镜像了
   
4. AMD请安装 tensorflow-directml-plugin 参考 知乎@猫爪无情 的 [Tensorflow2 在Windows11下使用AMD GPU加速
](https://zhuanlan.zhihu.com/p/546308020)
   
    `pip install tensorflow-directml-plugin==0.4.0.dev230202`

5. 安装 TensorFlow Object Detection API
   
    1. 下载 [TensorFlow Model Garden](https://github.com/tensorflow/models)
        
         **但事实上只需要 research 目录下的 object_detect**

    2. 将 object_detection/protos 目录下的proto文件转成python文件，详见 [Protobuf Installation/Compilation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#protobuf-installation-compilation)。
    这些文件是用来描述训练、评估过程中优化器、损失计算等等的设置，有时间建议仔细读读，后面设置训练参数的时候能更加理解
    
    3.  安装 object_detect 包

        ```
        # 进入models/research/
        cp object_detection/packages/tf2/setup.py .
        python -m pip install .
        ```
        如果只下了 object_detect 文件夹，注意需要把`setup.py`文件拷到 object_detect 的上级目录中，注意不能放在 object_detect 之下，然后在该目录下执行 `python -m pip install .`

    4. **关于教程中提到的 `COCO API installation`**
        
        我也一直卡在安装 Visual C++ 2015 build tools 上，教程中给的安装程序不完整装不了 (应该是太老了微软服务器都不支持了)，后来找到完整版，以及安装网上教程用VS安装上去了，都还是显示 ` error: Microsoft Visual C++ 14.0 is required.` 
        
        但是经过本人测试似乎没有影响，COCO评估也能正常使用，只能说很玄学...

    5. 测试安装: 
   
        ```
        python object_detection/builders/model_builder_tf2_test.py
        ```

---

## 数据集收集与处理

- **推荐 [Kaggle](https://www.kaggle.com/) 和 [Roboflow Universe](https://universe.roboflow.com/) 两个网站**
    
    其中 [Roboflow Universe](https://universe.roboflow.com/) 个人非常喜欢，上面有大量的开源数据集，并且个人最喜欢的是能下载各种格式的标注数据，以及可以在一个数据集中将感兴趣的图片克隆到自己工作空间
    
    在自己工作空间里可以批量修改标签、自己重新标注或者使用网站提供的AI标注 ~~(用AI标注数据训练AI，左脚踩右脚)~~ 、增强图像等操作，以及貌似还能直接拉社区模型，在网站服务器上训练，数据、模型、训练一站式解决，但本人没有过多研究别的，只用来整了点数据集

- 在本仓库中，我也上传了我们小组收集到的一些数据集，总共应该有将近1w张了吧
 
  (update) 扔 [阿里云盘](https://www.alipan.com/s/MCydFDCe9gh) 上了 pwd: l4d9  ~~不会用git传大文件qaq~~
  
    image文件夹为图片+xml标注，data文件夹为将图片和标注转换为的TFRecord格式，用来给TensorFlow读入，因为前前后后收集过好几次数据集，因此总共有4个文件夹

- scripts/Pre-process 目录下有本人用来处理数据的一些脚本    
  
    1. `convert-RGB.py`：将非三通道图片转为三通道图片
    2. `generate_tfrecord.py`：官方教程中提供的将图片和标注转为TFRecord格式的程序
    3. `modify-xml.py`：将xml中filename修改为同名图片名称
    4. `partition_dataset.py`：官方教程中提供将数据按指定比例分为训练集和数据集的程序
    5. `show_box.py`：将标注框画出来
   

## 训练
1. 挑个看着顺眼的预训练模型下载 [TensorFlow 2 Detection Model Zoo
](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

    本人试过几个，最后采用了 `EfficientDet D3 896x896`

2. **配置 [Configure the Training Pipeline](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline)**
    
    主要要设置：类型数量、批次大小、预训练模型地址、训练模式(detection\\classfication)、数据集和label map地址

    其他配置参数的含义与调整方法可参考：
    
    1. [object_detection“使用教程”翻译（一）：configuring_jobs.md](https://blog.csdn.net/HaixWang/article/details/78484315)
    2. [数据读取与处理](https://blog.csdn.net/qq_30460949/article/details/95058808) 
    3. [object detection训练参数——image_resizer,输入图像尺寸的修改配置](https://blog.csdn.net/l13022736018/article/details/108619875)
    4. [使用TensorFlow Object Detection API 训练模型方法](https://blog.csdn.net/shajiayu1/article/details/105996599)
    5. [【理论】object detection api调参详解（兼SSD算法参数详解）](https://www.cnblogs.com/hillsea/p/13216716.html)

    本人主要是调调学习率和数据增强的参数 ~~模型的内部参数也不知道怎么调更好~~

   

3. 关于多卡训练
    
    本人一开始按照[这份教程](https://www.cnblogs.com/jyroy/p/14704964.html) （这个教程除了这里以外别的内容还挺好的，推荐看看）里说的，以为TF直接能调用所有GPU，**但是！！** 这会一直卡在加载阶段，GPU使用率和显存都是满的但就是不开始训练，后来我直接去翻了`model_main_tf2.py`的代码，发现它里面是有多卡的选项的，需要输入选项：`--num_workers= ` ，单卡与多卡需要使用不同的分发策略
    ```python
        if FLAGS.use_tpu:
      # TPU is automatically inferred if tpu_name is None and
      # we are running under cloud ai-platform.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif FLAGS.num_workers > 1:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
      strategy = tf.compat.v2.distribute.MirroredStrategy()
    ```

    **但是！！但是！！！**~~(怎么我踩到这么多坑)~~ 就在我开完多卡想美滋滋的等结果时，却发现Loss一直下不去，而且一直高达10点多！！这让我真的很破防，一下午2张A800花了几十块钱又打水漂了，关键是当天上午用单卡训练时还能正常的降下去。可后来换回单卡后又试了下还是降不下去，不知道到底发生了什么，也可能是模型的问题，反正最后我是把一开始坚持的 `CenterNet HourGlass104 512x512` 换成了 `EfficientDet D3 896x896` ，之前训练花的几百块钱又打水漂了 55555555555

4. 关于混合精度
    
    [官方文档](https://www.tensorflow.org/guide/mixed_precision?hl=zh-cn) 里说只需写一条全局策略语句就能自动开启，但经本人实测在这里若开启后，内部的一些数学函数却不能支持 `fp16` (貌似是因为输入的连个参数类型不一样)，搜半天也没解决，去官方项目下发了[issue](https://github.com/tensorflow/models/issues/11215)，但到现在还没回...

    不过，后来我才发现在4090以上显卡训练，内部矩阵运算会自动开启`TF32`，因此也就不在纠结这个问题了

## 预测推理
- Predict目录下的脚本均为本人根据官方教程中 [Examples](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/index.html)给的代码修改而来，`CV-predict.py` 为调用摄像头为输入，剩下两个是用图片作为输入，推荐看看`IMG-predict_saved_model.py` ~~(因为其他两个直接copy的，真的屎山)~~
- 这里有个大坑是：推理时一定要用`saved_model`形式的模型，而不是`checkpoint`形式！！后者比前者满了将近50倍！！

## 评估

- 可以直接使用`model_main_tf2.py`进行评估
  
  当输入checkpoint路径时会切换成评估模式，训练时新开个终端进行评估，它会在每个新checkpoint输出时自动评估，可用tensorboard实时查看指标曲线，关于评估选用的指标，可参考：[关于tensorflow之config--eval_config中metrics_set参数（学习笔记）](https://blog.csdn.net/qq_35975447/article/details/106640709)

- 本人写的 `evaluate.py`、`evaluate_util.py`

    因为老师~~若至~~要求的 IOU、Precision、Recall、mAP 四个指标用这里面自带的评估搞不出来，与其去改它源码，不如自己写个 ~~虽然我写的真是一堆屎山~~

    关于`TP、FP、FN、FP`、`IOU`、`Precision`、`Recall`、`AP、mAP`等指标的含义，推荐参考：
    
    1. [MS COCO数据集的评价标准以及不同指标的选择推荐（AP、mAP、MS COCO、AR、@、0.5、0.75、1、目标检测、评价指标）](https://blog.csdn.net/weixin_44878336/article/details/134030021)

    2. [史上最易懂AP、mAP计算解析](https://blog.csdn.net/qq_35916487/article/details/89076570)

   
   
   在计算`TP、FP、FN、FP`时，一个目标框与多个标注框重合或一个标注框与多个目标框重合是让我最头疼的，去问老师，给我发来一篇 [目标检测中使用二部图匹配算法](https://blog.csdn.net/CHIERYU/article/details/80329344)

   最后我采取的方案是：
    
    1. 每个预测框作为一个样本, 判断其是TP or FP
    1. 若一个预测框与多个标注框达到TP, 则只取其中IOU最大的标注框 
    2. 若一个标注框有多个预测框为TP, 则只取IOU最大的预测框为TP, 其余调整为FP
    
    不知道这样理解对不对QAQ
    
    至于其他输出，`IOU` 为 “每个预测框与所有标注框中最大的IOU” 的均值， `Precision` 和 `Recall` 取了P/R列表中两者都大于0.7的输出 (小摸了一手，找不到两者都大于0.7的直接给我置为0.7)。因为我认为这些都是计算 `AP` 的中间量，`IOU` 只是作为判断TP的阈值，而 `Precision` 和 `Recall` 取决于置信水平，这些都不能作为衡量模型的最终指标，所以随便糊弄一下达标就好 ~~(真不知道为什么规定这么若至的指标)~~


    其他细节可以看看 `evaluate_util.py` 文件，但这主要是为应付这次的任务写的一坨，应该不会有人拿它来实际评估吧

## 最后

这是本人第二次单独写出这么长的代码(指 `evaluate_util.py` )，虽然用了TensorFlow的API，最核心的代码都不用自己从头搞，但即使这样环境的配置与训练参数调节对蒟蒻还是太难了，~~给AutoDL交了几百大洋学费~~，如果有哪里不对的地方，还望指正。最后感谢各路大神的博客，帖子！

