# 深度学习文档
## 动手学深度学习知识（https://zh-v2.d2l.ai/）

### 基础数学知识（只学习概念，具体遇到了再查找）

1. **张量与线性代数**
   （内容来自[什么是张量？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/140260245)）：
   	eg.数组就是一维张量,普通的矩阵就是二维张量（两个或多个下标值对应一个元素，是多对一的关系,）空间矩阵就是三维张量
   		比如对于一维张量[1,2,3]，

在PyTorch中可以表达为torch.tensor([1,2,3])

对于二维张量 (1 2 3 
						 4 5 6) ，

在PyTorch中可以表达为torch.tensor(**[[1,2,3],[4,5,6]]**)

聪明的读者可能发现了，多一个维度，我们就多加一个**[]**

所以对于上图中的“空间矩阵”，我们可以这样表达：

torch.tensor(**[[[9,1,8],[6,7,5],[3,4,2]],[[2,9,1],[8,6,7],[5,3,4]],[[1,5,9],[7,2,6],[4,8,3]]]**)

上面可能不容易看清楚，我们把它写的再好看一些：

```text
torch.tensor([
[[9,1,8],[6,7,5],[3,4,2]],
[[2,9,1],[8,6,7],[5,3,4]],
[[1,5,9],[7,2,6],[4,8,3]]
])
```

可以看到上面中间三行的每一行都是一个矩阵，所以我们就知道了,每多一个方向的数据，我们就多加一个[]

再重复一下形状的概念，这个概念也很重要，

对于矩阵 (123456) ，它有两行三列，我们说它的形状是（2，3）

对于上面的那个空间矩阵，在x轴、y轴、z轴上它都是三个数，所以我们说它的形状是（3，3，3）

### 线性神经网络

1. **线性回归** 
   y=w1*x+w2*x+b    其中w为权重，b为偏置量 
   线性代数表示：y = $\mathbf{w}^\mathrm{T}*x+b$	向量x对应于单个数据样本的特征,w为一个权重向量
   
   ​							y=$X*w+b$      X矩阵的每一行是一个样本，每一列是一个特征
2. **损失函数**



## 任务1 数据处理

1. IMDB数据集下载 [Sentiment Analysis (stanford.edu)](https://ai.stanford.edu/~amaas/data/sentiment/)
2. 读取数据：
* python的os标准库学习:
  * os.path.join(folder_path,filename)构建当前文件位置
  * os.listdir 以列表形式呈现当前文件位置的所有文件名
3. 词表构建
*    为什么要构建词表
     *    自然语言处理模型都是基于统计机器学习，只能在数学上进行各种计算，这样就势必要求将字符串表示的文本数字化
*    词表的基本形式
     *    将所有的词囊括进来，然后在文本中对每一个词都能够在这个词表里面查到它的索引
*    词表的代码实现（[构建词表与抽样——【torch学习笔记】_load_data_time_machine-CSDN博客](https://blog.csdn.net/weixin_43180762/article/details/125100217?ops_request_misc=%7B%22request%5Fid%22%3A%22170634104316777224472380%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170634104316777224472380&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~sobaiduend~default-2-125100217-null-null.nonecase&utm_term=构建词表&spm=1018.2226.3001.4450)
     *    预处理：去除空行空格，大小写转换。最简单为直接使用split()函数
          				此外还有去除句末的句号. replace函数可以预先去除句号.
     *    词表构建：
          *    counter类（collections库中）统计次数（其中counter类返回的是一个字典）
          *    sort函数进行排序
          *    items函数（将字典转换为一个可迭代的形式）eg.原counter函数后为{'name':'xin','sex':'male','job':'AI'}，转换为dict_items([('name', 'xin'), ('sex', 'male'), ('job', 'AI')])
          *    特殊标记：字典里的<unk>表示字典中没有出现过的词，而<pad>表示padding，因为很多模型一般都要求输入的语料长度一致，所以一种常用的处理方式就是把每一句话都跟[语料库](https://www.zhihu.com/search?q=语料库&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A802442061})中曾经出现过的最长的一句话对齐——实际不足的部分就用padding补齐。所以我们可以看到，第一句话的后面全都是表示padding的ID 1。
          *    最新发现来自https://www.zhihu.com/question/44832436/answer/802442061 torch中的torchtext可以辅助进行词表构建（将在任务二中详细阐述）

4. dataloader、Dataset学习([Dataset,Dataloader详解_dataset dataloader-CSDN博客](https://blog.csdn.net/junsuyiji/article/details/127585300?utm_medium=distribute.pc_feed_404.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-127585300-blog-null.262^v1^pc_404_mixedpudn&depth_1-utm_source=distribute.pc_feed_404.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-127585300-blog-null.262^v1^pc_404_mixedpud)）

* dataloader的意义：果数据量很大，考虑到内存有限、I/O 速度等问题，在训练过程中不可能一次性的将所有数据全部加载到内存中，也不能只用一个进程去加载，所以就需要多进程、迭代加载，而 **DataLoader** 就是基于这些需要被设计出来的。
  
* dataloader原理：引出迭代器的原理和dataloader具体实现[带你从零掌握迭代器及构建最简 DataLoader - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/340465632)**（暂存，目前还看不太懂）**
  
* dataloader基本实现：**DataLoader** 是一个迭代器，最基本的使用方法就是**传入一个 Dataset 对象**，它会根据参数 **batch_size** 的值生成一个 batch 的数据，节省内存的同时，它还可以实现多进程、数据打乱等处理。
  
* dataset代码实现：最简概述[pytorch笔记5-数据读取机制DataLoader - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/399073311)
  **Dataset是用来解决数据从哪里读取以及如何读取的问题**。pytorch给定的Dataset是一个抽象类，所有自定义的Dataset都要继承它，并且复写__getitem__()和__len__()类方法，__getitem__()的作用是接受一个索引，返回一个样本或者标签。
  进阶复写（包含文件等操作）[系统学习Pytorch笔记三：Pytorch数据读取机制(DataLoader)与图像预处理模块(transforms)_dataloader 输入两个变量-CSDN博客](https://blog.csdn.net/wuzhongqiang/article/details/105499476?ops_request_misc=%7B%22request%5Fid%22%3A%22170636591516800225521260%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170636591516800225521260&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~top_positive~default-1-105499476-null-null.nonecase&utm_term=Dataloader&spm=1018.2226.3001.4450)
  
  ****

## 任务二：文本嵌入（Word2vec的思想）（[【白话NLP】什么是词向量 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/81032021)）

1. 概述：对文本语料库进行预处理，将他们的[one-hot向量](https://www.zhihu.com/search?q=one-hot向量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A266068967})作为word2vec的输入，通过word2vec训练低维词向量（word embedding）。目前有两种训练模型（CBOW和Skip-gram），两种加速算法（Negative Sample与Hierarchical Softmax）（https://www.zhihu.com/question/44832436/answer/266068967）

2. 词表构建（此处采用torchtext进行构建）
   PS:经过查阅资料并多次尝试，最后得出结论：需要完全对应的版本才能安装成功，我的pytorch版本是1.12.1，对应的torchtext版本应该是0.13.1（公式为，设pytorch为1.a.b，则torchtext版本应该是0.(a+1).b）,最后使用 pip install torchtext== 0.13.1成功安装。
   
   https://blog.csdn.net/zuoli_/article/details/129270408
   我的pytorch是0.13.0所以pip install torchtext == 0.14.0 -i https://pypi.tuna.tsinghua.edu.cn/packages（使用清华镜像源加速）
