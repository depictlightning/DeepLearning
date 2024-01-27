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
          *    扩展（本任务中无需使用）：一个在语料库中不存在或已被删除的标记被映射为一个特殊的未知（“< unk>”）标记。我们还可以选择添加另外三个特殊标记。"< pad>“是一个用于填充的标记，”< bos>“表示一个句子的开始，”< eos>"表示一个句子的结束。
               

4. dataloader学习([Dataset,Dataloader详解_dataset dataloader-CSDN博客](https://blog.csdn.net/junsuyiji/article/details/127585300?utm_medium=distribute.pc_feed_404.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-127585300-blog-null.262^v1^pc_404_mixedpudn&depth_1-utm_source=distribute.pc_feed_404.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-127585300-blog-null.262^v1^pc_404_mixedpud)）vscode文件D:\computer programming\LearnPython

   dataloader的意义：
