## 数据集

位置在Financial sentiment analysis data文件夹中



### 情绪分类数据

all-data.csv：**金融短语库数据集（Financial Phrase Bank）**由4846个英语句子组成，这些句子是从金融新闻数据集中选出的，并由16位金融专家标记。这些数据是通过从LexisNexis数据库上的财经新闻文章中随机选择句子构建的。然后，专家们会根据这句话对公司股价的影响来给这些句子贴上标签。所有句子都有三个标签：肯定、否定和中性。Pyry Takala Pekka Korhonen Jyrki Wallenius Pekka Malo, Ankur Sinha. 2014. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, pages 782–796.



4个json文件（Headline_Trainingdata.json，Headlines_Testdata.json，Microblog_Trainingdata.json，Microblogs_Testdata.json）：数据集由**SemEval-2017任务“金融微博和新闻的细粒度情绪分析”（Semeval-2017 task 5）**提供。该数据集由1633个英语句子组成，这些句子来自不同的公开数据集，如雅虎财经。每个实例都由金融专家在-1到1的范围内标记，其中-1表示负面情绪，+1表示正面情绪。这些情绪分数被转换为积极 、消极和中性标签 ，其中分数大于0的是积极情绪，小于0的被分配为消极情绪，等于0的分数则为中性情绪。T. Daudert M. Huerlimann M. Zarrouk S. Handschuh K. Cortis, A. Freitas and B. Davis. 2017. Semeval-2017 task 5: Fine-grained sentiment analysis on financial microblogs and news. Proc. 11th Int. Workshop Semantic Eval. (SemEval-), 2017, pp. 519–535.



| Dataset               | Positive | Neutral | Negative |
| --------------------- | -------- | ------- | -------- |
| Financial Phrase Bank | 1363     | 2879    | 604      |
| SemEval-2017-Task5    | 653      | 529     | 451      |
| Total                 | 2016     | 3408    | 1055     |



### 新闻数据

gdelt.csv，CSV.header.dailyupdates.txt：**全球事件、语言和语调数据库（GDELT）**。数据集为GDELT 1.0版，统计了从2013年4月1日到2021年9月30日（共3101天）来自世界各地的网络新闻。为了预测虚拟货币，所以只过滤处理美国的新闻。在GDELT数据库中，同一日期内的所有新闻网址都压缩成zip文件存储在CSV表中。为了阅读新闻中的文本，我们首先解压每天的zip文件得到网址URL，然后逐一加载URL以从网站获取新闻中的文字。由于时间限制，我们只能对每天的新闻进行采样。



### 虚拟货币数据

coin_Bitcoin.csv，coin_Dogecoin.csv，coin_Ethereum.csv，coin_Litecoin.csv：来自kaggle或ccxt，与项目data没有区别。



## 代码

create_data.py：读取情绪分类数据并拼接在一起。构造训练集和测试集（8：2）为后续机器学习训练做准备，并构建训练集、测试集和验证集（8：1：1）为后续人工智能训练做准备。



ml_models.py：财经头条和财经新闻文本可能包含错误且格式不一致。为了准备数据，实验中执行了初始预处理，包括文本小写、标点符号删除、从文本中删除数字、清除空格、词干提取，最后进行标记化。机器学习模型包括逻辑回归（lm）和支持向量机（svc），配合数据处理中是否进行词干提取，一共四个模型进行预训练，结果如ml_results.txt所示，有词干提取的svc模型f1-score最大。



transformer.py：在预处理transformer模型的数据时，实验使用HugginFace库中的BertTokenizer对文本进行标记，以与预训练的BERT模型一致的方式对文本进行符号化和小写。



sentiment_scoring.py：使用svc和transformer模型进行情绪评分。以天为单位统计每个模型预测的具有积极/中性/消极情绪的文章数量（采样为1：30），过程中需要利用爬虫爬取网站文字内容。



arima.py：判断ARIMA(p,d,q)模型的参数



forecast.py：将情感数据加入到虚拟货币预测模型的训练之中。



## 结果

test_data.csv和train_data.csv为create_data.py中训练集和测试集（8：2）分割的结果。



test_set.csv，train_set.csv和validation_set.csv为create_data.py中训练集、测试集和验证集（8：1：1）分割的结果。



LR_model_withoutStem.pkl，LR_model_withStem.pkl， SVC_model_withoutStem.pkl和SVC_model_withStem.pkl为ml_models.py中训练的四个模型。



ml_results.txt为ml_models.py中四个模型结果对比。



FinBERT-Transformer.h5为transformer.py中训练的模型。



transformer_results.txt为transformer.py中transformer模型训练过程及结果。



sentiment_score_svc.csv和sentiment_score_trans.csv分别为sentiment_scoring.py中两个模型对新闻文章情绪评分的结果。



acf.png，pacf.png，acf_diff.png，pacf_diff.png为绘制的ACF和PACF图。



文件夹results内的excel为forecast.py中模型的训练结果。