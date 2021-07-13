# 文本分类
## 任务
给定一个句子，判断这个句子是正面（0）还是负面（1）
<br>
- 数据集：
    - training_label.txt 20万 具有标签
    - training_nolabel.txt 120万 没有标签
    - testing_data.txt 20万（10万public，10万private）
<br><br>

## 学习目的
监督学习：只是使用带标签的训练集
半监督：使用2w带标签监督学习，使用self-train利用无标签（因為 semi-supervise learning 在 labeled training data 數量較少時，比較能夠發揮作用）
<br>

word embedding:
BOW+DNN/RNN 
word2vec+LSTM
<br>

准确率/准确率曲线