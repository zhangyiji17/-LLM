import jieba
from transformers import BertTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json



texts = ['张某是一位智能编程助手', '这是测试文本']
seg_list = [list(jieba.cut(text, cut_all=False)) for text in texts]
print(seg_list)

# print("Default Mode: " + "/ ".join(seg_list))

# 加载自定义词典
# jieba.load_userdict("dict.txt")

# 将分词结果转换为空格分隔的字符串，便于Tokenizer处理
text_for_tokenizer = [" ".join(tokens) for tokens in seg_list]

# 初始化Tokenizer并进行拟合
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# tokenizer = BertTokenizer.from_pretrained('D:/RAG与LLM原理及实践/model/google-bert/bert-base-chinese/')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_for_tokenizer)

# 转文本为序列
seq = tokenizer.texts_to_sequences(text_for_tokenizer)
# print(text_for_tokenizer)
# print(seq)

# 查看分词结果
word_index = tokenizer.word_index
for word, index in word_index.items():
    print(f'{word}: {index}')

# 查看word_index
# print(tokenizer.word_index)

# 序列化tokenizer
ss = tokenizer.to_json()
# print(ss)
with open('tokenizer_word_index.json', 'w', encoding='utf-8') as f:
    f.write(ss)  # 将分词器配置保存到文件中

# 反序列化tokenizer
with open('tokenizer_word_index.json', 'r', encoding='utf-8') as f:
    json_string = f.read()
    # 从JSON字符串中加载分词器配置
    # tokenizer = Tokenizer.from_json(json_string)
    # 使用加载的分词器进行分词
    word_index = tokenizer.word_index
    for word, index in word_index.items():
        print(f'{word}: {index}')


ids = tokenizer.texts_to_sequences(seg_list)
print(ids)
p_id = pad_sequences(ids, padding='post')
print(p_id)

# to_json()方法将Tokenizer对象序列化为JSON格式的字符串，可以将其保存到文件中，以便在需要时重新加载Tokenizer对象。
def to_json(self):
    """
    :argument 关键字参数
    :return 分词器配置的JSON字符串。
    若要从JSON字符串中加载标记符，请使用keras.preprocessing.text.tokenizer_from_json(json_string)
    """
    config = self.get_config()  # 获取分词器的配置
    tokenizer_config = {
        'class_name': self.__class__.__name__,  # 分词器的类名
        'config': config  # 分词器的配置
    }
    return json.dumps(tokenizer_config)  # 将分词器的配置转换为JSON字符串


# get_config()方法返回分词器的配置，包括词汇表的大小、最大序列长度、是否忽略未知单词等参数。
def get_config(self):
    """
    :return 分词器的配置字典。
    """
    json_word_counts = json.dumps(self.word_counts)  # 将词汇表转换为JSON字符串
    json_word_docs = json.dumps(self.word_docs)  # 将文档频率表转换为JSON字符串
    json_index_docs = json.dumps(self.index_docs)  # 将索引文档频率表转换为JSON字符串
    json_word_index = json.dumps(self.word_index)  # 将词汇表索引转换为JSON字符串
    json_index_word = json.dumps(self.index_word)  # 将索引词汇表转换为JSON字符串

    return {
        'num_words': self.num_words,  # 词汇表的大小
        'filters': self.filters,  # 需要过滤的字符
        'lower': self.lower,  # 是否将文本转换为小写
        'split': self.split,  # 用于分割文本的分隔符
        'char_level': self.char_level,  # 是否对字符进行分词
        'oov_token': self.oov_token,  # 未知单词的标记
        'word_counts': json_word_counts,  # 词汇表
        'word_docs': json_word_docs,  # 文档频率表
        'index_docs': json_index_docs,  # 索引文档频率表
        'word_index': json_word_index,  # 词汇表索引
        'index_word': json_index_word  # 索引词汇表
    }


# 定义从JSON字符串加载标记符的函数
def tokenizer_from_json(json_string):
    """
    :argument 关键字参数
    :return 分词器对象。
    """
    tokenizer = Tokenizer()  # 创建一个空的分词器对象
    tokenizer_config = json.loads(json_string)  # 将JSON字符串转换为字典
    tokenizer.__setstate__(tokenizer_config)  # 使用字典设置分词器的状态
    return tokenizer  # 返回分词器对象



