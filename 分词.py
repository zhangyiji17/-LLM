import jieba
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

text = """你叫张某，是一位智能编程助手。这是张某测试文本。"""
seg_list = list(jieba.cut(text, cut_all=False))
# print(type(seg_list))
for seg in seg_list:
    print(seg)
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
print(text_for_tokenizer)
print(seq)
print(tokenizer.word_index['张'])

# 查看word_index
print(tokenizer.word_index)

# 序列化tokenizer
ss = tokenizer.to_json()
print(ss)

