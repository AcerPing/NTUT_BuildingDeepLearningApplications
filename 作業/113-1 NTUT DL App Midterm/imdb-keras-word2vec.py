# Cell 1: 導入所需的函式庫
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re
from tqdm import tqdm

# Cell 2: 讀取資料
print("讀取數據...")
try:
    train_df = pd.read_csv('IMDB_ntut_train.csv', encoding='utf-8')
    test_df = pd.read_csv('IMDB_ntut_test.csv', encoding='utf-8')
except UnicodeDecodeError:
    train_df = pd.read_csv('IMDB_ntut_train.csv', encoding='latin1')
    test_df = pd.read_csv('IMDB_ntut_test.csv', encoding='latin1')

print("訓練集大小:", len(train_df))
print("測試集大小:", len(test_df))

# Cell 3: 定義預處理函數
def preprocess_text(text):
    # 轉換為小寫
    text = text.lower()
    
    # 移除HTML標籤
    text = re.sub(r'<.*?>', '', text)
    
    # 只保留英文字母和空格
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    return text

# Cell 4: 預處理文本
print("預處理文本...")
train_texts = [preprocess_text(review) for review in tqdm(train_df['review'])]
test_texts = [preprocess_text(review) for review in tqdm(test_df['review'])]

# Cell 5: 設置 tokenizer
print("創建詞彙表...")
max_words = 10000  # 最大詞彙量
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts + test_texts)

# 轉換文本為序列
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Cell 6: 填充序列
maxlen = 200  # 最大序列長度
train_padded = pad_sequences(train_sequences, maxlen=maxlen)
test_padded = pad_sequences(test_sequences, maxlen=maxlen)

# Cell 7: 創建簡單的 Word2Vec 模型
embedding_dim = 100  # 詞向量維度

model = Sequential([
    Embedding(max_words, embedding_dim, input_length=maxlen),
    GlobalAveragePooling1D()
])

# Cell 8: 獲取文本向量表示
print("生成文本向量...")
train_vectors = model.predict(train_padded)
test_vectors = model.predict(test_padded)

# Cell 9: 訓練分類器
print("訓練分類器...")
le = LabelEncoder()
train_labels = le.fit_transform(train_df['sentiment'])

classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_vectors, train_labels)

# Cell 10: 進行預測與輸出結果
print("進行預測...")
predictions = classifier.predict(test_vectors)
predictions_labels = le.inverse_transform(predictions)

# 創建提交文件
submit_df = pd.DataFrame({
    'Id': range(len(predictions_labels)),
    'Prediction': predictions_labels
})

# 保存預測結果
submit_df.to_csv('IMDB_ntut_submit.csv', index=False)
print("預測結果已保存到 'IMDB_ntut_submit.csv'")

# Cell 11: 顯示結果統計
print("\n預測結果統計：")
print(submit_df['Prediction'].value_counts())
print("\n前5個預測結果：")
print(submit_df.head())

# Cell 12 (可選): 詞彙表分析
print("\n詞彙表信息:")
print(f"詞彙表大小: {len(tokenizer.word_index)}")
print(f"向量維度: {embedding_dim}")

# Cell 13 (可選): 顯示一些詞的索引
print("\n一些常見詞的索引:")
sample_words = ['movie', 'good', 'bad', 'excellent']
for word in sample_words:
    if word in tokenizer.word_index:
        print(f"{word}: {tokenizer.word_index[word]}")
    else:
        print(f"{word} 不在詞彙表中")
