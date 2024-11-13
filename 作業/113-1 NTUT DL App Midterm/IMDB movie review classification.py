# 首先安裝需要的套件
!pip install torch transformers tqdm pandas numpy scikit-learn

# Cell 1: 導入必要的函式庫
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Cell 2: 設置設備
# 檢查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

# Cell 3: 數據載入
# 載入數據
print("讀取數據...")
try:
    train_df = pd.read_csv('IMDB_ntut_train.csv', encoding='utf-8')
    test_df = pd.read_csv('IMDB_ntut_test.csv', encoding='utf-8')
except UnicodeDecodeError:
    train_df = pd.read_csv('IMDB_ntut_train.csv', encoding='latin1')
    test_df = pd.read_csv('IMDB_ntut_test.csv', encoding='latin1')

# 顯示數據基本信息
print("\n訓練集基本信息：")
print(train_df.info())
print("\n測試集基本信息：")
print(test_df.info())

# Cell 4: BERT 模型和 tokenizer 載入
# 載入 BERT tokenizer 和模型
print("載入 BERT 模型和 tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)
model.eval()

# Cell 5: 定義數據集類別
class IMDBDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

# Cell 6: 定義 BERT 嵌入函數
def get_bert_embeddings(texts, batch_size=8):
    dataset = IMDBDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="處理批次"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

# Cell 7: 獲取 BERT 嵌入
print("處理訓練集...")
train_embeddings = get_bert_embeddings(train_df['review'])
print("處理測試集...")
test_embeddings = get_bert_embeddings(test_df['review'])

# 保存嵌入向量（可選）
print("保存嵌入向量...")
np.save('train_bert_embeddings.npy', train_embeddings)
np.save('test_bert_embeddings.npy', test_embeddings)

# Cell 8: 相似度計算函數（可選的分析工具）
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 計算示例相似度
print("\n示範評論相似度計算：")
n_examples = min(3, len(train_embeddings))
for i in range(n_examples):
    for j in range(i+1, n_examples):
        similarity = cosine_similarity(train_embeddings[i], train_embeddings[j])
        print(f"評論 {i+1} 和評論 {j+1} 的相似度: {similarity:.4f}")

# Cell 9: 訓練分類器和進行預測
# 標籤編碼
le = LabelEncoder()
train_labels = le.fit_transform(train_df['sentiment'])

# 訓練分類器
print("訓練分類器...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_embeddings, train_labels)

# 進行預測
print("進行預測...")
predictions = classifier.predict(test_embeddings)
predictions_labels = le.inverse_transform(predictions)

# Cell 10: 創建和保存提交文件
# 創建提交文件
print("創建提交文件...")
submit_df = pd.DataFrame({
    'Id': range(len(predictions_labels)),
    'Prediction': predictions_labels
})

# 保存預測結果
submit_df.to_csv('IMDB_ntut_submit.csv', index=False)
print("預測結果已保存到 'IMDB_ntut_submit.csv'")

# 顯示預測結果統計
print("\n預測結果統計：")
print(submit_df['Prediction'].value_counts())

# 顯示前幾個預測結果
print("\n前5個預測結果：")
print(submit_df.head())

# Cell 11: 額外的分析（可選）
print("\n嵌入向量的形狀:")
print(f"訓練集: {train_embeddings.shape}")
print(f"測試集: {test_embeddings.shape}")

# 顯示分類器的性能指標
if 'sentiment' in train_df.columns:
    from sklearn.metrics import classification_report
    train_predictions = classifier.predict(train_embeddings)
    print("\n訓練集上的分類報告：")
    print(classification_report(train_labels, train_predictions))