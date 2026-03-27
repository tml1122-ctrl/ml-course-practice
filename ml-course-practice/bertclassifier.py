import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time

# --- 1. 定義模型結構 ---
class BertClassifier(nn.Module):
    def __init__(self, bert_model, classifier):
        super(BertClassifier, self).__init__()
        self.bert_model = bert_model
        self.classifier = classifier

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取得 BERT 的 [CLS] token 輸出
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# --- 2. 初始化與環境準備 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_base = BertModel.from_pretrained(model_name)
classifier_layer = nn.Linear(bert_base.config.hidden_size, 2)
model = BertClassifier(bert_base, classifier_layer).to(device)

# --- 3. 自動載入 IMDB 資料集 (不需手動上傳 CSV) ---
print("正在從雲端載入 IMDB 資料集...")
raw_datasets = load_dataset('imdb')
train_df = pd.DataFrame(raw_datasets['train'])
test_df = pd.DataFrame(raw_datasets['test'])

# --- 4. 逐步增加資料量 n 的循環測試 ---
batch_size = 16
# n 從 2, 4, 8, 16... 增加
n_list = [2**i for i in range(1, 16)]

for n in n_list:
    if n > len(train_df): break

    print(f"\n🚀 測試資料量 n = {n} (Batch Size = {batch_size})")

    # 抽取 n 筆資料
    sample_df = train_df.sample(n, random_state=42)
    texts = sample_df['text'].tolist()
    labels = sample_df['label'].tolist()

    # 編碼處理 (設定 max_length=128 以節省記憶體，若要更高精準度可改 512)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 優化器與損失函數
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 訓練模型
    model.train()
    start_time = time.time()
    try:
        epoch_loss = 0
        for batch in loader:
            input_ids, attention_mask, b_labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, b_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        duration = time.time() - start_time
        print(f"✅ 訓練完成 | 耗時: {duration:.2f}s | 平均 Loss: {epoch_loss/len(loader):.4f}")

        # --- 呈現測試結果 (隨機抽 100 筆測試集進行驗證) ---
        model.eval()
        test_sample = test_df.sample(100, random_state=42)
        test_enc = tokenizer(test_sample['text'].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

        with torch.no_grad():
            t_input = test_enc['input_ids'].to(device)
            t_mask = test_enc['attention_mask'].to(device)
            t_logits = model(t_input, t_mask)
            preds = torch.argmax(t_logits, dim=1).cpu()
            acc = (preds == torch.tensor(test_sample['label'].values)).float().mean()
            print(f"📊 測試結果 | 100 筆測試樣本準確率: {acc:.2%}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ 記憶體不足 (OOM)！Colab 無法負荷資料量 n={n}")
            break
        else:
            print(f"系統錯誤: {e}")
            break