# Cell 1: Import & Config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gc

# Cấu hình thiết bị
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Cấu hình đường dẫn
INPUT_EMBED_DIR = "/kaggle/input/cafa6-t5-embeddings"

CONFIG = {
    "num_labels": 1500,
    "batch_size": 128,
    "lr": 0.001,
    "epochs": 20, # Tăng epoch lên vì chạy rất nhanh
    "paths": {
        # File Embedding có sẵn
        "train_embeds": f"{INPUT_EMBED_DIR}/train_embeddings_esm2.npy",
        "train_ids":    f"{INPUT_EMBED_DIR}/train_ids_esm2.npy",
        "test_embeds":  f"{INPUT_EMBED_DIR}/test_embeddings_esm2.npy",
        "test_ids":     f"{INPUT_EMBED_DIR}/test_ids_esm2.npy",
        
        # File Labels gốc của cuộc thi
        "train_terms": "/kaggle/input/cafa-6-protein-function-prediction/Train/train_terms.tsv"
    }
}

# Cell 2: Load Embeddings
print("Loading embeddings from disk...")

# Load Train
train_embeds = np.load(CONFIG['paths']['train_embeds'])
train_ids = np.load(CONFIG['paths']['train_ids'], allow_pickle=True)

# Load Test
test_embeds = np.load(CONFIG['paths']['test_embeds'])
test_ids = np.load(CONFIG['paths']['test_ids'], allow_pickle=True)

# Tự động cập nhật kích thước đầu vào (Input Dimension)
EMBED_DIM = train_embeds.shape[1]

print(f"--- Data Loaded ---")
print(f"Train shape: {train_embeds.shape}")
print(f"Test shape:  {test_embeds.shape}")
print(f"Embedding Dimension detected: {EMBED_DIM}")

# Cell 3: Prepare Targets
import gc

print("--- Processing Targets & Fixing ID Format ---")

# 1. Load dữ liệu
train_terms = pd.read_csv(CONFIG['paths']['train_terms'], sep="\t")
train_ids = np.load(CONFIG['paths']['train_ids'], allow_pickle=True)

# Sửa lỗi Logic
def clean_id(pid):
    # Chuyển bytes sang string nếu cần
    if isinstance(pid, bytes):
        pid = pid.decode('utf-8')
    pid_str = str(pid).strip()
    
    # Logic tách chuỗi: "sp|A0A0C5B5G6|MOTSC_HUMAN" -> lấy "A0A0C5B5G6"
    parts = pid_str.split('|')
    if len(parts) > 1:
        return parts[1] # Lấy mã ở giữa
    return pid_str # Trả về nguyên gốc nếu không tìm thấy dấu |

# Áp dụng hàm sửa lỗi
train_ids_clean = [clean_id(pid) for pid in train_ids]

# Tạo map
id_map = {pid: i for i, pid in enumerate(train_ids_clean)}

# 2. Chọn Top Labels
top_terms = train_terms['term'].value_counts().index[:CONFIG['num_labels']]
term_to_idx = {term: i for i, term in enumerate(top_terms)}

# 3. Tạo ma trận Targets
num_samples = len(train_ids)
labels_matrix = np.zeros((num_samples, CONFIG['num_labels']), dtype=np.float32)

# Lọc dữ liệu
train_terms['EntryID'] = train_terms['EntryID'].astype(str).str.strip()
filtered_terms = train_terms[
    (train_terms['EntryID'].isin(id_map)) & 
    (train_terms['term'].isin(top_terms))
]

print(f"Số lượng dòng khớp: {len(filtered_terms)}")

if len(filtered_terms) == 0:
    print("Lỗi, vui lòng kiểm tra lại logic cắt chuỗi.")
else:
    # Điền số 1 vào ma trận
    print("Đang tạo ma trận nhãn...")
    for pid, term in tqdm(zip(filtered_terms['EntryID'], filtered_terms['term']), total=len(filtered_terms)):
        row_idx = id_map[pid]
        col_idx = term_to_idx[term]
        labels_matrix[row_idx, col_idx] = 1.0
        
    print(f"Tổng số nhãn dương: {labels_matrix.sum()}")

# Dọn dẹp
del train_terms, filtered_terms, train_ids_clean
gc.collect()

# Cell 4: Hybrid CNN-ResNet Model
class ProteinDataset(Dataset):
    def __init__(self, embeddings, targets=None, ids=None):
        self.embeddings = embeddings
        self.targets = targets
        self.ids = ids
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embed = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return embed, target
        return embed, self.ids[idx]

# Block phụ trợ
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c = x.size()
        y = x.view(b, c, 1)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y)
        return x * y

class AdvancedResBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.25):
        super(AdvancedResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
        self.se = SEBlock(out_features)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    def forward(self, x):
        return self.se(self.block(x)) + self.shortcut(x)

# Main Hybrid Model
class HybridSystem(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HybridSystem, self).__init__()
        
        # Nhánh 1: 1D-CNN (Tìm kiếm mẫu cục bộ)
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5), # Quét cửa sổ lớn
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.MaxPool1d(2),
            
            nn.Flatten(),
            nn.Linear(64 * (input_dim // 4), 512), # Thu gọn về 512 đặc trưng
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.3)
        )
        
        # Nhánh 2: Deep SE-ResNet
        self.resnet_branch = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(0.2),
            
            AdvancedResBlock(1024, 1024),
            AdvancedResBlock(1024, 512),
        ) # Output: 512 đặc trưng
        
        # Kết hợp (Concatenate)
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 1024), # 512 từ CNN + 512 từ ResNet
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(0.4)
        )
        
        # Multi-Sample Dropout (Kỹ thuật tăng điểm Kaggle)
        # Tạo 5 dropout khác nhau để dự đoán 5 lần và lấy trung bình
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 1. Chạy nhánh CNN (cần reshape thành 3D tensor: batch, channels, length)
        x_cnn = x.unsqueeze(1) # (B, 1280) -> (B, 1, 1280)
        out_cnn = self.cnn_branch(x_cnn)
        
        # 2. Chạy nhánh ResNet
        out_res = self.resnet_branch(x)
        
        # 3. Gộp lại
        combined = torch.cat([out_cnn, out_res], dim=1)
        features = self.fusion(combined)
        
        # 4. Multi-Sample Dropout Output
        # Tính toán output qua 5 lớp dropout khác nhau rồi cộng lại
        output = torch.zeros(features.size(0), self.fc.out_features).to(features.device)
        for dropout in self.dropouts:
            output += self.fc(dropout(features))
        
        return output / len(self.dropouts) # Lấy trung bình

# Cell 5: Training Hybrid Model (5-Fold)
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import OneCycleLR

# Hybrid Model Configuration
N_FOLDS = 5
EPOCHS_PER_FOLD = 18 
LR = 0.0015
BATCH_SIZE = 200 

# Focal Loss 
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        if self.logits: bce_loss = self.bce(inputs, targets)
        else: bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.reduce: return torch.mean(F_loss)
        else: return F_loss

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_embeds, labels_matrix)):
    print(f"\n>>> FOLD {fold+1}/{N_FOLDS}")
    
    X_train, X_val = train_embeds[train_idx], train_embeds[val_idx]
    y_train, y_val = labels_matrix[train_idx], labels_matrix[val_idx]
    
    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # Init Hybrid Model
    model = HybridSystem(input_dim=EMBED_DIM, num_classes=CONFIG['num_labels']).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = FocalLoss(gamma=2.0)
    
    # OneCycleLR
    scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS_PER_FOLD, pct_start=0.3)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        train_loss = 0
        for embeds, targets in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            embeds, targets = embeds.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(embeds)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step() 
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for embeds, targets in val_loader:
                embeds, targets = embeds.to(DEVICE), targets.to(DEVICE)
                outputs = model(embeds)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"model_fold_{fold}.pth")
            
    print(f"Fold {fold+1} Best Loss: {best_loss:.5f}")
    fold_metrics.append(best_loss)

print(f"\n--- TRAINING FINISHED --- Avg Loss: {np.mean(fold_metrics):.5f}")

# Cell 6: Ensemble Prediction with Hybrid Models
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

print("\n--- Generating Predictions (Hybrid System) ---")

# 1. Khôi phục mapping
print("Re-creating index mapping...")
df_terms = pd.read_csv(CONFIG['paths']['train_terms'], sep="\t")
top_terms = df_terms['term'].value_counts().index[:CONFIG['num_labels']]
idx_to_term = {i: term for i, term in enumerate(top_terms)}
del df_terms, top_terms
gc.collect()

# 2. Setup Data Loader
BATCH_SIZE = 256
test_dataset_final = ProteinDataset(test_embeds, ids=test_ids)
test_loader_final = DataLoader(test_dataset_final, batch_size=BATCH_SIZE, shuffle=False)

# 3. Load Models
models = []
for fold in range(N_FOLDS):
    # Dùng đúng class HybridSystem
    model = HybridSystem(input_dim=EMBED_DIM, num_classes=CONFIG['num_labels']).to(DEVICE)
    model.load_state_dict(torch.load(f"model_fold_{fold}.pth"))
    model.eval()
    models.append(model)
print(f"Loaded {len(models)} Hybrid models.")

# 4. Prediction
THRESHOLD = 0.015
TOP_K = 75       
output_file = "submission.tsv"

print(f"Inference... Threshold={THRESHOLD}, Top-K={TOP_K}")

with open(output_file, 'w') as f:
    with torch.no_grad():
        for step, (embeds, batch_ids) in enumerate(tqdm(test_loader_final)):
            embeds = embeds.to(DEVICE)
            
            avg_probs = None
            for model in models:
                logits = model(embeds)
                probs = torch.sigmoid(logits)
                if avg_probs is None: avg_probs = probs
                else: avg_probs += probs
            
            avg_probs /= len(models)
            avg_probs = avg_probs.cpu().numpy()
            
            batch_lines = []
            for i, pid in enumerate(batch_ids):
                row_probs = avg_probs[i]
                idx_candidates = np.where(row_probs > THRESHOLD)[0]
                
                if len(idx_candidates) > TOP_K:
                    candidate_probs = row_probs[idx_candidates]
                    final_indices = idx_candidates[np.argsort(candidate_probs)[-TOP_K:]]
                else:
                    final_indices = idx_candidates
                
                for idx in final_indices:
                    batch_lines.append(f"{pid}\t{idx_to_term[idx]}\t{row_probs[idx]:.3f}\n")
            
            f.writelines(batch_lines)
            del batch_lines, avg_probs, embeds
            if step % 50 == 0: gc.collect()

print(f"Submission saved to {output_file}")