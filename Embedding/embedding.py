# Generated from: embedding.ipynb
# Converted at: 2025-12-17T20:28:21.933Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install -q torch transformers sentencepiece biopython pandas tqdm 

import os

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import numpy as np
import pandas as pd
import os
import gc
from tqdm.auto import tqdm
import sys

# 1. CẤU HÌNH (CONFIGURATION)
class CFG:
    DATA_DIR = "/kaggle/input/cafa-6-protein-function-prediction" 
    TRAIN_FASTA = os.path.join(DATA_DIR, "Train/train_sequences.fasta")
    TEST_FASTA = os.path.join(DATA_DIR, "Test/testsuperset.fasta")
    
    OUTPUT_DIR = "/kaggle/working"
    
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    
    # Tham số chạy
    MAX_LEN = 1024      # Cắt các chuỗi dài hơn 1024 aa
    BATCH_SIZE = 32     # Tăng lên 64 nếu GPU khỏe, giảm xuống 16 nếu OOM
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {CFG.DEVICE}")
print(f"Model: {CFG.MODEL_NAME}")

# 2. HÀM XỬ LÝ (CORE FUNCTIONS)
def load_model():
    print("Downloading & Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
    model = AutoModel.from_pretrained(CFG.MODEL_NAME)
    
    # Chuyển sang Half Precision (FP16) để giảm 50% VRAM và tăng tốc
    if CFG.DEVICE.type == 'cuda':
        model = model.half()
    
    model.to(CFG.DEVICE)
    model.eval()
    return tokenizer, model

def extract_embeddings(fasta_path, prefix, tokenizer, model):
    if not os.path.exists(fasta_path):
        print(f"Skipping {prefix}: File not found at {fasta_path}")
        return

    print(f"\n--- Processing {prefix} Data ---")
    
    # 1. Đọc dữ liệu từ FASTA
    ids = []
    seqs = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        ids.append(str(record.id))
        # Cắt sơ bộ sequence để tránh tốn RAM khi tạo list (model sẽ cắt chính xác sau)
        seqs.append(str(record.seq)[:CFG.MAX_LEN + 50])
        
    n_samples = len(ids)
    print(f"Loaded {n_samples} sequences.")

    # 2. Sắp xếp theo độ dài (Smart Batching)
    # Gom các chuỗi có độ dài tương đương vào 1 batch để giảm padding -> Chạy nhanh hơn 30%
    sorted_indices = np.argsort([len(s) for s in seqs])[::-1] # Dài nhất xếp trước
    
    # 3. Khởi tạo mảng kết quả
    # Output dim của ESM-2 650M là 1280
    embed_dim = 1280 
    embeddings_matrix = np.zeros((n_samples, embed_dim), dtype=np.float16)
    
    # 4. Vòng lặp Extract
    with torch.inference_mode(): # Tắt gradient hoàn toàn
        for i in tqdm(range(0, n_samples, CFG.BATCH_SIZE), desc=f"Extracting {prefix}"):
            # Lấy batch index đã sort
            batch_idx = sorted_indices[i : i + CFG.BATCH_SIZE]
            batch_seqs = [seqs[idx] for idx in batch_idx]
            
            # Tokenize
            inputs = tokenizer(
                batch_seqs, 
                padding=True, 
                truncation=True, 
                max_length=CFG.MAX_LEN, 
                return_tensors="pt"
            )
            
            # Move to GPU
            input_ids = inputs['input_ids'].to(CFG.DEVICE)
            attention_mask = inputs['attention_mask'].to(CFG.DEVICE)
            
            # Forward Pass
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state # (Batch, Seq_Len, 1280)
            
            # Mean Pooling (Chỉ lấy trung bình các token thật, bỏ padding)
            # Chú ý: convert mask sang float để nhân
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            
            # Kết quả embedding cho batch này
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy().astype(np.float16)
            
            # Gán vào matrix tổng theo đúng vị trí index gốc
            embeddings_matrix[batch_idx] = batch_embeddings
            
            # Dọn dẹp VRAM định kỳ
            if i % 100 == 0:
                del input_ids, attention_mask, output, last_hidden_state
                torch.cuda.empty_cache()

    # 5. Lưu File
    out_emb_path = os.path.join(CFG.OUTPUT_DIR, f"{prefix}_embeddings_esm2.npy")
    out_ids_path = os.path.join(CFG.OUTPUT_DIR, f"{prefix}_ids_esm2.npy")
    
    # Convert sang float32 khi save
    np.save(out_emb_path, embeddings_matrix.astype(np.float32))
    np.save(out_ids_path, np.array(ids))
    
    print(f"Saved Embeddings: {out_emb_path} {embeddings_matrix.shape}")
    print(f"Saved IDs: {out_ids_path}")
    
    # Dọn RAM
    del embeddings_matrix, ids, seqs, sorted_indices
    gc.collect()

# 3. THỰC THI (MAIN)
if __name__ == "__main__":
    # Setup
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    
    # Load Model (Load 1 lần dùng cho cả 2 file)
    tokenizer, model = load_model()
    
    # Chạy Train
    extract_embeddings(CFG.TRAIN_FASTA, "train", tokenizer, model)
    
    # Chạy Test (Nếu có)
    extract_embeddings(CFG.TEST_FASTA, "test", tokenizer, model)
    
    print("\nALL DONE! Embeddings are ready in Output section.")