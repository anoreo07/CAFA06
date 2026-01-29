import os
import gc
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

class Config:
    BEST_SUB = '/kaggle/input/0-243-output/submission.tsv'
    
    SECOND_SUB = '/kaggle/input/NgoLinhbaochayfilenay/submission.tsv' 
    
    W_BEST = 0.6
    W_SECOND = 0.4
    
    GOA_PATH = '/kaggle/input/protein-go-annotations/goa_uniprot_all.csv'
    
    OBO_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Train/go-basic.obo'
    
    OUTPUT_FILE = 'submission.tsv'

cfg = Config()

def parse_obo_children(obo_path):
    print(f"Parsing OBO from {obo_path}...")
    children_map = defaultdict(set)
    if not os.path.exists(obo_path):
        print("WARNING: OBO file not found!")
        return children_map

    with open(obo_path, "r") as f:
        cur_id = None
        for line in f:
            line = line.strip()
            if line == "[Term]":
                cur_id = None
            elif line.startswith("id: "):
                cur_id = line.split("id: ")[1].strip()
            elif line.startswith("is_a: ") and cur_id:
                # Line: is_a: GO:XXXXX ! name
                parent = line.split()[1].strip()
                children_map[parent].add(cur_id)
            elif line.startswith("relationship: part_of ") and cur_id:
                parts = line.split()
                if len(parts) >= 3:
                    parent = parts[2].strip()
                    children_map[parent].add(cur_id)
    
    print(f"Parsed {len(children_map)} parents with children.")
    return children_map

def get_descendants(go_id, children_map):
    desc = set()
    stack = [go_id]
    visited = set()
    while stack:
        cur = stack.pop()
        if cur in visited: continue
        visited.add(cur)
        
        if cur in children_map:
            for child in children_map[cur]:
                if child not in desc:
                    desc.add(child)
                    stack.append(child)
    return desc

def process_goa_database(goa_path, children_map):
    print(f"\n>>> Loading GOA Database from {goa_path}...")
    
    if not os.path.exists(goa_path):
        raise FileNotFoundError(f"Vui lòng Add Data 'protein-go-annotations' vào Kaggle!")
    try:
        df = pd.read_csv(goa_path, usecols=['protein_id', 'go_term', 'qualifier'])
    except ValueError:
        df = pd.read_csv(goa_path, header=None, names=['protein_id', 'go_term', 'qualifier'])
    
    df = df.drop_duplicates()
    print(f"Loaded {len(df)} annotations.")

    print("Extracting NEGATIVE annotations...")
    neg_df = df[df['qualifier'].str.contains('NOT', na=False)].copy()
    
    neg_map = neg_df.groupby('protein_id')['go_term'].apply(list).to_dict()
    final_neg_keys = set()
    
    print("Propagating Negatives (This takes time)...")
    for pid, terms in tqdm(neg_map.items()):
        all_neg_terms = set(terms)
        for t in terms:
            all_neg_terms |= get_descendants(t, children_map)
        
        for t in all_neg_terms:
            final_neg_keys.add(f"{pid}_{t}")
            
    print(f"Identified {len(final_neg_keys)} negative Protein-Term pairs to remove.")
    del neg_df, neg_map
    gc.collect()

    print("Extracting POSITIVE annotations (Ground Truth)...")
    pos_df = df[~df['qualifier'].str.contains('NOT', na=False)].copy()
    
    pos_keys = set(pos_df['protein_id'].astype(str) + "_" + pos_df['go_term'].astype(str))
    
    print(f"Identified {len(pos_keys)} positive pairs to inject (Score = 1.0).")
    del df, pos_df
    gc.collect()
    
    return pos_keys, final_neg_keys

def load_submission_optimized(path):
    print(f"Loading submission: {path}")
    if not os.path.exists(path):
        print(f"Warning: File {path} not found. Skipping.")
        return pd.DataFrame()
        
    df = pd.read_csv(path, sep='\t', header=None, names=['protein_id', 'go_term', 'score'])
    
    df['key'] = df['protein_id'].astype(str) + "_" + df['go_term'].astype(str)
    return df

def main():
    # 1. Parse OBO
    children_map = parse_obo_children(cfg.OBO_PATH)
    
    # 2. Get Info from GOA Database
    pos_keys, neg_keys = process_goa_database(cfg.GOA_PATH, children_map)
    
    # 3. Load User Submissions
    print("\n>>> Loading User Submissions...")
    df_best = load_submission_optimized(cfg.BEST_SUB)
    
    # Nếu có file thứ 2 thì merge, không thì dùng file best thôi
    if cfg.SECOND_SUB and os.path.exists(cfg.SECOND_SUB):
        df_sec = load_submission_optimized(cfg.SECOND_SUB)
        
        print("Ensembling 2 files...")
        # Merge on key
        merged = pd.merge(df_best, df_sec[['key', 'score']], on='key', how='outer', suffixes=('_best', '_sec'))
        
        # Fill NaN
        merged['score_best'] = merged['score_best'].fillna(0)
        merged['score_sec'] = merged['score_sec'].fillna(0)
        
        # Weighted Average
        merged['score'] = (merged['score_best'] * cfg.W_BEST + merged['score_sec'] * cfg.W_SECOND)
        
        # Cleanup
        df_final = merged[['protein_id', 'go_term', 'score', 'key']]
        del df_sec, merged
    else:
        print("Using single best file.")
        df_final = df_best
    
    del df_best
    gc.collect()
    
    print("\n>>> Applying Logic...")
    original_count = len(df_final)
    
    # Lọc bỏ các dòng bị cấm
    print("1. Removing Negatives...")
    df_final = df_final[~df_final['key'].isin(neg_keys)]
    print(f"   Removed {original_count - len(df_final)} negative predictions.")
    
    # thêm/Sửa điểm thành 1.0
    print("2. Injecting Ground Truth...")
    
    # Những cái đã có -> Update điểm lên 1.0
    mask_in_sub = df_final['key'].isin(pos_keys)
    df_final.loc[mask_in_sub, 'score'] = 1.0
    print(f"   Boosted {sum(mask_in_sub)} existing predictions to 1.0")
    
    # Lấy danh sách pos_keys chưa có trong df_final
    existing_keys = set(df_final['key'])
    missing_pos_keys = pos_keys - existing_keys
    
    # Chỉ thêm nếu Protein ID nằm trong tập Test (Cái filter này tránh thêm rác)
    # Lấy danh sách Protein trong file submission làm chuẩn Test Set
    valid_test_proteins = set(df_final['protein_id'])
    
    new_rows = []
    for key in missing_pos_keys:
        pid, term = key.split('_', 1) 
        if pid in valid_test_proteins:
            new_rows.append({'protein_id': pid, 'go_term': term, 'score': 1.0})
            
    if new_rows:
        print(f"   Adding {len(new_rows)} completely new ground-truth rows...")
        df_new = pd.DataFrame(new_rows)
        df_final = pd.concat([df_final, df_new], ignore_index=True)
    
    print(f"\n>>> Saving to {cfg.OUTPUT_FILE}...")
    # bỏ score quá thấp để file nhẹ
    df_final = df_final[df_final['score'] > 0.001]
    
    # Sort cho đẹp hihihi
    df_final = df_final.sort_values(['protein_id', 'score'], ascending=[True, False])
    
    # Save
    df_final[['protein_id', 'go_term', 'score']].to_csv(cfg.OUTPUT_FILE, sep='\t', index=False, header=False)
    
    print("DONE! ")

if __name__ == "__main__":
    main()