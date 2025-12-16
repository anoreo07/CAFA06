    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from tqdm.auto import tqdm
    import os
    import gc
    import obonet
    import networkx as nx
    import random
    from Bio import SeqIO

    def seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Locked Random Seed: {seed}")

    seed_everything(42)

    class Config:
        RAW_DATA_DIR = '/kaggle/input/cafa-6-protein-function-prediction'
        EMBED_DIR = '/kaggle/input/cafa-6-t5-embeddings'
        
        TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'Train')
        TEST_DIR = os.path.join(RAW_DATA_DIR, 'Test')
        
        TRAIN_TERMS = os.path.join(TRAIN_DIR, 'train_terms.tsv')
        TRAIN_TAXONOMY = os.path.join(TRAIN_DIR, 'train_taxonomy.tsv')
        OBO_FILE = os.path.join(TRAIN_DIR, 'go-basic.obo')
        TEST_TAXONOMY = os.path.join(TEST_DIR, 'testsuperset-taxon-list.tsv')
        
        TRAIN_EMBEDS = os.path.join(EMBED_DIR, 'train_embeds.npy')
        TRAIN_IDS = os.path.join(EMBED_DIR, 'train_ids.npy')
        TEST_EMBEDS = os.path.join(EMBED_DIR, 'test_embeds.npy')
        TEST_IDS = os.path.join(EMBED_DIR, 'test_ids.npy')
        
        NUM_CLASSES = 1500
        BATCH_SIZE = 256
        EPOCHS = 12
        LR = 1e-3
        N_FOLDS = 5
        TOP_TAXONS_COUNT = 50
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    print(f"Running on: {cfg.DEVICE}")

    def clean_ids(id_list):
        cleaned = []
        for x in id_list:
            if isinstance(x, bytes):
                x = x.decode('utf-8')
            x = str(x).strip()
            if '|' in x:
                parts = x.split('|')
                if len(parts) >= 2:
                    cleaned.append(parts[1])
                else:
                    cleaned.append(x)
            else:
                cleaned.append(x)
        return np.array(cleaned)

    def load_and_align_data_fixed():
        print("--- 1. Loading Embeddings & Fixing IDs ---")
        X_train = np.load(cfg.TRAIN_EMBEDS)
        train_ids_raw = np.load(cfg.TRAIN_IDS)
        
        X_test = np.load(cfg.TEST_EMBEDS)
        test_ids_raw = np.load(cfg.TEST_IDS)
        
        train_ids = clean_ids(train_ids_raw)
        test_ids = clean_ids(test_ids_raw)
        
        print("--- 2. Processing Taxonomy Features ---")
        train_tax_df = pd.read_csv(cfg.TRAIN_TAXONOMY, sep='\t', dtype=str)
        train_tax_map = dict(zip(train_tax_df.iloc[:, 0], train_tax_df.iloc[:, 1]))
        
        test_tax_df = pd.read_csv(cfg.TEST_TAXONOMY, sep='\t', dtype=str)
        test_tax_map = dict(zip(test_tax_df.iloc[:, 0], test_tax_df.iloc[:, 1]))
        
        top_taxons = train_tax_df.iloc[:, 1].value_counts().head(cfg.TOP_TAXONS_COUNT).index.tolist()
        tax2idx = {t: i for i, t in enumerate(top_taxons)}
        
        def create_tax_features(ids, tax_map):
            feats = np.zeros((len(ids), cfg.TOP_TAXONS_COUNT), dtype=np.float32)
            match_count = 0
            for i, pid in enumerate(ids):
                if pid in tax_map:
                    match_count += 1
                    tid = tax_map[pid]
                    if tid in tax2idx:
                        feats[i, tax2idx[tid]] = 1.0
            print(f"   > Found taxonomy for {match_count}/{len(ids)} proteins")
            return feats

        print("Generating Taxon Vectors...")
        train_tax_feats = create_tax_features(train_ids, train_tax_map)
        test_tax_feats = create_tax_features(test_ids, test_tax_map)
        
        X_train_final = np.concatenate([X_train, train_tax_feats], axis=1)
        X_test_final = np.concatenate([X_test, test_tax_feats], axis=1)
        
        return X_train_final, train_ids, X_test_final, test_ids

    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim)
            )
            self.relu = nn.ReLU()
        def forward(self, x):
            return self.relu(x + self.net(x))

    class AdvancedModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.entry = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.blocks = nn.Sequential(
                ResidualBlock(1024),
                ResidualBlock(1024),
                ResidualBlock(1024)
            )
            self.head = nn.Linear(1024, num_classes)
            
        def forward(self, x):
            x = self.entry(x)
            x = self.blocks(x)
            return self.head(x)

    class ProteinDataset(Dataset):
        def __init__(self, X, Y=None):
            self.X = torch.from_numpy(X).float()
            self.Y = torch.from_numpy(Y).float() if Y is not None else None
        def __len__(self): return len(self.X)
        def __getitem__(self, i): 
            return (self.X[i], self.Y[i]) if self.Y is not None else self.X[i]

    def train_one_fold(fold, model, train_loader, val_loader):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        
        best_loss = float('inf')
        early_stop_count = 0
        patience = 3 
        
        for epoch in range(cfg.EPOCHS):
            model.train()
            t_loss = 0
            for x, y in train_loader:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()
                
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                    v_loss += criterion(model(x), y).item()
            
            avg_v_loss = v_loss / len(val_loader)
            scheduler.step(avg_v_loss)
            
            if avg_v_loss < best_loss:
                best_loss = avg_v_loss
                torch.save(model.state_dict(), f'model_fold_{fold}.pth')
                early_stop_count = 0
            else:
                early_stop_count += 1
                
            if early_stop_count >= patience:
                print(f"  Fold {fold}: Early stopping at epoch {epoch}")
                break
        
        return best_loss

    X_train_full, train_ids_full, X_test_full, test_ids_full = load_and_align_data_fixed()

    print("--- 3. Processing Labels ---")
    df = pd.read_csv(cfg.TRAIN_TERMS, sep='\t', names=['EntryID', 'term', 'aspect'])
    top_terms = df['term'].value_counts().head(cfg.NUM_CLASSES).index.tolist()
    term2idx = {t: i for i, t in enumerate(top_terms)}
    id_to_terms = df[df['term'].isin(top_terms)].groupby('EntryID')['term'].apply(set).to_dict()

    Y_full = np.zeros((len(train_ids_full), cfg.NUM_CLASSES), dtype=np.float32)
    valid_mask = []
    for i, pid in enumerate(train_ids_full):
        if pid in id_to_terms:
            valid_mask.append(True)
            for term in id_to_terms[pid]:
                Y_full[i, term2idx[term]] = 1.0
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    X_train_clean = X_train_full[valid_mask]
    Y_clean = Y_full[valid_mask]

    # Training
    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)
    input_dim = X_train_clean.shape[1]

    print(f"\n--- Starting Ensemble Training ({cfg.N_FOLDS} Folds) ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_clean)):
        print(f">> Fold {fold+1}/{cfg.N_FOLDS}")
        train_ds = ProteinDataset(X_train_clean[train_idx], Y_clean[train_idx])
        val_ds = ProteinDataset(X_train_clean[val_idx], Y_clean[val_idx])
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
        
        model = AdvancedModel(input_dim, cfg.NUM_CLASSES).to(cfg.DEVICE)
        train_one_fold(fold, model, train_loader, val_loader)
        
        del model, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()

    # Inference
    print("\n--- Running Inference on Test Set ---")
    test_loader = DataLoader(ProteinDataset(X_test_full), batch_size=cfg.BATCH_SIZE)
    final_preds = np.zeros((len(test_ids_full), cfg.NUM_CLASSES), dtype=np.float32)

    for fold in range(cfg.N_FOLDS):
        model = AdvancedModel(input_dim, cfg.NUM_CLASSES).to(cfg.DEVICE)
        model.load_state_dict(torch.load(f'model_fold_{fold}.pth'))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for x in tqdm(test_loader, desc=f"Predicting Fold {fold+1}"):
                x = x.to(cfg.DEVICE)
                logits = model(x)
                fold_preds.append(torch.sigmoid(logits).cpu().numpy())
                
        final_preds += np.vstack(fold_preds)

    final_preds /= cfg.N_FOLDS

    # GO Propagation
    print("--- Applying GO Structure Propagation ---")
    go_graph = obonet.read_obo(cfg.OBO_FILE)
    term_to_idx = {t: i for i, t in enumerate(top_terms)}

    parent_map = {}
    valid_set = set(top_terms)
    for term in top_terms:
        if term in go_graph:
            parents = list(go_graph.successors(term))
            for p in parents:
                if p in valid_set:
                    if p not in parent_map: parent_map[p] = []
                    parent_map[p].append(term)

    for _ in range(2):
        for parent, children in parent_map.items():
            if children:
                p_idx = term_to_idx[parent]
                c_indices = [term_to_idx[c] for c in children]
                max_child_score = final_preds[:, c_indices].max(axis=1)
                final_preds[:, p_idx] = np.maximum(final_preds[:, p_idx], max_child_score)

    print("--- Creating Submission File ---")
    submission_rows = []
    threshold = 0.005 

    for i, pid in enumerate(tqdm(test_ids_full)):
        idxs = np.where(final_preds[i] > threshold)[0]
        for idx in idxs:
            term = top_terms[idx]
            score = final_preds[i, idx]
            submission_rows.append(f"{pid}\t{term}\t{score:.3f}")
            
    with open('submission_dl.tsv', 'w') as f:
        f.write('\n'.join(submission_rows))

    class Config:
        RAW_DATA_DIR = '/kaggle/input/cafa-6-protein-function-prediction'
        EMBED_DIR = '/kaggle/input/cafa-6-t5-embeddings'
        
        DL_SUBMISSION = 'submission_dl.tsv' 
        
        TRAIN_SEQ = os.path.join(RAW_DATA_DIR, 'Train', 'train_sequences.fasta')
        TRAIN_TERMS = os.path.join(RAW_DATA_DIR, 'Train', 'train_terms.tsv')
        TEST_SEQ = os.path.join(RAW_DATA_DIR, 'Test', 'testsuperset.fasta')
        
        TRAIN_EMBEDS = os.path.join(EMBED_DIR, 'train_embeds.npy')
        TEST_EMBEDS = os.path.join(EMBED_DIR, 'test_embeds.npy')
        
        W_DL = 0.4
        W_KNN = 0.6
        
        ROUND_DECIMALS = 3  
        TOP_K = 65         
        MIN_SCORE = 0.01  
        
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()

    def run_exact_match():
        print(">>> 1. Running Exact Sequence Matching")
        
        # 1. Load Train Sequences
        print("   Indexing Train Sequences")
        train_seq_map = {}
        for record in SeqIO.parse(cfg.TRAIN_SEQ, "fasta"):
            seq_str = str(record.seq)
            
            if '|' in record.id:
                pid = record.id.split('|')[1]
            else:
                pid = record.id
                
            if seq_str not in train_seq_map:
                train_seq_map[seq_str] = []
            train_seq_map[seq_str].append(pid)
            
        # 2. Load Train Terms
        print("   Loading Train Terms")
        df_terms = pd.read_csv(cfg.TRAIN_TERMS, sep='\t', names=['id', 'term', 'aspect'])
        df_terms['id'] = df_terms['id'].astype(str).str.strip()
        
        top_terms = set(df_terms['term'].value_counts().head(5000).index)
        df_terms = df_terms[df_terms['term'].isin(top_terms)]
        
        id_to_terms = df_terms.groupby('id')['term'].apply(set).to_dict()
        
        # 3. Match Test Sequences
        print("   Matching Test Sequences")
        exact_matches = []
        match_count = 0
        
        for record in tqdm(SeqIO.parse(cfg.TEST_SEQ, "fasta"), desc="Exact Matching"):
            test_seq = str(record.seq)
            
            if '|' in record.id:
                test_id = record.id.split('|')[1]
            else:
                test_id = record.id
            
            if test_seq in train_seq_map:
                train_ids = train_seq_map[test_seq]
                
                collected_terms = set()
                for tid in train_ids:
                    if tid in id_to_terms:
                        collected_terms.update(id_to_terms[tid])
                
                if collected_terms:
                    match_count += 1
                    for term in collected_terms:
                        exact_matches.append({'id': test_id, 'term': term, 'exact_score': 0.99})
                    
        print(f"   >>> FOUND {match_count} EXACT MATCHES!")
        return pd.DataFrame(exact_matches)

    def run_knn():
        print("\n>>> 2. Running KNN")
        X_train = np.load(cfg.TRAIN_EMBEDS)
        X_test = np.load(cfg.TEST_EMBEDS)
        
        X_train = torch.from_numpy(X_train).to(cfg.DEVICE)
        X_train = torch.nn.functional.normalize(X_train, p=2, dim=1)
        
        X_test = torch.from_numpy(X_test)
        X_test = torch.nn.functional.normalize(X_test, p=2, dim=1)
        
        df = pd.read_csv(cfg.TRAIN_TERMS, sep='\t', names=['id', 'term', 'aspect'])
        train_ids_npy = np.load(os.path.join(cfg.EMBED_DIR, 'train_ids.npy'))
        train_ids_npy = [str(x).split('|')[1] if '|' in str(x) else str(x) for x in train_ids_npy]
        
        id_to_terms = df.groupby('id')['term'].apply(list).to_dict()
        idx_to_terms = {}
        for i, pid in enumerate(train_ids_npy):
            if pid in id_to_terms:
                idx_to_terms[i] = id_to_terms[pid]
                
        test_ids_npy = np.load(os.path.join(cfg.EMBED_DIR, 'test_ids.npy'))
        test_ids_npy = [str(x).split('|')[1] if '|' in str(x) else str(x) for x in test_ids_npy]
        
        knn_preds = []
        K = 10
        BATCH_SIZE = 500
        
        num_test = len(test_ids_npy)
        for i in tqdm(range(0, num_test, BATCH_SIZE), desc="KNN Computing"):
            batch_test = X_test[i : i + BATCH_SIZE].to(cfg.DEVICE)
            sim_matrix = torch.mm(batch_test, X_train.t())
            vals, inds = torch.topk(sim_matrix, k=K, dim=1)
            vals = vals.cpu().numpy()
            inds = inds.cpu().numpy()
            
            for j in range(len(batch_test)):
                tid = test_ids_npy[i + j]
                scores = {}
                total_weight = 0
                
                for k in range(K):
                    idx = inds[j, k]
                    sim = vals[j, k]
                    
                    if sim > 0.3: 
                        weight = sim * sim
                        total_weight += weight
                        if idx in idx_to_terms:
                            for term in idx_to_terms[idx]:
                                scores[term] = scores.get(term, 0) + weight
                                
                if total_weight > 0:
                    for term in scores:
                        scores[term] /= total_weight
                        scores[term] *= min(1.0, total_weight) 
                
                for term, score in scores.items():
                    if score > 0.01:
                        knn_preds.append({'id': tid, 'term': term, 'knn_score': score})
                        
        del X_train, X_test
        torch.cuda.empty_cache()
        gc.collect()
        return pd.DataFrame(knn_preds)

    def main():
        # 1. Exact Match
        df_exact = run_exact_match()
        
        # 2. KNN
        df_knn = run_knn()
        
        # 3. Load DL
        print("\n>>> 3. Loading DL submission")
        try:
            df_dl = pd.read_csv(cfg.DL_SUBMISSION, sep='\t', header=None, names=['id', 'term', 'dl_score'])
        except:
            print("Error loading DL submission")
            return

        # 4. Merge
        print(">>> 4. Merging Strategies")
        print("Merging")
        merged = pd.merge(df_dl, df_knn, on=['id', 'term'], how='outer')
        merged['dl_score'] = merged['dl_score'].fillna(0)
        merged['knn_score'] = merged['knn_score'].fillna(0)
        
        merged['ensemble_score'] = (cfg.W_DL * merged['dl_score']) + (cfg.W_KNN * merged['knn_score'])
        
        print("   Overriding with Exact Matches")
        if not df_exact.empty:
            merged = pd.merge(merged, df_exact, on=['id', 'term'], how='outer')
            merged['exact_score'] = merged['exact_score'].fillna(0)
            merged['final_score'] = merged[['ensemble_score', 'exact_score']].max(axis=1)
        else:
            merged['final_score'] = merged['ensemble_score']
            
        print(">>> 5. Optimizing")
        
        print(f"Filtering scores < {cfg.MIN_SCORE}")
        final_df = merged[merged['final_score'] > cfg.MIN_SCORE]
        
        print(f"Rounding scores")
        final_df['final_score'] = final_df['final_score'].round(cfg.ROUND_DECIMALS)
        
        print(f"Limiting ")
        final_df = final_df.sort_values(['id', 'final_score'], ascending=[True, False])
        final_df = final_df.groupby('id').head(cfg.TOP_K)
        
        # Save
        print(">>> Saving submission.tsv")
        final_df[['id', 'term', 'final_score']].to_csv('submission.tsv', sep='\t', index=False, header=False)
        
        print("Done!")
    if __name__ == "__main__":
        main()