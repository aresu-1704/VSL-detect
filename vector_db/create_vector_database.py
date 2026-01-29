import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pickle

# Import encoder
import sys
sys.path.append(os.path.dirname(__file__))
from encoders.encode import VSLEncoder


def create_vector_database(dataset_dir, ckpt_path, output_path, device="cpu"):
    """
    Tạo vector database từ các file .npz trong Dataset/Processed
    
    Args:
        dataset_dir: Đường dẫn tới Dataset/Processed
        ckpt_path: Đường dẫn tới file checkpoint của encoder
        output_path: Đường dẫn lưu vector database
        device: "cpu" hoặc "cuda"
    """
    # Khởi tạo encoder
    print(f"Đang khởi tạo encoder từ {ckpt_path}...")
    encoder = VSLEncoder(ckpt_path=ckpt_path, device=device)
    
    # Dictionary lưu vectors
    # Cấu trúc: {gloss_name: {'embeddings': [emb1, emb2, ...], 'file_paths': [path1, path2, ...]}}
    database = {}
    
    # Đọc label_map.json để biết tên các gloss
    import json
    label_map_path = os.path.join(dataset_dir, "label_map.json")
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    print(f"Tìm thấy {len(label_map)} gloss trong label_map.json")
    
    # Duyệt qua từng thư mục gloss
    dataset_path = Path(dataset_dir)
    gloss_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    print(f"\nĐang xử lý {len(gloss_folders)} thư mục gloss...")
    
    for gloss_folder in tqdm(gloss_folders, desc="Xử lý các gloss"):
        gloss_name = gloss_folder.name
        
        # Khởi tạo entry cho gloss này
        if gloss_name not in database:
            database[gloss_name] = {
                'embeddings': [],
                'file_paths': [],
                'label': None
            }
        
        # Duyệt qua các file .npz trong thư mục
        npz_files = list(gloss_folder.glob('*.npz'))
        total_files += len(npz_files)
        
        for npz_file in npz_files:
            try:
                # Load file .npz
                data = np.load(npz_file, allow_pickle=True)
                
                # Lấy label từ file .npz để cross-reference với label_map
                if 'label' in data:
                    label_id = int(data['label'])
                    # Cập nhật label cho gloss này nếu chưa có
                    if database[gloss_name]['label'] is None:
                        database[gloss_name]['label'] = label_id
                
                # Lấy sequence (key 'sequence' hoặc 'input' tùy file)
                if 'sequence' in data:
                    sequence = data['sequence']
                elif 'input' in data:
                    sequence = data['input']
                else:
                    print(f"\n  Warning: Không tìm thấy key 'sequence' hoặc 'input' trong {npz_file}")
                    failed_files += 1
                    continue
                
                # Chuyển sequence thành tensor và encode
                # sequence shape: (T, 201)
                x = torch.tensor(sequence, dtype=torch.float32)
                x = x.unsqueeze(0).to(encoder.device)  # (1, T, 201)
                
                # Encode
                with torch.no_grad():
                    embedding = encoder.model(x)  # (1, 256)
                    embedding = embedding[0].cpu().numpy()  # (256,)
                
                # Lưu vào database
                database[gloss_name]['embeddings'].append(embedding)
                database[gloss_name]['file_paths'].append(str(npz_file))
                processed_files += 1
                
            except Exception as e:
                print(f"\n  Lỗi khi xử lý {npz_file}: {e}")
                failed_files += 1
                continue
    
    # Chuyển embeddings từ list sang numpy array
    print("\nĐang chuyển đổi embeddings sang numpy arrays...")
    for gloss_name in database:
        if database[gloss_name]['embeddings']:
            database[gloss_name]['embeddings'] = np.array(database[gloss_name]['embeddings'])
    
    # Lưu database
    print(f"\nĐang lưu vector database vào {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(database, f)
    
    # Thống kê
    print(f"\n{'='*60}")
    print(f"Hoàn tất tạo vector database!")
    print(f"{'='*60}")
    print(f"Tổng số file .npz: {total_files}")
    print(f"  - Xử lý thành công: {processed_files}")
    print(f"  - Thất bại: {failed_files}")
    print(f"\nSố lượng gloss: {len(database)}")
    print(f"Vector database được lưu tại: {output_path}")
    
    # In thống kê chi tiết
    total_embeddings = sum(len(db['embeddings']) for db in database.values())
    print(f"Tổng số embeddings: {total_embeddings}")
    
    return database


def load_vector_database(db_path):
    """
    Load vector database từ file
    
    Args:
        db_path: Đường dẫn tới file database
        
    Returns:
        database dictionary
    """
    with open(db_path, 'rb') as f:
        database = pickle.load(f)
    return database


def query_similar(database, query_embedding, top_k=5):
    """
    Tìm các gloss tương tự với query embedding
    
    Args:
        database: Vector database
        query_embedding: Embedding cần tìm (256,)
        top_k: Số lượng kết quả trả về
        
    Returns:
        List of (gloss_name, similarity_score, file_path)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    results = []
    
    for gloss_name, data in database.items():
        embeddings = data['embeddings']
        if len(embeddings) == 0:
            continue
        
        # Tính cosine similarity với tất cả embeddings của gloss này
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            embeddings
        )[0]
        
        # Lấy similarity lớn nhất
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        file_path = data['file_paths'][max_idx]
        
        results.append((gloss_name, max_sim, file_path))
    
    # Sắp xếp theo similarity giảm dần
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


if __name__ == "__main__":
    # Cấu hình
    DATASET_DIR = "./Dataset/Processed"
    CHECKPOINT_PATH = "./encoders/models/BiLSTM_encoder.pt"  # Cập nhật đường dẫn checkpoint
    OUTPUT_PATH = "./vector_db/vector_database.pkl"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Sử dụng device: {DEVICE}")
    
    # Tạo vector database
    database = create_vector_database(
        dataset_dir=DATASET_DIR,
        ckpt_path=CHECKPOINT_PATH,
        output_path=OUTPUT_PATH,
        device=DEVICE
    )
    
    # Demo: Load và query
    print("\n" + "="*60)
    print("Demo: Load database và thử query")
    print("="*60)
    
    # Load database
    loaded_db = load_vector_database(OUTPUT_PATH)
    print(f"Đã load database với {len(loaded_db)} gloss")
    
    # Lấy một embedding ngẫu nhiên để test
    sample_gloss = list(loaded_db.keys())[0]
    sample_embedding = loaded_db[sample_gloss]['embeddings'][0]
    
    print(f"\nQuery với embedding từ gloss: '{sample_gloss}'")
    results = query_similar(loaded_db, sample_embedding, top_k=5)
    
    print("\nTop 5 kết quả tương tự:")
    for i, (gloss, score, path) in enumerate(results, 1):
        print(f"{i}. {gloss}: {score:.4f}")
        print(f"   File: {os.path.basename(path)}")
