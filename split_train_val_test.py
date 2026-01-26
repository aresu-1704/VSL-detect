import json
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def split_train_val_test():
    """
    Chia dataset thành train/val/test với tỷ lệ 80/10/10.
    Đảm bảo mỗi gloss đều có trong cả 3 splits.
    """
    
    # Đường dẫn
    data_processed_path = Path("Dataset/processed")
    data_path = data_processed_path / "data"
    output_path = Path("Split")
    
    # Xóa thư mục Split cũ nếu có
    if output_path.exists():
        print(f"Đang xóa thư mục Split cũ...")
        shutil.rmtree(output_path)
    
    # Tạo thư mục output
    train_path = output_path / "train"
    val_path = output_path / "val"
    test_path = output_path / "test"
    
    for path in [train_path, val_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SPLIT DATASET - Train/Val/Test (80/10/10)")
    print("=" * 60)
    
    # Đọc label map
    print("\n[1/3] Đọc label map...")
    with open(data_processed_path / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    print(f"  ✓ Tìm thấy {len(label_map)} glosses trong label_map")
    
    # Copy label_map vào thư mục Split
    shutil.copy2(data_processed_path / "label_map.json", output_path / "label_map.json")
    print(f"  ✓ Đã copy label_map.json vào {output_path}")
    
    # Lấy danh sách tất cả các thư mục gloss
    gloss_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"\n[2/3] Tìm thấy {len(gloss_dirs)} gloss directories trong data/")
    
    # Thống kê
    total_files = 0
    train_count = 0
    val_count = 0
    test_count = 0
    glosses_processed = 0
    glosses_empty = 0
    
    print("\n[3/3] Đang split từng gloss...")
    
    for gloss_dir in tqdm(gloss_dirs, desc="  Processing glosses"):
        folder_name = gloss_dir.name
        
        # Lấy tất cả file .npz trong gloss này
        npz_files = list(gloss_dir.glob("*.npz"))
        
        if len(npz_files) == 0:
            glosses_empty += 1
            # Vẫn tạo thư mục rỗng để đảm bảo structure nhất quán
            (train_path / folder_name).mkdir(exist_ok=True)
            (val_path / folder_name).mkdir(exist_ok=True)
            (test_path / folder_name).mkdir(exist_ok=True)
            continue
        
        # Shuffle ngẫu nhiên
        random.shuffle(npz_files)
        
        # Tính số lượng cho mỗi split
        n_files = len(npz_files)
        n_train = int(n_files * 0.8)
        n_val = int(n_files * 0.1)
        # n_test sẽ là phần còn lại để đảm bảo không mất file
        
        # Chia files
        train_files = npz_files[:n_train]
        val_files = npz_files[n_train:n_train + n_val]
        test_files = npz_files[n_train + n_val:]
        
        # Tạo thư mục cho gloss trong mỗi split
        train_gloss_dir = train_path / folder_name
        val_gloss_dir = val_path / folder_name
        test_gloss_dir = test_path / folder_name
        
        train_gloss_dir.mkdir(exist_ok=True)
        val_gloss_dir.mkdir(exist_ok=True)
        test_gloss_dir.mkdir(exist_ok=True)
        
        # Copy files vào mỗi split
        for f in train_files:
            shutil.copy2(f, train_gloss_dir / f.name)
            train_count += 1
        
        for f in val_files:
            shutil.copy2(f, val_gloss_dir / f.name)
            val_count += 1
        
        for f in test_files:
            shutil.copy2(f, test_gloss_dir / f.name)
            test_count += 1
        
        total_files += n_files
        glosses_processed += 1
    
    # Kiểm tra số lượng folders trong mỗi split
    train_folders = len([d for d in train_path.iterdir() if d.is_dir()])
    val_folders = len([d for d in val_path.iterdir() if d.is_dir()])
    test_folders = len([d for d in test_path.iterdir() if d.is_dir()])
    
    # Tổng kết
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)
    print(f"Số lượng glosses có data: {glosses_processed}")
    print(f"Số lượng glosses rỗng: {glosses_empty}")
    print(f"Tổng số files: {total_files}")
    print(f"\n{'Split':<10} {'Files':<10} {'Tỷ lệ':<10} {'Folders':<10}")
    print("-" * 60)
    print(f"{'Train':<10} {train_count:<10} {train_count/total_files*100:>6.1f}%   {train_folders:<10}")
    print(f"{'Val':<10} {val_count:<10} {val_count/total_files*100:>6.1f}%   {val_folders:<10}")
    print(f"{'Test':<10} {test_count:<10} {test_count/total_files*100:>6.1f}%   {test_folders:<10}")
    
    # Kiểm tra consistency
    print("\n" + "-" * 60)
    if train_folders == val_folders == test_folders == len(gloss_dirs):
        print("✓ PASS: Tất cả splits đều có đủ số lượng glosses!")
    else:
        print("✗ WARNING: Số lượng glosses không đồng nhất giữa các splits!")
    
    print(f"\nOutput directory: {output_path.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    # Set random seed để có thể reproduce
    random.seed(42)
    np.random.seed(42)
    
    split_train_val_test()
