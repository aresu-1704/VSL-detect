import pandas as pd
import shutil
import os
from pathlib import Path
from tqdm import tqdm

def split_dataset(
    label_csv_path="Dataset/Labels/label.csv",
    videos_dir="Dataset/Videos",
    output_dir="Dataset",
    split_at_row=2500,
    batch_size=50
):
    """
    Tách dataset thành 2 phần dựa vào số dòng.
    
    Args:
        label_csv_path: Đường dẫn đến file CSV chứa labels
        videos_dir: Thư mục chứa các file video gốc
        output_dir: Thư mục cha để tạo Data_1 và Data_2
        split_at_row: Số dòng đầu tiên cho Data_1 (không tính header)
        batch_size: Số lượng video copy mỗi lần
    """
    
    # Đọc file CSV
    print("Đang đọc file CSV...")
    df = pd.read_csv(label_csv_path)
    print(f"Tổng số dòng (không tính header): {len(df)}")
    
    # Tách dataframe thành 2 phần
    df_part1 = df.iloc[:split_at_row]
    df_part2 = df.iloc[split_at_row:]
    
    print(f"\nData_1: {len(df_part1)} dòng")
    print(f"Data_2: {len(df_part2)} dòng")
    
    # Tạo cấu trúc thư mục
    parts = [
        ("Data_1", df_part1),
        ("Data_2", df_part2)
    ]
    
    for part_name, part_df in parts:
        print(f"\n{'='*60}")
        print(f"Xử lý {part_name}")
        print(f"{'='*60}")
        
        # Tạo thư mục
        part_dir = Path(output_dir) / part_name
        videos_output_dir = part_dir / "Videos"
        labels_output_dir = part_dir / "Labels"
        
        videos_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Đã tạo thư mục: {part_dir}")
        
        # Lưu file CSV
        label_output_path = labels_output_dir / "label.csv"
        part_df.to_csv(label_output_path, index=False)
        print(f"Đã lưu file CSV: {label_output_path}")
        
        # Copy video theo batch
        video_files = part_df['VIDEO'].tolist()
        total_videos = len(video_files)
        copied_count = 0
        skipped_count = 0
        
        print(f"\nĐang copy {total_videos} video...")
        
        # Sử dụng tqdm để hiển thị tiến trình
        with tqdm(total=total_videos, desc=f"Copy videos {part_name}") as pbar:
            for i in range(0, total_videos, batch_size):
                batch = video_files[i:i + batch_size]
                
                for video_name in batch:
                    src_path = Path(videos_dir) / video_name
                    dst_path = videos_output_dir / video_name
                    
                    try:
                        if src_path.exists():
                            shutil.copy2(src_path, dst_path)
                            copied_count += 1
                        else:
                            print(f"\n⚠️  Không tìm thấy: {src_path}")
                            skipped_count += 1
                    except Exception as e:
                        print(f"\n❌ Lỗi khi copy {video_name}: {e}")
                        skipped_count += 1
                    
                    pbar.update(1)
        
        print(f"\n✅ Hoàn thành {part_name}:")
        print(f"   - Đã copy: {copied_count} videos")
        print(f"   - Bỏ qua: {skipped_count} videos")
    
    print(f"\n{'='*60}")
    print("🎉 HOÀN THÀNH TÁCH DATASET!")
    print(f"{'='*60}")
    print(f"\nCấu trúc thư mục đã tạo:")
    print(f"  {output_dir}/")
    print(f"    ├── Data_1/")
    print(f"    │   ├── Videos/  ({len(df_part1)} videos)")
    print(f"    │   └── Labels/")
    print(f"    │       └── label.csv ({len(df_part1)} dòng)")
    print(f"    └── Data_2/")
    print(f"        ├── Videos/  ({len(df_part2)} videos)")
    print(f"        └── Labels/")
    print(f"            └── label.csv ({len(df_part2)} dòng)")


if __name__ == "__main__":
    # Chạy script với các tham số mặc định
    split_dataset(
        label_csv_path="Dataset/Labels/label.csv",
        videos_dir="Dataset/Videos",
        output_dir="Dataset",
        split_at_row=2500,  # 2500 dòng đầu cho Data_1
        batch_size=50       # Copy 50 videos mỗi lần
    )
