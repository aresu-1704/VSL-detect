import os

# Danh sách gloss (viết hoa)
folders = [
    "GIA ĐÌNH", "HỌ HÀNG", "NHÀ", "TỔ TIÊN", "ÔNG BÀ",
    "ÔNG", "BÀ", "NỘI", "NGOẠI", "BA",
    "MẸ", "CHA MẸ", "CON", "CON NUÔI", "ANH EM",
    "ANH", "CHỊ", "EM", "ANH TRAI", "CHỊ GÁI",
    "EM TRAI", "EM GÁI", "BÁC", "CHÚ", "MỢ",
    "THÍM", "CÔ", "DÌ", "HỌ NGOẠI", "ANH CẢ",
    "ANH HAI", "CHỊ CẢ", "CHỊ HAI", "ÚT", "ANH RUỘT",
    "ANH HỌ", "ANH RỂ", "ANH VỢ", "ANH CHỊ EM", "CHỊ HỌ",
    "CHỊ DÂU", "CHỊ CHỒNG", "EM HỌ", "EM RỂ", "EM TRAI",
    "EM DÂU", "EM GÁI", "EM GÁI", "EM TRAI", "ÔNG NỘI",
    "ÔNG NGOẠI", "BÀ NỘI", "BÀ NGOẠI", "HẠNH PHÚC", "BÌNH AN"
]

# Thư mục gốc để chứa tất cả
root_dir = "../../dataset/GIA ĐÌNH"

# Tạo thư mục gốc nếu chưa có
os.makedirs(root_dir, exist_ok=True)

# Tạo từng thư mục con
for folder in folders:
    path = os.path.join(root_dir, folder)
    os.makedirs(path, exist_ok=True)

print("Đã tạo xong tất cả thư mục.")
