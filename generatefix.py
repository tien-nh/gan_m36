# import pandas as pd
# import os

# def filter_and_add_path():
#     # --- CẤU HÌNH ---
#     INPUT_CSV = "../datacsv/fold_3_test.csv"
#     OUTPUT_CSV = "../datacsv/fold_3_test_add_generated.csv" # File kết quả
#     GEN_FOLDER = "Generated_Test_M36" # Tên folder chứa ảnh đã sinh
    
#     print(f"--- Đang xử lý file: {INPUT_CSV} ---")
    
#     if not os.path.exists(INPUT_CSV):
#         print(f"LỖI: Không tìm thấy file {INPUT_CSV}")
#         return

#     # 1. Đọc file
#     df = pd.read_csv(INPUT_CSV)
#     print(f"-> Tổng số dòng ban đầu: {len(df)}")

#     # 2. LỌC DỮ LIỆU (Bước quan trọng nhất)
#     # Chỉ giữ lại Class 1 (CN to CN) và Class 5 (MCI to AD)
#     valid_classes = ["Class 1 (CN to CN)", "Class 5 (MCI to AD)"]
    
#     # Kiểm tra xem cột 'class_label' có tồn tại không
#     if 'class_label' not in df.columns:
#         print("LỖI: File CSV không có cột 'class_label'. Kiểm tra lại tên cột!")
#         return

#     # Lệnh lọc: Giữ lại dòng mà class_label nằm trong danh sách valid_classes
#     df_filtered = df[df['class_label'].isin(valid_classes)].copy()
    
#     print(f"-> Số dòng sau khi lọc (Chỉ giữ CN-CN và MCI-AD): {len(df_filtered)}")

#     # 3. Thêm cột đường dẫn ảnh sinh
#     # Logic: Generated_M36/<subjectID>.nii.gz
    
#     def generate_path(row):
#         subj_id = str(row['subject ID'])
#         file_name = f"{subj_id}.nii.gz"
#         full_path = os.path.join(GEN_FOLDER, file_name)
        
#         # Kiểm tra file có tồn tại không
#         if os.path.exists(full_path):
#             return full_path
#         else:
#             # Nếu lọc đúng mà vẫn không thấy ảnh -> Có thể lỗi lúc sinh ảnh
#             print(f"Cảnh báo: Không thấy ảnh sinh cho ID {subj_id}")
#             return None 

#     print("Đang gán đường dẫn ảnh sinh...")
#     df_filtered['img_path_generated'] = df_filtered.apply(generate_path, axis=1)

#     # 4. (Tùy chọn) Xóa luôn những dòng Class 1/5 mà không tìm thấy ảnh sinh (nếu có)
#     # Để đảm bảo file sạch 100% không có ô rỗng (NaN)
#     missing_count = df_filtered['img_path_generated'].isna().sum()
#     if missing_count > 0:
#         print(f"-> Phát hiện {missing_count} dòng thuộc Class 1/5 nhưng KHÔNG CÓ ảnh sinh. Đang xóa...")
#         df_filtered = df_filtered.dropna(subset=['img_path_generated'])

#     # 5. Lưu file
#     df_filtered.to_csv(OUTPUT_CSV, index=False)
#     print(f"-> HOÀN TẤT! File mới đã lưu tại: {OUTPUT_CSV}")
#     print(f"-> Tổng số dòng cuối cùng: {len(df_filtered)}")
    
#     # In thử 5 dòng đầu
#     print("\n--- 5 DÒNG ĐẦU TIÊN ---")
#     print(df_filtered[['subject ID', 'class_label', 'img_path_generated']].head())

# if __name__ == "__main__":
#     filter_and_add_path()