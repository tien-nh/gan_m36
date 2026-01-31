import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import pandas as pd
import glob
import random

class LongitudinalCSVDataset(Dataset):
    def __init__(self, root_dir, csv_file, mode='train', target_shape=(160, 160, 96), augment=False):
        """
        root_dir: Đường dẫn đến thư mục gốc (VD: ./filter_ds)
        csv_file: Đường dẫn đến file CSV
        """
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.augment = augment
        # 1. Đọc CSV
        df = pd.read_csv(csv_file)
        
        # 2. Lọc dữ liệu theo Class
        self.data_pairs = []
        valid_classes = ["Class 1 (CN to CN)", "Class 5 (MCI to AD)"]
        
        for idx, row in df.iterrows():
            cls_str = row['class_label']
            if cls_str in valid_classes:
                label = 0 if "Class 1" in cls_str else 1
                
                # CSV chứa path đến file, VD: NIfTI/022_S_0014/.../I28901.nii.gz
                path_a_rel = row['img_path_A'] 
                path_b_rel = row['img_path_B']
                
                self.data_pairs.append({
                    'path_a': path_a_rel,
                    'path_b': path_b_rel,
                    'label': label,
                    'subj_id': row['subject ID']
                })
        
        print(f"Dataset loaded: {len(self.data_pairs)} pairs found.")

    def __len__(self):
        return len(self.data_pairs)

    def _find_nifti_file(self, relative_path):
        """
        Hàm tìm file thông minh:
        1. Kiểm tra chính xác đường dẫn trong CSV.
        2. Nếu không thấy, thử tìm bất kỳ file .nii nào trong thư mục cha (fallback).
        """
        full_path = os.path.join(self.root_dir, relative_path)
        
        # Cách 1: Đường dẫn trong CSV chính xác là file tồn tại
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return full_path
            
        # Cách 2: CSV trỏ sai tên file (hoặc file đã bị đổi tên), nhưng đúng thư mục ngày tháng
        # Lấy thư mục cha: ./filter_ds/NIfTI/022_S_0014/2005-10-25
        parent_dir = os.path.dirname(full_path)
        
        if not os.path.exists(parent_dir):
            # Nếu thư mục ngày tháng cũng không có, thử tìm thư mục tương tự (ít dùng nhưng an toàn)
            raise FileNotFoundError(f"Không tìm thấy thư mục: {parent_dir}")

        # Quét tất cả file .nii hoặc .nii.gz trong thư mục đó
        search_path = os.path.join(parent_dir, "*.nii*")
        found_files = glob.glob(search_path)
        
        if len(found_files) > 0:
            # Lấy file đầu tiên tìm được trong thư mục đó
            return found_files[0]
            
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {parent_dir} (Path gốc: {relative_path})")

    def normalize(self, img_data):
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        if max_val - min_val == 0:
            return np.zeros_like(img_data)
        return (img_data - min_val) / (max_val - min_val)

    def resize_or_pad(self, img_data):
        d, h, w = img_data.shape
        td, th, tw = self.target_shape
        
        new_img = np.zeros(self.target_shape, dtype=np.float32)
        
        start_d = max(0, (td - d) // 2)
        start_h = max(0, (th - h) // 2)
        start_w = max(0, (tw - w) // 2)
        
        end_d = start_d + min(d, td)
        end_h = start_h + min(h, th)
        end_w = start_w + min(w, tw)
        
        src_start_d = max(0, (d - td) // 2)
        src_start_h = max(0, (h - th) // 2)
        src_start_w = max(0, (w - tw) // 2)
        
        src_end_d = src_start_d + (end_d - start_d)
        src_end_h = src_start_h + (end_h - start_h)
        src_end_w = src_start_w + (end_w - start_w)

        new_img[start_d:end_d, start_h:end_h, start_w:end_w] = \
            img_data[src_start_d:src_end_d, src_start_h:src_end_h, src_start_w:src_end_w]
            
        return new_img

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        
        try:
            # 1. Tìm đường dẫn file (Dùng hàm mới đã sửa)
            file_path_a = self._find_nifti_file(item['path_a'])
            file_path_b = self._find_nifti_file(item['path_b'])
            
            # 2. Load NIfTI
            img_a = nib.load(file_path_a).get_fdata()
            img_b = nib.load(file_path_b).get_fdata()
            
            # 3. Chuẩn hóa & Resize
            img_a = self.normalize(img_a)
            img_b = self.normalize(img_b)
            
            img_a = self.resize_or_pad(img_a)
            img_b = self.resize_or_pad(img_b)
            
            # 4. Convert to Tensor (Channel, D, H, W)
            tensor_a = torch.from_numpy(img_a).float().unsqueeze(0)
            tensor_b = torch.from_numpy(img_b).float().unsqueeze(0)

            if self.augment:
                # A. Random Flip (Lật ngang) - Xác suất 50%
                # Lật cả A và B giống hệt nhau để không bị lệch không gian
                if random.random() > 0.5:
                    # dims=[3] là trục chiều ngang (W) trong (C, D, H, W)
                    tensor_a = torch.flip(tensor_a, dims=[3])
                    tensor_b = torch.flip(tensor_b, dims=[3])

                # B. Random Noise (Thêm nhiễu) - Xác suất 50%
                # Thêm nhiễu Gaussian nhẹ (2%) vào ảnh đầu vào để chống Overfit
                if random.random() > 0.5:
                    noise = torch.randn_like(tensor_a) * 0.02
                    tensor_a = tensor_a + noise

            label = torch.tensor(item['label'], dtype=torch.long)
            
            return tensor_a, tensor_b, label
            
        except Exception as e:
            print(f"Error processing subject {item['subj_id']}: {e}")
            # Để tránh crash training, bạn có thể return None và xử lý ở collate_fn, 
            # nhưng tốt nhất là raise để biết data lỗi
            raise e