!pip install ultralytics opencv-python matplotlib
import os
import shutil
import random
import yaml
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def prepare_dataset(raw_data_dir, output_dir, classes):
    """
    Chuẩn bị dataset từ thư mục raw data
    :param raw_data_dir: Thư mục chứa ảnh và nhãn ban đầu
    :param output_dir: Thư mục đầu ra theo chuẩn YOLO
    :param classes: Danh sách các lớp hoa quả
    """
    # Tạo thư mục theo cấu trúc YOLO
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/val', exist_ok=True)

    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(raw_data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)  # Xáo trộn ngẫu nhiên

    # Chia tỉ lệ train/val (80/20)
    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Hàm helper để copy file
    def copy_files(files, split_type):
        for file in files:
            base_name = os.path.splitext(file)[0]
            
            # Copy ảnh
            img_src = f'{raw_data_dir}/{file}'
            img_dst = f'{output_dir}/images/{split_type}/{file}'
            shutil.copy(img_src, img_dst)
            
            # Copy nhãn
            label_src = f'{raw_data_dir}/{base_name}.txt'
            if os.path.exists(label_src):
                label_dst = f'{output_dir}/labels/{split_type}/{base_name}.txt'
                shutil.copy(label_src, label_dst)

    # Copy các file vào thư mục tương ứng
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    # Tạo file data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }

    with open(f'{output_dir}/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f'Đã chuẩn bị xong dataset tại {output_dir}')



def train_fruit_detector(data_yaml, model_size='n', epochs=50, imgsz=640):
    """
    Huấn luyện mô hình YOLOv8 để nhận diện hoa quả
    :param data_yaml: Đường dẫn đến file data.yaml
    :param model_size: Kích thước mô hình (n, s, m, l, x)
    :param epochs: Số epoch huấn luyện
    :param imgsz: Kích thước ảnh đầu vào
    """
    # Load mô hình YOLOv8 pretrained
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Huấn luyện mô hình
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,  # Có thể điều chỉnh tùy VRAM
        patience=10,  # Dừng sớm nếu không cải thiện
        name='fruit_detection',
        optimizer='auto',
        lr0=0.01,  # Learning rate
        device='0' if torch.cuda.is_available() else 'cpu'  # Dùng GPU nếu có
    )
    
    print("Huấn luyện hoàn tất!")
    return results


def visualize_results(result_dir):
    """
    Hiển thị kết quả huấn luyện
    :param result_dir: Thư mục chứa kết quả huấn luyện
    """
    # Đường dẫn đến các biểu đồ
    results_png = f'{result_dir}/results.png'
    val_batch = f'{result_dir}/val_batch0_pred.jpg'
    
    if os.path.exists(results_png):
        # Hiển thị biểu đồ kết quả
        img = cv2.imread(results_png)
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Biểu đồ đánh giá huấn luyện')
        plt.axis('off')
        plt.show()
    
    if os.path.exists(val_batch):
        # Hiển thị ảnh dự đoán mẫu
        img = cv2.imread(val_batch)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Kết quả nhận diện trên tập validation')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # Cấu hình dataset
    RAW_DATA_DIR = 'raw_fruit_data'  # Thư mục chứa ảnh và nhãn gốc
    OUTPUT_DIR = 'fruit_dataset_yolo'  # Thư mục đầu ra
    
    # Danh sách các loại hoa quả (thay đổi theo dataset của bạn)
    FRUIT_CLASSES = ['apple', 'banana', 'orange', 'grape', 'mango']
    
    # Bước 1: Chuẩn bị dataset
    print("Đang chuẩn bị dataset...")
    prepare_dataset(RAW_DATA_DIR, OUTPUT_DIR, FRUIT_CLASSES)
    
    # Bước 2: Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    data_yaml_path = f'{OUTPUT_DIR}/data.yaml'
    train_results = train_fruit_detector(
        data_yaml=data_yaml_path,
        model_size='n',  # n=nan, s=small, m=medium, l=large, x=extra large
        epochs=50,
        imgsz=640
    )
    
    # Bước 3: Hiển thị kết quả
    print("Hiển thị kết quả huấn luyện...")
    result_dir = 'runs/detect/fruit_detection'
    visualize_results(result_dir)
    print("Quá trình hoàn tất! Model tốt nhất được lưu tại:", f'{result_dir}/weights/best.pt')
