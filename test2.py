import torch
import torchvision.models as models
from torchvision import transforms, datasets
from torch import nn, optim
from PIL import Image
import numpy as np
import pyautogui
import pygetwindow as gw
import cv2
import time
import os

# Tải mô hình ResNet18 đã huấn luyện sẵn hoặc huấn luyện lại nếu cần
def load_model():
    model = models.resnet18(pretrained=True)

    # Cập nhật mô hình cho nhiệm vụ nhận diện ảnh mẫu
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)  # Thêm 8 lớp mới: tim.png, vang.png, S.png, D.png, F.png, J.png, K.png, L.png

    # Sử dụng GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_path = r'C:\Tools\model.pth'
    if os.path.exists(model_path):
        print("Tải mô hình đã huấn luyện...")
        # Tải mô hình và bỏ qua các tham số của lớp fc
        state_dict = torch.load(model_path)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}  # Bỏ qua lớp fc
        model.load_state_dict(state_dict, strict=False)  # Tải mô hình mà không yêu cầu strict matching
    else:
        print("Không tìm thấy mô hình đã huấn luyện, sẽ huấn luyện lại từ đầu.")
    return model, device

# Đặt hàm loss và optimizer
def set_optimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer

# Chuẩn bị dataset với các ảnh mẫu (tim.png, vang.png, S.png, D.png, F.png, J.png, K.png, L.png)
def prepare_dataset():
    """ Chuẩn bị dataset với các ảnh mẫu. """
    dataset_dir = 'dataset'
    classes = ['tim', 'vang', 'S', 'D', 'F', 'J', 'K', 'L']
    
    # Tạo thư mục cho các lớp nếu chưa tồn tại
    for class_name in classes:
        os.makedirs(os.path.join(dataset_dir, class_name), exist_ok=True)

    # Danh sách ảnh và thư mục của chúng
    images = [
        ('C:\\Tools\\tim.png', 'tim'),
        ('C:\\Tools\\vang.png', 'vang'),
        ('C:\\Tools\\S.png', 'S'),
        ('C:\\Tools\\D.png', 'D'),
        ('C:\\Tools\\F.png', 'F'),
        ('C:\\Tools\\J.png', 'J'),
        ('C:\\Tools\\K.png', 'K'),
        ('C:\\Tools\\L.png', 'L')
    ]
    
    # Lưu ảnh vào thư mục tương ứng
    for img_path, class_name in images:
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize ảnh về kích thước chuẩn của mô hình ResNet
        img.save(os.path.join(dataset_dir, class_name, f"{class_name}.png"))

# Đào tạo mô hình để nhận diện các ảnh mẫu
def train_model(model, device, optimizer, criterion):
    prepare_dataset()

    # Tăng cường dữ liệu (Data Augmentation)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(root='dataset', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Huấn luyện mô hình
    model.train()
    for epoch in range(500):  # Tăng số vòng lặp huấn luyện lên 500 lần
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Tiến hành huấn luyện
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    # Lưu mô hình đã huấn luyện
    model_path = r'C:\Tools\model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Mô hình đã được huấn luyện và lưu trữ tại {model_path}!")

# Hàm nhận diện hình ảnh
def predict(image, model, device):
    # Chuyển ảnh thành tensor
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Nhận diện với mô hình ResNet18
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()

# Hàm kiểm tra sự xuất hiện của ảnh mẫu trong cửa sổ của ứng dụng "Genshin Impact"
def match_image_in_app(app_title="Genshin Impact", model=None, device=None):
    try:
        # Lấy danh sách cửa sổ với title chứa "Genshin Impact"
        app_windows = [win for win in gw.getAllTitles() if app_title in win]

        if not app_windows:
            print(f"Không tìm thấy cửa sổ với tiêu đề chứa '{app_title}'.")
            return
        
        # Lấy cửa sổ đầu tiên tìm thấy
        app_window = gw.getWindowsWithTitle(app_windows[0])[0]

        # Lấy tọa độ cửa sổ của ứng dụng
        left, top, right, bottom = app_window.left, app_window.top, app_window.right, app_window.bottom
        
        # Chụp ảnh màn hình trong vùng cửa sổ ứng dụng
        screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
        screenshot = np.array(screenshot)

        # Chuyển đổi ảnh sang định dạng phù hợp với OpenCV
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(screenshot)

        # Dự đoán đối tượng trong ảnh màn hình
        screen_class_id = predict(img, model, device)

        # Kiểm tra sự xuất hiện của các ảnh mẫu trong cửa sổ ứng dụng
        images = [
            ('C:\\Tools\\tim.png', 'tim'),
            ('C:\\Tools\\vang.png', 'vang'),
            ('C:\\Tools\\S.png', 'S'),
            ('C:\\Tools\\D.png', 'D'),
            ('C:\\Tools\\F.png', 'F'),
            ('C:\\Tools\\J.png', 'J'),
            ('C:\\Tools\\K.png', 'K'),
            ('C:\\Tools\\L.png', 'L')
        ]

        for img_path, class_name in images:
            template = Image.open(img_path)
            predicted_class = predict(template, model, device)
            if screen_class_id == predicted_class:
                print(f"Ảnh {class_name}.png đã xuất hiện trong cửa sổ ứng dụng!")

        # Hiển thị ảnh chụp màn hình (tùy chọn)
        cv2.imshow("Screen Capture", screenshot)
        cv2.waitKey(1)
        
    except IndexError:
        print(f"Không tìm thấy cửa sổ ứng dụng {app_title}.")

# Chọn xem có muốn huấn luyện lại mô hình hay không
def ask_to_train_again():
    response = input("Bạn có muốn huấn luyện lại mô hình không? (y/n): ")
    return response.lower() == 'y'

# Bắt đầu chương trình
if ask_to_train_again():
    model, device = load_model()
    criterion, optimizer = set_optimizer(model)
    print("Đang huấn luyện lại mô hình...")
    train_model(model, device, optimizer, criterion)
else:
    print("Tiếp tục với mô hình hiện tại.")

# Vòng lặp kiểm tra cửa sổ ứng dụng liên tục
while True:
    model, device = load_model()
    match_image_in_app(model=model, device=device)
    time.sleep(0.5)  # Tạm dừng 0.5 giây giữa các lần kiểm tra
