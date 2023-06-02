import mlflow
import mlflow.pytorch
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import requests
import time
from datetime import datetime

# MLflow 서버에 로그를 저장합니다.
mlflow.set_tracking_uri("http://172.20.0.2:5000")

# mlflow experiment 저장 폴더 이름 변경 (현재 날짜)
now = datetime.now()

# 실험 생성 (실험이름을 날짜와 시간으로 지정)
mlflow.set_experiment(now.strftime("%Y%m%d"))

# gpu 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current cuda device is', device)

# build model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(3136, 1000)   
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
           
           
# 하이퍼 파라미터 설정
batch_sizes = [16, 32, 64]
learning_rates = [0.01, 0.001, 0.0001]
epoch_nums = [10, 15, 20]

     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/sooh/final-proj/user/data")
    args = parser.parse_args()

    # Load MNIST dataset
    data_path = args.data  # 데이터셋 경로를 인자로부터 받음
    train = pd.read_csv(os.path.join(data_path, 'mnist_train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'mnist_test.csv'))
    
    X_train = train.drop("label", axis=1).values.reshape(-1, 1, 28, 28).astype('float32') / 255
    y_train = train["label"].values
    X_test = test.drop("label", axis=1).values.reshape(-1, 1, 28, 28).astype('float32') / 255
    y_test = test["label"].values
    
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # 각 하이퍼파라미터 조합에 대해
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for epoch_num in epoch_nums:
                # MLflow 실행 시작
                with mlflow.start_run(run_name="ml-run"):
                    # 하이퍼파라미터 로깅
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("learning_rate", learning_rate)
                    mlflow.log_param("epoch_num", epoch_num)

                    # 데이터 로더 설정
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

                    # 모델을 CUDA 디바이스로 이동
                    model = CNN().to(device)

                    # 손실 함수와 최적화 도구 설정
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = nn.CrossEntropyLoss()

                    # 학습 루프
                    for epoch in range(epoch_num):
                        model.train()
                        train_loss = 0.0
                        for images, labels in train_loader:
                            images = images.to(device)
                            labels = labels.to(device)
                            
                            # 순방향 패스
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            
                            # 역방향 패스와 가중치 갱신
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            train_loss += loss.item()

                        # 에포크 마다 손실 로깅
                        train_loss = train_loss / len(train_loader)
                        print(f"Epoch [{epoch+1}/{epoch_num}], Train Loss: {train_loss}")
                        mlflow.log_metric("train_loss", train_loss, step=epoch)

                    # 검증 루프
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        correct = 0
                        total = 0
                        for images, labels in val_loader:
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            val_loss += loss.item()
                        
                        # 에포크 마다 검증 손실과 정확도 로깅
                        val_loss = val_loss / len(val_loader)
                        val_acc = correct / total
                        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
                        mlflow.log_metric("val_loss", val_loss)
                        mlflow.log_metric("val_acc", val_acc)

                    # 학습된 모델 저장
                    mlflow.pytorch.log_model(model, "model")
                    
