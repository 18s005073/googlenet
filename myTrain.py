import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import GoogLeNet


def main():
    # 指定使用GPU还是CPU进行训练
    device = torch.device("cpu")
    print("Using {} device".format(device))
    # 设置训练参数
    batch_size = 32
    epochs = 10
    best_acc = 0.0
    save_path = "./GoogleNet.pth"
    learn_rate = 0.0003

    # 对数据进行转换
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # 设置文件路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "data_set", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 加载数据
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # 将数据类别保存为json文件
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    # 初始化网络，设置优化函数
    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)

    # 开始训练wangluo
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # 训练
        net.train()
        runing_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            runing_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():  # 不进行梯度运算
            val_bar = tqdm(validate_loader,file=sys.stdout)
            for val_data in val_bar:
                val_images,val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y,val_labels.to(device)).sum().item()

        val_accurate = acc/val_num
        print("[epoch {}] train_loss:{:.3f} val_accurate:{:.3f}"
              .format(epoch+1),runing_loss/train_steps,val_accurate)

        if val_accurate>best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(),save_path)

    print("Finished Training")

if __name__ == '__main__':
    main()




