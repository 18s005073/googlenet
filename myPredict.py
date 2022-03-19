import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import GoogLeNet

def main():
    device = torch.device("cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),((0.5,0.5,0.5)))]
    )

    # 加载图片
    image_path = "../tulip2.jpg"
    assert os.path.exists(image_path),"file: '{}' does not exist.".format(image_path)
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img,dim=0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path),"file: '{}' does not exist.".format(json_path)

    json_file = open(json_path,'r')
    class_indict = json.load(json_file)

    # 创建模型
    model = GoogLeNet(num_classes=5,aux_logits=False).to(device)
    # 加载模型权重
    weights_path = "./googleNet.pth"
    assert os.path.exists(weights_path),"file: '{}' does not exist.".format(weights_path)
    missing_keys,unexpected_keys = model.load_state_dict(
        torch.load(weights_path,map_location=device),strict=False
    )

    # 进行预测
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output,dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class:{}  prob:{:.3f}".format(
        class_indict[str(predict_cla)],predict[predict_cla].numpy()
    )
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}  prob:{:.3f}".format(class_indict[str(i)],
              predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()




