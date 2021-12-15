from .darknet_cv import yolo
from glob import glob
import os
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import time
import copy
import random
import cv2
#from google.colab.patches import cv2_imshow
import glob
from efficientnet_pytorch import EfficientNet
from collections import Counter

model_name = 'efficientnet-b0' 

image_size = EfficientNet.get_image_size(model_name)

os.chdir("barcodeless/prediction/pred_yolo")
PATH = os.getcwd().replace("\\", "/")
print("-!"*30)

print(PATH)
EFFICIENTNET = [ PATH+"/efficient_weights/"+i for i in os.listdir("efficient_weights")]
# print(EFFICIENTNET)
efficientnet_path = {}
for net in EFFICIENTNET:
    path = net
    category = net.split("/")[-1].split("_")[0]
    efficientnet_path[category] = path


print("-"*40)
# print(efficientnet_path)



TYPES = ['corn', 'sand', 'bucket', 'bar', 'drink', 'snack']
CLASS_NAMES = {
    'corn' : {
        '0' : '라베스트',
        '1' : '로투스',
        '2' : '부라보콘',
        '3' : '와쿠와쿠',
        '4' : '월드콘',
    },
    'sand' : {
        '0' : '국화빵',
        '1' : '붕어싸만코',
        '2' : '빵또아',
        '3' : '잇츠와플',
        '4' : '찰떡아이스',
        '5' : '쿠키오'
    },
    'bucket' : {
        '0' : '구구크러스터',
        '1' : '위즐',
    },
    'bar' : {
        '0' : '누가바',
        '1' : '돼지바',
        '2' : '메로나',
        '3' : '비비빅',
        '4' : '빠삐코',
        '5' : '뽕따',
        '6' : '수박바',
        '7' : '쌍쌍바',
        '8' : '옥동자',
        '9' : '죠스바',
        '10' : '주물러',
        '11' : '캔디바',
        '12' : '쿠앤크',
    },
    'drink' : {
        '0' : '데미소다',
        '1' : '몬스타',
        '2' : '밀키스',
        '3' : '스타벅스',
        '4' : '스프라이트',
        '5' : '쌕쌕',
        '6' : '캐나다드라이',
        '7' : '코카콜라',
        '8' : '환타'
    },
    'snack' : {
        '0' : '롤리폴리',
        '1' : '롯데샌드',
        '2' : '마가렛트',
        '3' : '몽쉘',
        '4' : '빅파이',
        '5' : '빠다코코낫',
        '6' : '찰떡파이',
        '7' : '초코쿠키',
        '8' : '쿠크다스',
        '9' : '포키'
    },
}
# print(EFFICIENTNET)

def yolo_result_list():
    res_images = os.listdir("C:/Users/user/Desktop/workspace/stone/media/result")
    print(f"res_images{res_images}")
    return res_images


def effi(IMAGE_LIST):
    items = []
    for image in IMAGE_LIST:
        type = image.split("_")[1]
        if type in TYPES:
            print(f"type : {type}")
            class_names = CLASS_NAMES[type]

            ## 학습 코드
            # print(len(CLASS_NAMES[type]))
            model = EfficientNet.from_pretrained(model_name, num_classes=len(CLASS_NAMES[type])) 
            # print(efficientnet_path[type])
            device = torch.device('cpu')
            model.load_state_dict(torch.load(efficientnet_path[type], map_location=device))
            # print(model)

            image_list = [ "C:/Users/user/Desktop/workspace/stone/media"+"/result/"+image for image in os.listdir("C:/Users/user/Desktop/workspace/stone/media/result") ]  # test image 넣기

            # print(image_list)

            # for i in range(len(image_list)):
            # print(image_list[i].split('/')[-1].split("_")[1])

            image_in = "C:/Users/user/Desktop/workspace/stone/media"+"/result/"+image

            temp = image_in.split('/')[-1].split("_")[1]
            if temp == type:
                image = cv2.imread(image_in, cv2.IMREAD_ANYCOLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                #cv2.imshow(f"test{i}",image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                # Preprocess image
                tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
                img = tfms(Image.open(image_in)).unsqueeze(0)

                # Classify
                model.eval()
                with torch.no_grad():
                    # print(img)
                    # imshow(img)
                    # img = img.cuda()
                    outputs = model(img)

                # Print predictions
                # plt.imshow(test_image)
                print(image_in)
                print(f"{type}")
                for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():    
                    prob = torch.softmax(outputs, dim=1)[0, idx].item()
                    print('[', class_names[str(idx)], ': {p:.2f}% ]'.format(p=prob*100))
                    items.append(class_names[str(idx)])
                print('-----')
            else:
                continue
        else:
            continue
    return Counter(items)   


def rm_results():
    res_images = os.listdir("C:/Users/user/Desktop/workspace/stone/media/result")
    # print(res_images)
    for image in res_images:
        os.remove("C:/Users/user/Desktop/workspace/stone/media"+"/result/"+image)


if __name__ in "__main__":
    # "C:/Users/user/Desktop/workspace/stone/media/images"
    IMAGE_PATH = "test.jpg"
    yolo(IMAGE_PATH)
    image_ls = yolo_result_list()
    print(image_ls)
    effi(image_ls)


    rm_results()