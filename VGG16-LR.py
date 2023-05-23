import torch
import argparse
from PIL import Image
import numpy as np
import cv2
from torchvision import datasets, models, transforms
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import pickle
import csv
import os


class Predictor:

    def __init__(self, img_path, result_path):
        self.img_path = img_path
        self.result_path = result_path + 'predicted_result.csv'
        self.weight_path = "./Model/VGG16_all_best_weights.pt"
        self.model_path = "./Model/VGG16-LR.pkl"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def img2tensor(self):
        names = []
        list_images = []
        files = os.listdir(self.img_path)
        for filename in files:
            if filename.endswith(".jpg"):
                name = filename.split(".")[0]
                image = np.array(Image.open(self.img_path+'/'+filename).convert("L"))
                # image = cv2.imread(os.path.join(self.img_path, filename))
                list_images.append(torch.from_numpy(cv2.resize(image / 255 * 11073, (256, 256))).to(torch.float32))
                names.append(name)
        tensor_images = torch.stack(list_images)
        tensor_images = tensor_images.unsqueeze(1)
        return tensor_images, names

    def define_model(self):
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.features.parameters():
            param.requires_grad = False
        vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        num_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_features, 3)
        return vgg16

    def extract_features(self, data_loader):
        model = self.define_model()
        model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        # remove the fully connected layers of VGG16
        vgg16 = nn.Sequential(*list(model.features.children()))
        vgg16.eval()
        features = []
        with torch.no_grad():
            for data in data_loader:
                inputs = data.to(self.device)
                output = vgg16(inputs)
                output = output.view(output.size(0), -1)
                features.append(output)
        features = torch.cat(features, dim=0)
        return features

    def predict(self):
        output = []
        fields = ['Image number', 'Kind']
        images_tensor, names = self.img2tensor()
        data_loader = DataLoader(images_tensor, batch_size=32, shuffle=False)
        features = self.extract_features(data_loader)
        # load pre-trained Logistic regresion classifier
        with open(self.model_path, 'rb') as f:
            vgg16_lr = pickle.load(f)
        # Use the pre-trained vgg16_lr to perform classification tasks
        result = vgg16_lr.predict(features)
        for i in range(len(result)):
            if result[i] == '0':
                kind = 'meningioma'
            elif result[i] == '1':
                kind = 'glioma'
            else:
                kind = 'pituitary tumor'
            output.append([names[i], kind])
        #save csv file
        with open(self.result_path, 'w') as f:
        # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="It's usage tip.", description="Running the CNN-SVM")
    parser.add_argument('-f', '-image_path', required=True, type=str, help='The brain tumor images path')
    parser.add_argument('-s', '-save_path', required=True, type=str, help='The path to save your predicted results')
    args = parser.parse_args()

    image_path = args.f
    result_path = args.s
    predictor = Predictor(image_path, result_path)
    predictor.predict()
