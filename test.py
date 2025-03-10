from fileinput import filename
import os
from tokenize import group
import natsort
import torch
import torchvision
import natsort
import torch.nn as nn
from torchvision import transforms, models
from torchvision import transforms
from color_encoder.utils_color_encoder import *
from PIL import Image



def load_color_encoder( color_encoder_path, device):
    color_encoder = models.resnet34(pretrained=False)
    color_encoder.fc = nn.Linear(in_features=512, out_features=6, bias=False)
    color_encoder = color_encoder.to(device)
    encoder_checkpoint = torch.load(color_encoder_path, map_location=device)
    color_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    color_encoder.eval()
    return color_encoder


def ColorCorrection( model_function, filename , device):
    test_folder = "exp/test_ref"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(512)])
    newpath = "exp/y_color_correction"
    
    #print(f'\n[ Validation at epoch {epoch+1} ]')
    image_path = os.path.join(test_folder, filename)

    if os.path.exists(image_path):
        img_scan = transform(Image.open(image_path)).to(device)

        output = model_function(img_scan[None, ...].to(device))
        [r_mean, g_mean, b_mean, r_std, g_std, b_std] = output[0].cpu().numpy()

        mean_pred = r_mean, g_mean, b_mean
        std_pred = r_std, g_std, b_std

        image_scan, mean_scan, std_scan = load_and_preprocess_image(image_path)

        pred_shift = apply_color_shift(image_scan.copy(), mean_pred, std_pred, mean_scan, std_scan)

        pred_shift = transform(pred_shift)

        torchvision.utils.save_image(pred_shift, f'{newpath}/{filename}')#_{epoch+1}

def ColorCorrection_y0(img_tensors, folder, model_function, device ):
    transform = transforms.Compose([transforms.Resize(512)]) # 这里不需要ToTensor()
    newpath = "exp/y_color_correction"
    
    # 确保输出路径存在
    os.makedirs(newpath, exist_ok=True)

    # 对每张图像进行处理
    for i, img_tensor in enumerate(img_tensors):
        img_tensor = img_tensor.to(device) # 保证输入数据在同一设备上

        # 通过模型获得颜色参数预测
        output = model_function(img_tensor[None, ...].to(device)) # 输入形状应为[1, 3, 512, 512]
        [r_mean, g_mean, b_mean, r_std, g_std, b_std] = output[0].cpu().detach().numpy()

        mean_pred = r_mean, g_mean, b_mean
        std_pred = r_std, g_std, b_std

        # 假设你现在已经有原始图像的张量，不需要从文件加载
        # 直接从 img_tensor 计算均值和标准差
        mean_scan = img_tensor.mean(dim=[1, 2]) # 按[H, W]维度计算均值
        std_scan = img_tensor.std(dim=[1, 2]) # 按[H, W]维度计算标准差

        # 计算颜色偏移
        pred_shift = apply_color_shift(img_tensor.clone(), mean_pred, std_pred, mean_scan.cpu().numpy(), std_scan.cpu().numpy())

        # 这里不需要再进行转换为Tensor，因为输入本身已经是Tensor
        # 如果需要保存为图像文件，则需要进行一些后处理，如转为[0, 1]范围的值或保存为图片
        torchvision.utils.save_image(pred_shift, f'{newpath}/corrected_image_{i}.png')
