# ColorCorrection-Murals


This repository is the official PyTorch implementation of "Descanning: From Scanned to the Original Images with a Color Correction Diffusion Model".

## 🔭 Requirements

```
python >= 3.8  
torch >= 1.10.2
torchvision >= 0.11.3  
tqdm >= 4.62.2  
numpy >= 1.22.1  
opencv-python >= 4.5.4.60  
natsort >= 8.1.0  
matplotlib >= 3.4.3  
Pillow >= 9.4.0  
scipy >= 1.7.3  
scikit-image >= 0.16.2  
```

```
pip install -r requirements.txt
```

## 💫 Training

To train Color Encoder:

### Color Encoder (Global Color Correction)

1. Configure settings in ```color_encoder/train_color_encoder.py```. (e.g. dataset path, batch size, epochs).
 - If you want to log the training process, set ```logging=True```.  
2. Execute the below code to train the color encoder.
   ```
   python3 color_encoder/train_color_encoder.py
   ```
3. The last saved model will become ```color_encoder.h5```. It will used to train the conditional DDPM (below part).
