import io
import cv2
import numpy as np
import torch
import base64
from torch import nn
import segmentation_models_pytorch as smp
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)
ENCODER = 'efficientnet-b7'
WEIGHTS = 'imagenet'

class SegmentationModel (nn.Module):
    def __init__(self):
        super (SegmentationModel , self).__init__()

        self.arc = smp.Unet(
            encoder_name= ENCODER , # loading pre-trained model
            encoder_weights=  WEIGHTS , # loading pre-trained weights
            in_channels= 3 ,
            classes=  6, # (we have 6 classes including background)
            activation = 'softmax'
        )
        
        
    def forward(self, images):
        logits = self.arc(images)
        return logits
model = SegmentationModel()
model.to('cpu')
model.load_state_dict(torch.load("models/Model0.1.pt", map_location=torch.device('cpu')))


def transform_image(image):
    image = cv2.resize(image, (320,320))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image ,(2,0,1)).astype(np.float32)
    image = image/255.0 # normalizing original image tensor [0,1] range
    return torch.Tensor(image)

def mask_to_rgb(mask, label=False):
    color_map={
        0: (0, 0, 0), # Background
        1: (255, 0, 0), # Class 1
        2: (0, 255, 0), # Class 2
        3: (0, 0, 255), # Class 3
        4: (0, 128, 128), # Class 4
        5: (128, 0, 128), # Class 5
    }
    color_to_class = {
      (255, 0, 0): "Horse",  # Red pixel represents Horse
      (0, 255, 0): "Bench",  # Green pixel represents Bench
      (0, 0, 255): "Water Dispenser",  # Blue pixel represents Water dispenser
      (0, 128, 128): "Trash bin",  # Blue pixel represents Dust bin
      (128, 0, 128): "Stop Sign",  # Blue pixel represents stop sign
    }
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Count the number of occurrences of each RGB color in the predicted mask.
    counts = {}
    for row in range(320):
        for column in range(320):
            class_index = mask[0][row][column]
            if class_index != 0:
                rgb = color_map[class_index]
                if rgb in counts:
                    counts[rgb] += 1
                else:
                    counts[rgb] = 1
                    
    # Determine the RGB color with the highest count that is not the background color.
    max_count = 0
    max_rgb = None
    for rgb, count in counts.items():
        if count > max_count and rgb != (0,0,0):
            max_count = count
            max_rgb = rgb
    
    # Replace all non-background class indexes in the mask with the chosen RGB color.
    if max_rgb is not None:
        rgb_mask = np.zeros((320, 320, 3), dtype=np.uint8)
        for row in range(320):
            for column in range(320):
                class_index = mask[0][row][column]
                if class_index != 0:
                    rgb = color_map[class_index]
                    if rgb == max_rgb:
                        rgb_mask[row][column] = rgb
                    else:
                        rgb_mask[row][column] = max_rgb
                else:
                    rgb_mask[row][column] = (0,0,0)
    else:
        # If all non-background colors have zero occurrences, use the original function.
        rgb_mask = np.zeros((320, 320, 3), dtype=np.uint8)
        for row in range(320):
            for column in range(320):
                class_index = mask[0][row][column]
                rgb_mask[row][column] = color_map[class_index]
  
    if label:
      return rgb_mask, color_to_class.get(max_rgb)    
    return rgb_mask

def get_prediction(image_bytes):
    import copy
    image = copy.deepcopy(image_bytes)
    tensor = transform_image(image_bytes)
    logits_mask = model(tensor.to('cpu').unsqueeze (0)) #(C, H, W) -> (1, C, H, W)
    pred_mask_prob = torch.softmax(logits_mask, dim=1)  # (batch_size, num_classes, height, width)
    _, pred_mask = torch.max(pred_mask_prob, dim=1)  # (batch_size, height, width)
    rgb_mask, label = mask_to_rgb(pred_mask, label=True)
    alpha_mask = np.where(np.all(rgb_mask == (0, 0, 0), axis=-1), 0, 0.9)
    rgba_mask = np.concatenate([rgb_mask, alpha_mask[..., np.newaxis]], axis=-1)
    plt.imshow(image)
    plt.imshow(rgba_mask, alpha=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='JPEG')
    buf.seek(0)
    overlay_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return rgb_mask, label, overlay_img

@app.route('/')
def home():
    return render_template('index.html')

def convert_image_to_base64(image):
    img_pil = Image.fromarray(image)
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (320, 320))
    mask, label, overlay_img = get_prediction(img)
    mask_string = convert_image_to_base64(mask)
    original_string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')
    return render_template('result.html', image=mask_string, label=label, original_image=original_string, overlay_img=overlay_img)



if __name__ == "__main__":
    app.run()