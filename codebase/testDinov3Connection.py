from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval()

img = Image.open("data/real/real1.jpeg").convert("RGB")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)           # ModelOutput
    z = out.last_hidden_state[:,0]  # CLS-Token -> (1, 768)
print(z.shape)