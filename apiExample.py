#!/bin/python3
from PIL import Image
import requests, base64, io, argparse

SD_WEBUI = 'http://127.0.0.1:7860'

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

with open(args.filename, 'rb') as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

payload = {
    "input_image": base64_image,
    "detection_prompt": "background",
    "positive_prompt": "waterfall",
    "sam_model_name": "sam_hq_vit_h.pth",
    "dino_model_name": "GroundingDINO_SwinB (938MB)",
}

response = requests.post(url=f'{SD_WEBUI}/replacer/replace', json=payload)
if response.status_code == 200:
    response = response.json()
    if response['image']:
        image = Image.open(io.BytesIO(base64.b64decode(response['image']))).show()
else:
    print(response.json())
