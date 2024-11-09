import os
import json

from PIL import Image
from datasets import Dataset

class SarcasmDataLoader:
    def __init__(self, annotation='annotations', file_annotation='vimmsd_train.json', folder_image='images'):
        self.data = []

        # Path of folder images and file annotations
        self.file_json = os.path.join('.', annotation, file_annotation)
        self.folder_image = os.path.join('.', folder_image)
        
        lable2id={
            "image-sarcasm": 0,
            "text-sarcasm": 1,
            "multi-sarcasm": 2,
            "not-sarcasm": 3,
            None: -1,
        }

        with open(self.file_json, 'r', encoding='utf-8') as file:
            data_json = json.load(file)
        for idx, value in data_json.items():
            image_path = os.path.join(self.file_image, value['image'])
            try:
                img = Image.open(image_path).convert("RGB")
            except:
                img = None
                
            if img is None:
                continue
            
            try:
                label = lable2id[value['label']]
            except:
                label = -1
                
            # Store data as paths or preprocessed values
            self.data.append({
                'id': idx,
                'image': img,
                'text': value['caption'],
                'label': label,
            })

    def to_hf_dataset(self):
        # Convert the list of dictionaries to Hugging Face Dataset format
        hf_data = {
            'id': [item['id'] for item in self.data],
            'image': [item['image'] for item in self.data],
            'text': [item['text'] for item in self.data],
            'label': [item['label'] for item in self.data]
        }
        return Dataset.from_dict(hf_data)