import re
import torch

from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

from dataset.sacarsm_dataset import SarcasmDataLoader

'''
    ERROR MIGHT HAPPEN
    >> 'https://discuss.pytorch.org/t/runtimeerror-operator-torchvision-nms-does-not-exist/192829?u=serser'
'''


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(pretrained_name='naver-clova-ix/donut-base-finetuned-rvlcdip'):
    processor = DonutProcessor.from_pretrained(pretrained_name)
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_name)

    return model, processor

def load_image(img_path='Encoder-AnhQuan-DSC_B\\images\\0a00fce16a8b96cd9fd766f3221fb434d585256a9df4922a3a54f4ffcea0f6ff.jpg'):
    img = Image.open(img_path)

    
    return

if __name__=='__main__':
    model, processor = load_model()

    # Load datasets
    train_dataset = SarcasmDataLoader('vimmsd_train.json')
    val_dataset = SarcasmDataLoader('vimmsd_val.json')
    test_dataset = SarcasmDataLoader('vimmsd_test.json')

    image = train_dataset[0]['image']