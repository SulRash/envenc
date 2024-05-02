from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path

import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

model_path = "bczhou/TinyLLaVA-2.0B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

print(model)

