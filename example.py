import torch
from nevax.model import Neva

#usage
img = torch.randn(1, 3, 256, 256)
caption_tokens = torch.randint(0, 4)

model = Neva()
output = model(img, caption_tokens)