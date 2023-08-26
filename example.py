import torch
from nevax.model import Neva

#usage
img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

model = Neva()
output = model(img, caption)
print(output.shape) # (1, 1024, 20000)


