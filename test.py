import torch
from kain.senses import ImageEncoder, ImageDecoder


x = torch.ones([4, 3, 224, 224])
enet = ImageEncoder([224, 224], [14, 14], 3, 5, 2, 1000, 0.1)
dnet = ImageDecoder([224, 224], [14, 14], 3, 5, 2, 1000, 0.1)
print(x.shape)
x = enet(x)
print(x.shape)
x = dnet(x, x)
print(x.shape)


# enc_net = TextEncoder(['Basic Latin', 'Thai'], 128, 3, 8, 2, 12, 0.1)
# dec_net = TextDecoder(['Basic Latin', 'Thai'], 128, 3, 8, 2, 12, 0.1)
# x = ["สวัสดีครับ ผมชื่อเน็ท"]
# print(x)
# y = enc_net.predict(x)
# print(y)
# x = dec_net.predict(y, y)
# print(x)


