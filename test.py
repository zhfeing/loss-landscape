import os
import torch


pwd = "run/cifar100/val/vit_mutual"
coord = torch.load(os.path.join(pwd, "coordinates.pth"), map_location="cpu")


x = coord["x_coordinate"]
y = coord["y_coordinate"]

X, Y = torch.meshgrid(x, y)

X = X.flatten()
Y = Y.flatten()


eval = torch.load(os.path.join(pwd, "eval.pth"), map_location="cpu")
acc = eval["acc"]
loss = eval["loss"]
error = 100 * (1 - acc)
# eval = torch.log(eval)
# print(X.tolist(), Y.tolist(), sep="\n")
print(error.flatten().tolist())
