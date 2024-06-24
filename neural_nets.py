import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()

chars = sorted(list(set("".join(words))))
chars.insert(0,'.')

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in stoi.items()}

xs, ys = [], []

for w in words[:1]:
    word = ['.'] + list(w) + ['.']
    for char1, char2 in zip(word,word[1:]):
        idx1 = stoi[char1]
        idx2 = stoi[char2]
        xs.append(idx1)
        ys.append(idx2)

xs, ys = torch.tensor(xs), torch.tensor(ys)
print(f'{xs = }')
print(f'{ys = }')

x_enc = F.one_hot(xs,num_classes=27)
print(x_enc.shape)
print(x_enc)

plt.imshow(x_enc)
plt.show()