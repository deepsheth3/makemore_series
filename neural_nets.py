import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()

chars = sorted(list(set("".join(words))))
chars.insert(0,'.')

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}

xs, ys = [], []

for w in words:
    word = ['.'] + list(w) + ['.']
    for char1, char2 in zip(word,word[1:]):
        idx1 = stoi[char1]
        idx2 = stoi[char2]
        xs.append(idx1)
        ys.append(idx2)

xs, ys = torch.tensor(xs), torch.tensor(ys)
# print(f'{xs = }')
# print(f'{ys = }')

x_enc = F.one_hot(xs,num_classes=27).float()
# print(x_enc.shape)
# print(x_enc)

# plt.imshow(x_enc)
# plt.show()

g = torch.Generator().manual_seed(2147483647)
W = torch.rand((27,27),generator=g, requires_grad= True)
logits = x_enc @ W
counts = logits.exp()
probs = counts / counts.sum(1,keepdim=True)

# print(W[:,13])

neg_log_like = torch.zeros(5)
for i in range(5):
    x = xs[i].item()
    y = ys[i].item()
    print('-----')
    print(f'input = {itos[x]}, expected output = {itos[y]}')
    p = probs[i,y]
    print(f'Probability assigned by the neural net to the expected output = {p.item()}')
    logp = torch.log(p)
    print(f'Negative Log Likelyhood = {-logp.item()}')
    neg_log_like[i] = (-logp.item())

print('-----')
print(f'Mean neg_log_like = {neg_log_like.mean().item()}')

# print('--------------Gradient Decent------------------')

nums = xs.nelement()

print(f'{nums = }')

for k in range(100):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(nums), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -50 * W.grad

for i in range(5):
    name = ''
    idx = 0
    while True:
        xenc = F.one_hot(torch.tensor([idx]),num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1,keepdims=True)
        idx = torch.multinomial(p,num_samples=1,replacement=True, generator=g).item()
        if idx == 0:
            break
        name += itos[idx]
    print(f'{name = }')
