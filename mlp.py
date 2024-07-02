import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F

data = open('names.txt', 'r', encoding='utf-8').read().splitlines()

chars = sorted(list(set("".join(data))))
chars.insert(0,'.')

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

block_size = 3
X, y = [], []

for w in data[:5]:
    print(w)
    context = [0]*block_size
    for ch in w+'.':
        ix = stoi[ch]
        print("".join(itos[x] for x in context), '---->', itos[ix] )
        X.append(context)
        y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
y = torch.tensor(y)

print(X.shape)

g = torch.Generator().manual_seed(2136812323)
C = torch.rand((27,2)) # Initialise randomly in the start
W1 = torch.rand((6,100))
b1 = torch.rand(100)
W2 = torch.rand((100,27))
b2 = torch.rand(27)
parameters = [C, W1, b1, W2, b2]

end_c = F.one_hot(torch.tensor(5),num_classes=27)
end_c = end_c.float() @ C
# emb = C[X] 
# h = torch.tanh(emb.view(emb.shape[0],W1.shape[0]) @ W1 + b1)
# logits = h @ W2 + b2
# # counts = logits.exp()
# # probs = counts/counts.sum(1,keepdim=True)
# # loss = -probs[torch.arange(32), y].log().mean()
# loss = F.cross_entropy(logits,y)
# print(f'{loss = }')

for p in parameters:
    p.requires_grad = True

for _ in range(10):
    emb = C[X] 
    h = torch.tanh(emb.view(emb.shape[0],W1.shape[0]) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits,y)
    print(f'{loss = }')
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -0.1*p.grad
    