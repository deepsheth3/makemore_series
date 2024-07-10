import random
import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F

data = open('names.txt', 'r', encoding='utf-8').read().splitlines()
chars = sorted(list(set("".join(data))))
chars.insert(0,'.')

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}


block_size = 3
def build_dataset(words):
    X, y = [], []
    for w in words:
        context = [0]*block_size
        for ch in w+'.':
            ix = stoi[ch]
            X.append(context)
            y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    y = torch.tensor(y)
    return X, y

random.seed(42)
random.shuffle(data)
n1 = int(0.8 * len(data))
n2 = int(0.9 * len(data))

X_train, y_train = build_dataset(data[:n1])
X_test, y_test = build_dataset(data[n1:n2])
X_val, y_val = build_dataset(data[n2:])

print(f'X_train:{X_train.shape} y_train: {y_train.shape}')

    
g = torch.Generator().manual_seed(2136812323)
C = torch.rand((27,3)) # Initialise randomly in the start
W1 = torch.rand((9,300))
b1 = torch.rand(300)
W2 = torch.rand((300,27))
b2 = torch.rand(27)
parameters = [C, W1, b1, W2, b2]

end_c = F.one_hot(torch.tensor(5),num_classes=27)
end_c = end_c.float() @ C

for p in parameters:
    p.requires_grad = True

loss_i = []
step = []

for i in range(50000):
    ix = torch.randint(0,X_train.shape[0],(128,))
    emb = C[X_train[ix]] 
    h = torch.tanh(emb.view(emb.shape[0],W1.shape[0]) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits,y_train[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -0.01*p.grad
    step.append(i)
    loss_i.append(loss.log10().item())
print(f'loss: {loss.item()}')

emb = C[X_test] 
h = torch.tanh(emb.view(emb.shape[0],W1.shape[0]) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,y_test)
print(f'loss: {loss.item()}')

plt.plot(step,loss_i)
plt.show()

for i in range(20):
    name = ''
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(emb.shape[0],W1.shape[0]) @ W1 + b1)
        logits = h @ W2 + b2
        prob = F.softmax(logits,dim=1)
        ix = torch.multinomial(prob,num_samples=1,replacement=True, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        name += itos[ix]
    print(f'{name = }')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(C[:,0].data, C[:,1].data, C[:,2].data,s=300)
for i in range(C.shape[0]):
    ax.text(C[i,0].item(), C[i,1].item(),C[i,2].item(), itos[i],ha='center',va='center',color='white')
plt.show()