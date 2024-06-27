import torch
from matplotlib import pyplot as plt

words = open('names.txt').read().splitlines()

# print(sorted(b.items(), key = lambda kv : kv[1]))

chars = sorted(list(set("".join(words))))
chars.insert(0,'.')

N = torch.zeros((27,27),dtype=torch.int32)

stoi = {s:i for i,s in enumerate(chars)}

# print(stoi)

for w in words:
    word = ['.'] + list(w) + ['.']
    for char1, char2 in zip(word,word[1:]):
        idx1 = stoi[char1]
        idx2 = stoi[char2]
        N[idx1, idx2] += 1

itos = {i:s for s,i in stoi.items()}

# plt.figure(figsize=(32,32))
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i , chstr, ha='center', va='bottom', color= 'gray', fontsize=6)
#         plt.text(j, i , N[i,j].item(), ha='center', va='top', color= 'gray', fontsize=6)
# plt.axis('off')
# plt.imshow(N, cmap='Blues')
# plt.show()

# p = N[0].float()
# p /= p.sum()

# print(p)

g = torch.Generator().manual_seed(2136812323)
# idx = torch.multinomial(p,replacement=True,num_samples=1,generator=g).item()
# print(itos[idx])

P = (N+20).float()
P = P/P.sum(1,keepdims=True)
print(P[0].sum())
for i in range(20):
    idx = 0
    name = ''
    while True:
        p = P[idx]
        # p = N[idx].float()
        # p /= p.sum()
        idx = torch.multinomial(p,replacement=True,num_samples=1,generator=g).item()
        if idx == 0:
            break
        name += itos[idx]

log_likelihood = .0
n = 0
for w in ['fdbfhjbghjb']:
    word = ['.'] + list(w) + ['.']
    for char1, char2 in zip(word,word[1:]):
        idx1 = stoi[char1]
        idx2 = stoi[char2]
        prob = P[idx1,idx2]
        n += 1
        logProb = torch.log(prob)
        log_likelihood += logProb
        # print(f'{char1}{char2} :{prob:.4f} {logProb:.4f}')

print(f'{-log_likelihood=:.4f}')
print(f'{-log_likelihood/n:.4f}')