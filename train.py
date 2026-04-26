#!/usr/bin/env python3.12
# %%
# read it into inspect it
with open("input.txt","r",encoding='utf-8') as f:
    text = f.read()
# %%
#print lenth of dataset
print(f"length of dataset : {len(text)} characters")

# %%
#lets look at the first 1000 characters
print(text[:1000])
# %%
#here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(f"unique characters : {vocab_size}")

# %%
#create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] #encode takes a string and returns a list of integers
decode  = lambda l: "".join([itos[i] for i in l]) #decode takes a list of integers and returns a string

print(encode("hii there"))
print(decode(encode("hii there")))
# %%
#lets encode the entire dataset and store it in a torch tensor
import torch
data = torch.tensor(encode(text),dtype=torch.long)
print(data.shape,data.dtype)
print(data[:1000]) #print the first 1000 encoded characters
# %%
#split the data into train and test sets
n = int(0.9*len(data)) #first 90% will be train
train_data = data[:n]
val_data = data[n:]
# %%
#blocksize is the length of the sequence we will feed into the model
block_size = 8
train_data[:block_size+1] #first 9 characters of the train data
decode(train_data[:block_size+1].tolist()) #decode the first 9 characters of the train data

# %%
x = train_data[:block_size] #input sequence of length block_size
y = train_data[1:block_size+1] #target sequence is the input sequence shifted by one character
#print("input sequence:", decode(x.tolist()))
#print("target sequence:", decode(y.tolist()))
for t in range(block_size):
    context = x[:t+1] #the context is the input sequence up to time step t
    target = y[t] #the target is the character at time step t
    print(f"when input is {context} the target is {target}")
# %%
#batching the data
torch.manual_seed(1337) #set the random seed for reproducibility
batch_size = 4 #number of sequences in a batch
block_size = 8 #length of each sequence, max context length for predictions

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,)) #randomly select 
    x = torch.stack([data[i:i+block_size] for i in ix]) #stack the input sequences into a batch
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix]) #stack the target sequences into a batch
    return x,y

xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

print("--------------")
for b in range(batch_size):
    for t in range(block_size):
        context  = xb[b,:t+1] #the context is the input sequence up to time step t
        target = yb[b,t] #the target is the character at time step t
        print(f"when input is {context} the target is {target}")

# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits  = self.token_embedding_table(idx) # (B,T,C) where C is the vocab size
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) #reshape the logits to be (B*T,C) so that we can compare it to the targets which is (B*T)
            targets = targets.view(-1) #reshape the targets to be (B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self(idx)
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

m = BigramLanguageModel(vocab_size)
out, loss = m(xb, yb)
print(out.shape) #(batch_size, block_size, vocab_size)
print(loss)
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
# %%
#create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# %%
batch_size = 32
max_steps = 10000
for steps in range(max_steps):
    #sample a batch of data
    xb, yb = get_batch("train")

    #evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
# %%
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

# %%
#attention is all you need
torch.manual_seed(1337)
B,T,C = 4,8,2 #batch, time, channels
x = torch.randn(B,T,C) #the shape of x is (B,T,C)
x.shape
# %%
#version 1: using for loops
# we want x[b,t] = mean {x<=t} (x[b,t])
xbow  = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,: t+1] # (t+1, C)
        xbow[b,t] = torch.mean(xprev, 0) # (C,)
print(xbow.shape)
# %%
x[0]

# %%
xbow[0]

# %%
#version 2: using matrix operations
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B,T,T) @ (B,T,C)
torch.allclose(xbow, xbow2)
# %%
#version 3: using softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T,T)
wei = wei.masked_fill(tril ==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
# %%
torch.manual_seed(32)
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a, dim=1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print("a::")
print(a)
print("b::")
print(b)
print("c::")
print(c)
# %%