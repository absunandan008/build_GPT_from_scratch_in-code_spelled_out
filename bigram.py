import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 #how many independent sequences will we process in parallel?
block_size = 8 #what is the maximum context length for predictions?
max_iters = 5000
#learning_rate = 1e-2
#decrease learning rate because self attention can't handle very high learning rate
learning_rate = 1e-3
# optimize for gpu or macbook with mps
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu' 
eval_iters = 200
n_embd = 32 #embedding dimension
#----------
torch.manual_seed(1337)
#----------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt","r",encoding='utf-8') as f:
    text = f.read()

#here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# creating mapping for decode and encode
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] #encode takes a string and returns a list of integers
decode  = lambda l: "".join([itos[i] for i in l]) #decode takes a list of integers and returns a string

#train and split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) #first 90% will be train
train_data = data[:n]
val_data = data[n:]

#batching the data
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #randomly generate batch_size starting indices for the sequences
    x = torch.stack([data[i:i+block_size] for i in ix]) #stack the sequences into a tensor of shape (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #the target is the input sequence shifted by one character
    x, y = x.to(device), y.to(device) #move the data to the device
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #set the model back to training mode
    return out

#self attention
class Head(nn.Module):
    """one head of self attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x) #(B,T,C)
        k = self.key(x) #(B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,C) @ (B,C,T) --> (B,T,T)
        #compute attention scores(affinities)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        #perform the weighted aggregation of the values
        v = self.value(x) #(B,T,C)
        out = wei @ v
        return out

#multi head attentions
class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

#feed forward normla MLP
class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        #self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        #instead of using vocab_size, we are going to use embedding dimension of 32, and then we will project it to vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #embedding dimension is 32
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #shape (block_size, n_embd)
        ##self.sa_head = Head(n_embd) #self attention head
        self.sa_heads = MultiHeadAttention(4, n_embd//4) #multihead attantion 4 communication channel, 8 self attention
        #now add linear layer to project the embedding to vocab size
        self.ffwd = FeedForward(n_embd) #feed forward network
        self.lm_head = nn.Linear(n_embd, vocab_size) #project the embedding to vocab size

    def forward(self, idx, targets=None):
        B,T = idx.shape
        #idx and targets are both (B,T) tensors of integers
        #logits = self.token_embedding_table(idx) #(B,T,C) where C is the vocab size
        #now we will use embedding so insted of logits being (B,T,C) it will be (B,T,n_embd)
        tok_emb = self.token_embedding_table(idx) #(B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, n_embd) position embedding for each time step
        x = tok_emb + pos_emb #(B, T, n_embd) add the
        ##x = self.sa_head(x) #(B, T, n_embd) self attention head
        x = self.sa_heads(x) #multihead attention
        x = self.ffwd(x) #feed forward network

        #logits = self.lm_head(tok_emb) #(B, T, vocab_size)
        logits = self.lm_head(x) #(B, T, vocab_size) project the embedding to vocab size

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            #logits, loss = self(idx) #(B,T,C)
            logits, loss = self(idx_cond) #(B,T,C)

            logits = logits[:,-1,:] #(B,C) focus only on the last time step
            probs = F.softmax(logits, dim=-1) #(B,C) get probabilities for the next token
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1) sample the next token from the distribution
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1) append the new token to the context
        return idx

#we are not passing vocab size now because we are using the global variable vocab_size
model = BigramLanguageModel()
m = model.to(device)

#create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)    #training loop
for iter in range(max_iters):
    #every once in a while evaluate the loss on train and val sets
    if iter % 100 == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample a batch of data
    xb, yb = get_batch("train")

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device) #starting with a single token, generate up to 100 new tokens
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
