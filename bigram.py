import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 #how many independent sequences will we process in parallel?
block_size = 8 #what is the maximum context length for predictions?
max_iters = 3000
learning_rate = 1e-2
# optimize for gpu or macbook with mps
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu' 
eval_iters = 200

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) #(B,T,C) where C is the vocab size

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
            logits, loss = self(idx) #(B,T,C)
            logits = logits[:,-1,:] #(B,C) focus only on the last time step
            probs = F.softmax(logits, dim=-1) #(B,C) get probabilities for the next token
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1) sample the next token from the distribution
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1) append the new token to the context
        return idx

model = BigramLanguageModel(vocab_size)
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
