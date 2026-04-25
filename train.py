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