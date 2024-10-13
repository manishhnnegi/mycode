import torch
import torch.nn as nn
from torch.nn import functional as F

#------------------------------------------------------------
#---------------------------------------------------
# hyperparameters:
#-------------------------------------------------------
batch_size = 16
block_size = 64 # max input seq_len(context len)
max_iters = 10000 # max no of steps for training 
eval_interval = 1000 # after how many steps the evaluation will take place
eval_iters = 400  # how many sample of batches will use for evaluation

lr = 3e-4
n_embd = 384 # embed vector dim (channel)
n_head = 6 # 100 / 2 = 3 each (vect div in 2 parts of each 3dims)
n_layer = 6 # no of block used in decoder 
dropout = 0.2

data_path = "hindi.txt" #"input.txt"
# torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

#------------------------------------------------------------
#---------------------------------------------------
# Data Preperation:
#-----------------------------------
#----------------------------------------------

# load dataset:
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# text to numeric ids conversion:
#-------------------------------------
# create vocab:
# vocab: all the unique chars in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"{vocab_size=}")
# create mapping: chars to numeric ids
stoi = { ch:id for id, ch in enumerate(chars) }
itos = { id:ch for id, ch in enumerate(chars) }

# create encoder/decoder:
encode = lambda s : [stoi[c] for c in s] # takes a string output list of integers
decode = lambda l : "".join([itos[i] for i in l]) # takes a list of intgs and output string of chars


# train test split:
#--------------------------------------------------
data = torch.tensor(encode(text), dtype= torch.long) # int64
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading:
#------------------------------------------
# generate a small batches of data of inputs x and outputs y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,)) # random int in b/w(starting indx , lastindx) of batch size(batch))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#------------------------------------------------------------
#---------------------------------------------------
# model building:
#------------------------------------------------------------
#---------------------------------------------------
class Head(nn.Module):
    """ one head self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tri = torch.tril(torch.ones(block_size, block_size))
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', self.tri)

    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, h)
        q = self.query(x) # (B, T, h)
        v = self.value(x) # (B, T, h)

        att = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,h)@(B,h,T) -> (B,T,T)
        att = att.masked_fill(self.tri[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim= -1) # (B,T,T)
        att = self.dropout(att)

        out = att @ v # (B,T,T)@(B,T,h) -> (B,T,h)
        return out

class MultiHeadAtt(nn.Module):

    def __init__(self,num_heads,  head_size):
        super().__init__()
        self.hd = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.hd], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.mhead = MultiHeadAtt(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # x = x + self.mhead(x)   #skip connections
        # x = x + self.ffwd(x)

        x = x + self.mhead(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.net(x)

class GPTFromScratch(nn.Module):

    def __init__(self):
        super().__init__()
        self.embd_table = nn.Embedding(vocab_size, n_embd) # (vocab_size,C)
        self.pos_table = nn.Embedding(block_size, n_embd) # (T,C)
        # self.head = Head(n_embd)
        # self.mhead = MultiHeadAtt(2, n_embd//2)
        # self.ffwd = FeedFoward(n_embd)
        
        self.block = nn.Sequential(
                        Block(n_embd, n_head),
                        Block(n_embd, n_head),
                        Block(n_embd, n_head),
                        Block(n_embd, n_head),
                        )
        
        #self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None):
        B, T = idx.shape  # (B -> batch, T -> block_size(seq_len))
        
        # both xb, yb shape is (B,T) tensor of ints
        tok_emb = self.embd_table(idx) #o/p -> (B,T,C)
        pos_emb = self.pos_table(torch.arange(T, device=device))
        x  = tok_emb + pos_emb # (B,T,C)-> (B,T,C)+ (C,T)
        #x = self.head(x)
        # x = self.mhead(x)
        # x = self.ffwd(x)
        x = self.block(x)
        x = self.ln_f(x) # (B,T,C)

        logits = self.lm_head(x) # (B,T,vocab_size)
             
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_token):
        # idx is (B.T) array
        for _ in range(max_new_token):
            #crop ids to only consider last block_size tokens
            idx_cond =  idx[:, -block_size:] #(B,T)
            # predictions
            logits, loss = self(idx_cond)  #(B,T,C)
            # take only last time step
            logits = logits[:, -1, :]  #(B,-1, C) -> (B, T+1th, C)
            probs = F.softmax(logits, dim = -1)  #(B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            # append sample in the running sequence
            idx = torch.cat((idx,idx_next), dim = 1) #(B, T+1)
        
        return idx  #(B, T+1)

#------------------------------------------------------------
#---------------------------------------------------
# training & Evaluation loop:
#------------------------------------------------------------
#---------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTFromScratch()
m = model.to(device)
# print the no of params in the model
total_params = sum(p.numel() for p in m.parameters())
print(f"The total no of params in the model is {total_params}")

# create torch optimiser:
optimizer = torch.optim.AdamW(model.parameters(), lr= lr)


# training loop:
for iter in range(max_iters):

    # evaluation of Loss on train and val
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # Forward Pass:
    # evaluate loss
    logits, loss = model(xb,yb)

    #print('loss:', loss.item())
    #print(logits.shape)
    # set grad = zero
    optimizer.zero_grad(set_to_none=True)

    # back propagation:
        # grad calculation and param update
    loss.backward()
    optimizer.step()



# Assuming `model` is your trained model
torch.save(model.state_dict(), 'model_weights.pth')

#------------------------------------------------------------
#---------------------------------------------------
# Inferencing from trained model:
#------------------------------------------------------------
#---------------------------------------------------

# generate from the model
context = torch.zeros((1,1), dtype = torch.long, device = device) #shape (B,T)-> (1,1)
print(decode(m.generate(context, max_new_token= 500)[0].tolist()))
open('more.txt', 'w',  encoding='utf-8', errors='ignore').write(decode(m.generate(context, max_new_token=1000)[0].tolist()))