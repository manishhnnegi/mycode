# final version

# my_model.py
#--------------------------------
import torch
import math
from torch import nn

class PostionalEncoding(nn.Module):
    def __init__(self,d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # create metrix of shape (seq_len , emd_vec_dim)
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # create it 2D
        mid = torch.arange(0, d_model, 2, dtype=torch.float) *( -math.log(10000.0) / d_model)
        div = torch.exp(mid)
        #div =  1/(10000**2i/d_model) -> exp[-2i * log(10000)/d_model]
        # sin(pos * div) or  cos(pos *div)
        #for even nd odd index in emb_vector
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # pe -> shape(seq_len, d_model)
        # change it to -> (1, seq_len, d_model)
        # to add batch dim 
        # its req for adding pe with input emb x
        pe = pe.unsqueeze(0)
        # register to buffer
        self.register_buffer('pe',pe)
    

    def forward(self, x):
        # shape of x-> (batch, seq_len, emd_vec_dim)
        # shape pe -> ( 1, seq_len, emd_vec_dim)
        x = x + self.pe[:, :x.shape[1], :]
        return x
    
class EncoderBlock(nn.Module):
   
    def __init__(self,
                self_attention_block,
                feed_forward_block,
                ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

    def forward(self, x, mask=None):
       x = self.self_attention_block(x, x, x)
       x = self.feed_forward_block(x)
       return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        
    def forward(self, x , mask=None):
        for layer in self.layers:
            x = layer(x)
        return x
      
class MultiHeadAttentionBlock(nn.Module):
   
    def __init__(self,
                d_model:int,
                h:int):
        super().__init__()
        self.d_model = d_model  # embd vec dim
        # no of head -> (division of embed_dim)
        # d_model_part   d_k  = d_model / k
        self.h = h  
        assert d_model % h ==0, {"d_model is not visible by h"}
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
    
    # below is self attention function
    @staticmethod
    def attention(query, key, value, mask = None):
        d_k = query.shape[-1]
        #cal attention 
        #attention_score = (q*k.T / root(model_d)) * v
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        # for decoder self attention
        # it should not see the future 
        # instancess while calculating sefl attention
        if mask is not None:  
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)        
        return attention_scores @ value, attention_scores
    

    def forward(self, q, k, v, mask = None):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)   
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask)
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return x
    
class InputEmbeddings(nn.Module):
  def __init__(self, d_model, vocab_size):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self,x):
    return self.embedding(x) * math.sqrt(self.d_model)

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.self_attention_block(x, x, x, tgt_mask)
        x = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
       
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):
   
    def __init__(self, 
                src_embed: InputEmbeddings, 
                src_pos: PostionalEncoding,
                tgt_embed: InputEmbeddings, 
                tgt_pos: PostionalEncoding,
                encoder: Encoder,
                decoder: Decoder,
                projection_layer:ProjectionLayer,

                ):
        super().__init__()
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
       src = self.src_embed(src)
       src = self.src_pos(src)
       return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, 
               tgt, tgt_mask):       
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size:int,
                      src_seq_len:int,
                      tgt_vocab_size:int,
                      tgt_seq_len:int,
                      d_model:int=512,    # input embedding vector dim
                      d_ff: int = 2048,   # feed forward dim
                      dropout: int = 0.1,
                      h: int = 2,   # no of head
                      N: int = 1,   # no of blocks of Enxoder decoder
                      ):

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    src_pos = PostionalEncoding(d_model, src_seq_len)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    tgt_pos = PostionalEncoding(d_model, tgt_seq_len)

    encoder_blocks = []
    encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)       
    for _ in range(N):
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h)
    decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)        
    for _ in range(N):
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block=feed_forward_block)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(src_embed, src_pos, tgt_embed, tgt_pos, encoder, decoder, projection_layer)
    return transformer


