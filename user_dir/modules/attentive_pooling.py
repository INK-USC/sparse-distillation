import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class AttentivePooling(nn.Module):
    def __init__(self, args):
        super(AttentivePooling, self).__init__()
        
        self.pad = args.pad
        self.attn_dim = args.attn_dim
        self.embed_dim = args.encoder_embed_dim

        self.ulinear = nn.Linear(self.embed_dim, self.attn_dim, bias=False)
        self.vlinear = nn.Linear(self.embed_dim, self.attn_dim, bias=False)

        self.tlinear = nn.Linear(self.attn_dim, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()

    def forward(self, x, x_mask, avg_x):
        """
        x : batch_size * seq_len * embed_dim
        avg_x : batch_size * embed_size
        """

        batch_size, seq_len, embed_dim = x.size()

        x_proj = self.ulinear(x) # batch_size * seq_len * attn_dim
        # print(x_proj.size())

        q_proj = self.vlinear(avg_x) # batch_size * attn_dim
        q_proj = q_proj.unsqueeze(1).expand(batch_size, seq_len, self.attn_dim) # batch_size * seq_len * attn_dim
        # print(q_proj.size())

        attn_scores = self.tlinear(torch.tanh(x_proj + q_proj)).squeeze(-1) # batch_size * seq_len
        # print(attn_scores.size())
        attn_scores.data.masked_fill_(x_mask.data, -float("inf"))
        attn_weights = F.softmax(attn_scores, dim=1) # batch_size * seq_len
        outputs = attn_weights.unsqueeze(1).bmm(x).squeeze(1)
        # (batch_size * 1 * seq_len) (bmm)  (batch_size * seq_len * attn_dim) -> (batch_size * 1 * attn_dim) -> (batch_Size * attn_dim)

        return outputs