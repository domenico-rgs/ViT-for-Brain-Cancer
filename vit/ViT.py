import torch
import torch.nn as nn

import TBlock as tb

'''
Weights initialization function.
If the layer is a linear layer, the weights are initialized using a truncated normal distribution with a standard deviation of 0.02. Eventually the bias is set to 0.
If the layer is a layer normalization layer, the weights are initialized to 1 and the bias to 0.
'''
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        #nn.init.constant_(m.weight, 1.0) #DEBUG
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

'''
Vision Transformer for hyperspectral brain images classification/segmentation, by default the images have a size of 128x128
and a number of bands equal to 25. The classes are [tumor, vein, artery, normal, duraMater].
'''
class HyBrainViT (nn.Module):
    def __init__(self, patchSize, nBlocks, mlp_dim, numHeads, mode=None, embedDim=64, numClasses=4, dropout=0.1, channels=25):
        super().__init__()

        self.patchSize = patchSize
        self.nBlocks = nBlocks
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.numClasses = numClasses

        self.patchEmbed = nn.Linear(patchSize, embedDim) #input: [B channels 1], output: [B channels embedDim] -> [B 25 64], one embedding per band, to be rearranged

        #Learned class token and position embeddings
        self.clsToken = nn.Parameter(torch.zeros(1, 1, embedDim))
        self.posEmbed = nn.Parameter(torch.zeros(1, channels+1, embedDim))

        # Initialization of the additional embeddings using a truncated normal distribution
        nn.init.trunc_normal_(self.clsToken.data, std=0.02)
        nn.init.trunc_normal_(self.posEmbed.data, std=0.02)

        self.dropout = nn.Dropout(dropout)

        #Set of block transformers
        self.blocks = nn.ModuleList([tb.TBlock(embedDim, numHeads, mlp_dim, dropout) for i in range(nBlocks)])
        
        #CAF
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(nBlocks-2):
            self.skipcat.append(nn.Conv2d(channels+1, channels+1, [1, 2], 1, 0))

        self.norm = nn.LayerNorm(embedDim)
        self.mlpHead = nn.Linear(embedDim, numClasses)

        self.apply(init_weights)

    def forward(self, px):
        B, _, _ = px.shape

        #Patch embedding
        x = self.patchEmbed(px)

        #Add clsToken
        clsToken = self.clsToken.expand(B, -1, -1)
        x = torch.cat((clsToken, x), dim=1)

        #Add positional embeddings
        posEmbed = self.posEmbed
        x = x + posEmbed
        x = self.dropout(x)

        #Transformer blocks (+CAF), adapted from https://arxiv.org/pdf/2107.02988.pdf
        if self.mode != 'CAF':
            for blk in self.blocks:
                x = blk(x)
        else:
            last_output = []
            nb = 0
            for blk in self.blocks:           
                last_output.append(x)
                if nb > 1:  
                    x = self.skipcat[nb-2](torch.cat([x.unsqueeze(3), last_output[nb-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = blk(x)
                nb += 1

        x = x[:, 0] # get clsToken
        
        x = self.norm(x)
        x = self.mlpHead(x) #return the class logits for the pixel

        return x
