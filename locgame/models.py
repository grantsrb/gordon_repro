import torch
import torch.nn as nn
from torch.nn import ReLU, Tanh
import numpy as np
import time
import os
import torch.nn.functional as F
from transformer.custom_modules import *
from transformer.models import *
from ml_utils.utils import update_shape

d = {i:"cuda:"+str(i) for i in range(torch.cuda.device_count())}
DEVICE_DICT = {-1:"cpu", **d}

N_COLORS = 7
N_SHAPES = 7

class LocatorBase(TransformerBase):
    def __init__(self,obj_recog=False,rew_recog=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.obj_recog = obj_recog
        self.rew_recog = rew_recog

class TransformerLocator(LocatorBase):
    def __init__(self, cnn_type="SimpleCNN", **kwargs):
        super().__init__(**kwargs)
        self.cnn_type = cnn_type
        self.cnn = globals()[self.cnn_type](**kwargs)
        self.pos_encoder = PositionalEncoder(self.cnn.seq_len,
                                             self.emb_size)
        max_num_steps = 20
        self.extractor = Decoder(max_num_steps,
                                 emb_size=self.emb_size,
                                 attn_size=self.attn_size,
                                 n_layers=self.dec_layers,
                                 n_heads=self.n_heads,
                                 act_fxn=self.act_fxn,
                                 use_mask=False,
                                 init_decs=False,
                                 prob_embs=self.prob_embs,
                                 prob_attn=self.prob_attn)

        # Learned initialization for memory transformer init vector
        self.h_init = torch.randn(1,1,self.emb_size)
        divisor = float(np.sqrt(self.emb_size))
        self.h_init = nn.Parameter(self.h_init/divisor)

        self.locator = nn.Sequential(
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.class_h_size),
            globals()[self.act_fxn](),
            nn.LayerNorm(self.class_h_size),
            nn.Linear(self.class_h_size, 2),
            nn.Tanh()
        )
        # Obj recognition model
        if self.obj_recog:
            self.color = nn.Sequential(
                nn.LayerNorm(self.emb_size),
                nn.Linear(self.emb_size, self.class_h_size),
                globals()[self.act_fxn](),
                nn.LayerNorm(self.class_h_size),
                nn.Linear(self.class_h_size, N_COLORS)
            )
            self.shape = nn.Sequential(
                nn.LayerNorm(self.emb_size),
                nn.Linear(self.emb_size, self.class_h_size),
                globals()[self.act_fxn](),
                nn.LayerNorm(self.class_h_size),
                nn.Linear(self.class_h_size, N_SHAPES)
            )

    def fresh_h(self, batch_size=1):
        return self.h_init.repeat(batch_size,1,1)

    def reset_h(self, batch_size=1):
        self.h = self.fresh_h(batch_size)
        return self.h

    def forward(self, x, h=None):
        """
        x: torch float tensor (B,C,H,W)
        """
        if h is None:
            h = self.h
        feats = self.cnn(x)
        feats = self.pos_encoder(feats)
        h = self.extractor(h, feats)
        loc = self.locator(h[:,0])
        if self.obj_recog:
            color = self.color(h[:,0])
            shape = self.shape(h[:,0])
        else:
            color,shape = [],[]
        self.h = torch.cat([self.fresh_h(len(x)),h],axis=1)
        return loc,color,shape

class RNNLocator(LocatorBase):
    def __init__(self, cnn_type="SimpleCNN", rnn_type="GRUCell",
                                             aud_targs=False,
                                             fixed_h=False,
                                             **kwargs):
        """
        cnn_type: str
            the class of cnn to use for creating features from the image
        rnn_type: str
            the class of rnn to use for the temporal model
        aud_targs: bool
            if true, a color and shape must be specified at each step. 
            This creates two separate embeddings that are concatenated
            to the hidden state and projected down into the appropriate
            size for feature extraction and for the rnn. Stands for 
            audible targs
        fixed_h: bool
            if true, the h value is reset at each step in the episode
        """
        super().__init__(**kwargs)
        self.cnn_type = cnn_type
        self.rnn_type = rnn_type
        self.aud_targs = aud_targs
        self.fixed_h = fixed_h
        if self.fixed_h: print("USING FIXED H VECTOR!!")
        self.cnn = globals()[self.cnn_type](**kwargs)
        self.pos_encoder = PositionalEncoder(self.cnn.seq_len,
                                             self.emb_size)
        self.extractor = Attncoder(1, emb_size=self.emb_size,
                                 attn_size=self.attn_size,
                                 n_layers=self.dec_layers,
                                 n_heads=self.n_heads,
                                 act_fxn=self.act_fxn,
                                 use_mask=False,
                                 init_decs=False,
                                 gen_decs=False,
                                 prob_embs=self.prob_embs,
                                 prob_attn=self.prob_attn)

        # Learned initialization for rnn hidden vector
        self.h_shape = (1,self.emb_size)
        self.h_init = torch.randn(self.h_shape)
        divisor = float(np.sqrt(self.emb_size))
        self.h_init = nn.Parameter(self.h_init/divisor)

        if self.aud_targs:
            self.color_embs = nn.Embedding(N_COLORS, self.emb_size)
            self.shape_embs = nn.Embedding(N_SHAPES, self.emb_size)
            self.aud_projection = nn.Linear(3*self.emb_size,
                                            self.emb_size)

        self.rnn = getattr(nn,self.rnn_type)(input_size=self.emb_size,
                                             hidden_size=self.emb_size)

        self.locator = nn.Sequential(
            #nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.class_h_size),
            globals()[self.act_fxn](),
            #nn.LayerNorm(self.class_h_size),
            nn.Linear(self.class_h_size, 2),
            nn.Tanh()
        )
        # Reward model
        self.pavlov = nn.Sequential(
            #nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.class_h_size),
            globals()[self.act_fxn](),
            #nn.LayerNorm(self.class_h_size),
            nn.Linear(self.class_h_size, 1)
        )
        # Obj recognition model
        if self.obj_recog and not self.aud_targs:
            self.color = nn.Sequential(
                nn.LayerNorm(self.emb_size),
                nn.Linear(self.emb_size, self.class_h_size),
                globals()[self.act_fxn](),
                nn.LayerNorm(self.class_h_size),
                nn.Linear(self.class_h_size, N_COLORS)
            )
            self.shape = nn.Sequential(
                nn.LayerNorm(self.emb_size),
                nn.Linear(self.emb_size, self.class_h_size),
                globals()[self.act_fxn](),
                nn.LayerNorm(self.class_h_size),
                nn.Linear(self.class_h_size, N_SHAPES)
            )

    def reset_h(self, batch_size=1):
        """
        returns an h that is of shape (B,E)
        """
        self.h = self.h_init.repeat(batch_size,1)
        return self.h

    def forward(self, x, h=None, color_idx=None, shape_idx=None):
        """
        x: torch float tensor (B,C,H,W)
        h: optional float tensor (B,E)
        color_idx: long tensor (B,1)
        shape_idx: long tensor (B,1)
        """
        if h is None:
            h = self.h
        if self.aud_targs: # Create new h if conditional predictions
            color_emb=self.color_embs(color_idx)
            shape_emb=self.shape_embs(shape_idx)
            h = torch.cat([h, color_emb.reshape(-1,self.emb_size),
                              shape_emb.reshape(-1,self.emb_size)],
                              axis=-1)
            h = self.aud_projection(h)
        feats = self.cnn(x)
        feats = self.pos_encoder(feats)
        if self.fixed_h:
            h = self.reset_h(len(x))
        feat = self.extractor(h.unsqueeze(1), feats)
        h = self.rnn(feat.mean(1),h)
        loc = self.locator(h)
        if self.obj_recog and not self.aud_targs:
            color = self.color(h)
            shape = self.shape(h)
        else:
            color,shape = [],[]
        if self.rew_recog:
            rew = self.pavlov(h)
        else:
            rew = []
        self.h = h
        return loc,color,shape,rew

class PooledRNNLocator(RNNLocator):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pos_encoder = NullOp()
        self.extractor = Pooler(self.cnn.shapes[-1],
                                emb_size=self.emb_size,
                                ksize=5)
        print("Using PooledRNNLocator")

class ConcatRNNLocator(RNNLocator):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pos_encoder = NullOp()
        self.extractor = Concatenater(self.cnn.shapes[-1],
                                      emb_size=self.emb_size,
                                      ksize=5)
        print("Using ConcatRNNLocator")

class CNNBase(nn.Module):
    def __init__(self, img_shape=(3,84,84), act_fxn="ReLU",
                                            emb_size=512,
                                            attn_size=64,
                                            n_heads=6,
                                            **kwargs):
        """
        img_shape: tuple of ints (chan, height, width)
            the incoming image size
        act_fxn: str
            the name of the desired activation function
        emb_size: int
            the size of the "embedding" layer which is just the size
            of the final output channel dimension
        attn_size: int
            the size of the attentional features for the multi-head
            attention mechanism
        n_heads: int
            the number of attention heads in the multi-head attention
        """
        super().__init__()
        self.img_shape = img_shape
        self.act_fxn = act_fxn
        self.emb_size = emb_size
        self.attn_size = attn_size
        self.n_heads = n_heads

    def get_conv_block(self, in_chan, out_chan, ksize=3, 
                                          stride=1,
                                          padding=0,
                                          bnorm=True,
                                          act_fxn="ReLU",
                                          drop_p=0):
        """
        returns a set of operations that make up a single layer in
        a convolutional neural network

        in_chan: int
            the number of incoming channels
        out_chan: int
            the number of outgoing channels
        ksize: int
            the size of the convolutional kernel
        stride: int
            the stride length of the convolutional kernel
        padding: int
            the padding of the convolution
        bnorm: bool
            determines if batch normalization should be used
        act_fxn: str
            the name of the activation function
        drop_p: float [0,1]
            the probability of setting an activation to 0
        """
        block = []
        block.append(nn.Conv2d(in_chan,out_chan,ksize,
                                                stride=stride,
                                                padding=padding))
        if bnorm:
            block.append(nn.BatchNorm2d(out_chan))
        if act_fxn is not None:
            block.append(globals()[act_fxn]())
        if drop_p > 0:
            block.append(nn.Dropout(drop_p))
        return block

class SimpleCNN(CNNBase):
    """
    Simple model
    """
    def __init__(self, emb_size, intm_attn=0, **kwargs):
        """
        emb_size: int
        intm_attn: int
            an integer indicating the number of layers for an attention
            layer in between convolutions
        """
        super().__init__(**kwargs)
        self.emb_size = emb_size
        self.intm_attn = intm_attn
        self.conv_blocks = nn.ModuleList([])
        self.intm_attns = nn.ModuleList([])
        self.shapes = []
        shape = self.img_shape[-2:]
        self.shapes.append(shape)
        if self.img_shape[1] <= 84:
            chans = [32,64,128,256,self.emb_size]
            stride = 1
            ksize = 3
        else:
            chans = [3,32,64,128,256,self.emb_size]
            stride = 2
            ksize = 5
            print("using extra layer for larger image size")
        self.chans = chans
        padding = 0
        block = self.get_conv_block(in_chan=self.img_shape[-3],
                                    out_chan=self.chans[0],
                                    ksize=ksize,
                                    stride=stride,
                                    padding=padding,
                                    bnorm=True,
                                    act_fxn=self.act_fxn,
                                    drop_p=0)
        self.conv_blocks.append(nn.Sequential(*block))
        shape = update_shape(shape, kernel=ksize, stride=stride,
                                                  padding=padding)
        self.shapes.append(shape)
        if self.intm_attn > 0:
            attn = ConvAttention(chans[0], shape,
                                           n_layers=self.intm_attn,
                                           attn_size=self.attn_size,
                                           act_fxn=self.act_fxn)
            self.itmd_attns.append(attn)
        for i in range(len(chans)-1):
            if i in {1,3}: stride = 2
            else: stride = 1
            block = self.get_conv_block(in_chan=chans[i],
                                        out_chan=chans[i+1],
                                        ksize=ksize,
                                        stride=stride,
                                        padding=padding,
                                        bnorm=True,
                                        act_fxn=self.act_fxn,
                                        drop_p=0)
            self.conv_blocks.append(nn.Sequential(*block))
            shape = update_shape(shape, kernel=ksize, stride=stride,
                                                      padding=padding)
            self.shapes.append(shape)
            print("model shape {}: {}".format(i,shape))
            if self.intm_attn > 0:
                attn = ConvAttention(chans[0], shape,
                                               n_layers=self.intm_attn,
                                               attn_size=self.attn_size,
                                               act_fxn=self.act_fxn)
                self.itmd_attns.append(attn)
        self.seq_len = shape[0]*shape[1]

    def forward(self, x, *args, **kwargs):
        """
        x: float tensor (B,C,H,W)
        """
        fx = x
        for i,block in enumerate(self.conv_blocks):
            fx = block(fx)
            if i < len(self.intm_attns):
                fx = self.intm_attns(fx)
        return fx.reshape(fx.shape[0],fx.shape[1],-1).permute(0,2,1)

class MediumCNN(CNNBase):
    """
    Middle complexity model
    """
    def __init__(self, emb_size, intm_attn=0, **kwargs):
        """
        emb_size: int
        intm_attn: int
            an integer indicating the number of layers for an attention
            layer in between convolutions
        """
        super().__init__(**kwargs)
        self.emb_size = emb_size
        self.intm_attn = intm_attn
        self.conv_blocks = nn.ModuleList([])
        self.intm_attns = nn.ModuleList([])
        self.shapes = []
        shape = self.img_shape[-2:]
        self.shapes.append(shape)
        chans = [8,32,64,128,256,self.emb_size]
        stride = 2
        ksize = 7
        self.chans = chans
        padding = 0
        block = self.get_conv_block(in_chan=self.img_shape[-3],
                                    out_chan=self.chans[0],
                                    ksize=ksize,
                                    stride=stride,
                                    padding=padding,
                                    bnorm=True,
                                    act_fxn=self.act_fxn,
                                    drop_p=0)
        self.conv_blocks.append(nn.Sequential(*block))
        shape = update_shape(shape, kernel=ksize, stride=stride,
                                                  padding=padding)
        self.shapes.append(shape)
        if self.intm_attn > 0:
            attn = ConvAttention(chans[0], shape,
                                           n_layers=self.intm_attn,
                                           attn_size=self.attn_size,
                                           act_fxn=self.act_fxn)
            self.itmd_attns.append(attn)

        ksize = 3
        for i in range(len(chans)-1):
            if i in {1,3}: stride = 2
            else: stride = 1
            block = self.get_conv_block(in_chan=chans[i],
                                        out_chan=chans[i+1],
                                        ksize=ksize,
                                        stride=stride,
                                        padding=padding,
                                        bnorm=True,
                                        act_fxn=self.act_fxn,
                                        drop_p=0)
            self.conv_blocks.append(nn.Sequential(*block))
            shape = update_shape(shape, kernel=ksize, stride=stride,
                                                      padding=padding)
            self.shapes.append(shape)
            print("model shape {}: {}".format(i,shape))
            if self.intm_attn > 0:
                attn = ConvAttention(chans[0], shape,
                                               n_layers=self.intm_attn,
                                               attn_size=self.attn_size,
                                               act_fxn=self.act_fxn)
                self.itmd_attns.append(attn)
        self.seq_len = shape[0]*shape[1]

    def forward(self, x, *args, **kwargs):
        """
        x: float tensor (B,C,H,W)
        """
        fx = x
        for i,block in enumerate(self.conv_blocks):
            fx = block(fx)
            if i < len(self.intm_attns):
                fx = self.intm_attns(fx)
        return fx.reshape(fx.shape[0],fx.shape[1],-1).permute(0,2,1)

def deconv_block(in_depth, out_depth, ksize=3, stride=1,
                                               padding=0,
                                               bnorm=False,
                                               act_fxn='ReLU',
                                               drop_p=0):
    """
    Creates a deconvolution layer

    in_depth: int
    out_depth: int
    ksize: int
    stride: int
    padding: int
    bnorm: bool
        determines if a batchnorm layer should be inserted just after
        the deconvolution
    act_fxn: str
        the name of the activation class
    drop_p: float
        the probability of an activation being dropped
    """
    block = []
    block.append(nn.ConvTranspose2d(in_depth, out_depth,
                                              ksize,
                                              stride=stride,
                                              padding=padding))
    if bnorm:
        block.append(nn.BatchNorm2d(out_depth))
    block.append(nn.Dropout(drop_p))
    if activation is not None:
        block.append(getattr(nn, act_fxn))
    return nn.Sequential(*block)

class SimpleDeconv(nn.Module):
    def __init__(self, emb_shape, img_shape, h_size, s_size,
                                                     bnorm=True,
                                                     noise=0):
        """
        emb_shape - list like (C, H, W)
            the initial shape to reshape the embedding inputs
            (can take from encoder.emb_shape)
        img_shape - list like (C, H, W)
            the final shape of the decoded tensor
        h_size - int
            size of belief vector h
        bnorm - bool
            optional, if true, model uses batchnorm
        noise - float
            standard deviation of gaussian noise added at each layer
        """
        super(Decoder, self).__init__()
        self.emb_shape = emb_shape
        self.img_shape = img_shape
        self.h_size = h_size
        self.noise = noise
        self.bnorm = bnorm

        depth, height, width = emb_shape
        first_ksize = 9
        ksize = 3
        padding = 0
        modules = []
        self.sizes = []
        modules.append(Reshape((-1, depth, height, width)))
        deconv = deconv_block(depth, depth, ksize=ksize,stride=1,
                                                       padding=0,
                                                       bnorm=self.bnorm,
                                                       noise=self.noise)
        height, width = update_shape((height,width), kernel=ksize,
                                                      op="deconv")
        self.sizes.append((height, width))
        modules.append(deconv)

        while height < self.img_shape[-2] and width < self.img_shape[-1]:
            stride = 2 if i % 3 == 0 else 1
            modules.append(deconv_block(depth, depth, ksize=ksize,
                                        padding=padding, stride=stride,
                                        bnorm=self.bnorm, noise=noise))
            height, width = update_shape((height,width), kernel=ksize,
                                                         stride=stride,
                                                         op="deconv")
            self.sizes.append((height, width))
            print("h:", height, "| w:", width)
        
        diff = height-self.img_shape[-2]
        modules.append(nn.Conv2d(depth,self.img_shape[0],diff+1))
        height, width = update_shape((height,width), kernel=3)
        print("decoder:", height, width)
        self.sizes.append((height, width))
        
        self.sequential = nn.Sequential(*modules)
        emb_size = int(np.prod(emb_shape))
        self.resize = nn.Sequential(nn.Linear(h_size, emb_size),
                                            Reshape((-1, *emb_shape)))

    def forward(self, x):
        """
        x - torch FloatTensor
            should be h and s concatenated
        """
        emb = self.resize(x)
        return self.sequential(emb)

    def extra_repr(self):
        s = "emb_shape={}, img_shape={}, bnorm={}, noise={}"
        return s.format(self.emb_shape, self.img_shape, self.bnorm,
                                                        self.noise)

class Pooler(nn.Module):
    """
    A simple class to act as a dummy extractor that actually performs
    a final convolution followed by a global average pooling
    """
    def __init__(self, shape, emb_size=512, ksize=5):
        """
        shape: tuple of ints (H,W)
        emb_size: int
        ksize: int
        """
        super().__init__()
        self.emb_size = emb_size
        self.ksize = ksize
        self.shape = shape
        self.conv = nn.Conv2d(self.emb_size, self.emb_size, self.ksize)
        self.activ = nn.ReLU()
        self.layer = nn.Sequential( self.conv, self.activ)

    def forward(self, h, x):
        """
        h: dummy
        x: torch FloatTensor (B,S,E)
            the features from the cnn
        """
        shape = (len(x), self.emb_size, self.shape[0], self.shape[1])
        x = x.permute(0,2,1).reshape(shape)
        fx = self.layer(x)
        return fx.reshape(*shape[:2],-1).mean(-1).unsqueeze(1) # (B,1,E)

class Concatenater(nn.Module):
    """
    A simple class to act as a dummy extractor that actually performs
    a another convolution followed by a feature concatenation and 
    nonlinear projection to a single feature vector.
    """
    def __init__(self, shape, emb_size=512, ksize=5, h_size=1000):
        """
        shape: tuple of ints (H,W)
        emb_size: int
        ksize: int
        """
        super().__init__()
        self.emb_size = emb_size
        self.ksize = ksize
        self.shape = shape
        self.h_size = h_size
        self.conv = nn.Conv2d(self.emb_size, self.emb_size, self.ksize)
        self.activ = nn.ReLU()
        self.layer = nn.Sequential( self.conv, self.activ)
        new_shape = update_shape(self.shape, kernel=ksize,
                                             stride=1,
                                             padding=0)
        flat_size = new_shape[-2]*new_shape[-1]*self.emb_size
        self.collapser = nn.Sequential(
                    nn.Linear(flat_size,self.h_size),
                    nn.ReLU(),
                    nn.Linear(self.h_size, self.emb_size)
                    )


    def forward(self, h, x):
        """
        h: dummy
        x: torch FloatTensor (B,S,E)
            the features from the cnn
        """
        shape = (len(x), self.emb_size, self.shape[0], self.shape[1])
        x = x.permute(0,2,1).reshape(shape)
        fx = self.layer(x).reshape(len(x), -1)
        fx = self.collapser(fx)
        return fx.reshape(len(x),1,self.emb_size) # (B,1,E)

class NullOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Reshape(nn.Module):
    """
    Reshapes the activations to be of shape (B, *shape) where B
    is the batch size.
    """
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

    def extra_repr(self):
        return "shape={}".format(self.shape)

