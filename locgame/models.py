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

class CustomModule:
    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except:
            return False

    def get_device(self):
        try:
            return next(self.parameters()).get_device()
        except:
            return -1

class RSSM(nn.Module, CustomModule):
    def __init__(self, h_size, s_size, a_size, rnn_type="GRU",
                                               min_sigma=0.0001):
        super(RSSM, self).__init__()
        """
        h_size - int
            size of belief vector h
        s_size - int
            size of state vector s
        a_size - int
            size of action space vector a

        min_sigma - float
            the minimum value that the state standard deviation can take
        """
        if rnn_type == "GRU":
            rnn_type = "GRUCell"
        assert rnn_type == "GRUCell" # Only supported type currently
        self.h_size = h_size
        self.s_size = s_size
        self.a_size = a_size
        self.rnn_type = rnn_type
        self.min_sigma = min_sigma

        self.rnn = getattr(nn, rnn_type)(input_size=(s_size+a_size),
                                    hidden_size=h_size) # Dynamics rnn
        # Creates mu and sigma for state gaussian
        self.state_layer = nn.Linear(h_size, 2*s_size) 

    def forward(self, h, s, a):
        x = torch.cat([s,a], dim=-1)
        h_new = self.rnn(x, h)
        musigma = self.state_layer(h_new)
        mu, sigma = torch.chunk(musigma, 2, dim=-1)
        sigma = F.softplus(sigma) + self.min_sigma
        return h_new, mu, sigma
    
    def extra_repr(self):
        s = "h_size={}, s_size={}, a_size={}, n_layers={}, min_sigma={}"
        return s.format(self.h_size, self.s_size, self.a_size,
                                self.n_layers, self.min_sigma)


class LocatorBase(TransformerBase, CustomModule):
    def __init__(self,obj_recog=False,rew_recog=False,
                                      n_numbers=7,
                                      n_colors=7,
                                      n_shapes=7,
                                      *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.obj_recog = obj_recog
        self.rew_recog = rew_recog
        self.n_numbers = n_numbers
        self.n_colors = n_colors
        self.n_shapes = n_shapes

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
                nn.Linear(self.class_h_size, self.n_colors)
            )
            self.shape = nn.Sequential(
                nn.LayerNorm(self.emb_size),
                nn.Linear(self.emb_size, self.class_h_size),
                globals()[self.act_fxn](),
                nn.LayerNorm(self.class_h_size),
                nn.Linear(self.class_h_size, self.n_shapes)
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
                                             countOut=0,
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
        countOut: bool or int
            if true, then counting out the number of objects is part of
            the process
        """
        super().__init__(**kwargs)
        self.cnn_type = cnn_type
        self.rnn_type = rnn_type
        self.aud_targs = aud_targs
        self.fixed_h = fixed_h
        self.count_out = countOut
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

        emb_count = 1
        if self.count_out:
            emb_count += 1
            self.number_embs =nn.Embedding(self.n_numbers,self.emb_size)
        if self.aud_targs:
            emb_count += 2
            self.color_embs = nn.Embedding(self.n_colors,self.emb_size)
            self.shape_embs = nn.Embedding(self.n_shapes,self.emb_size)
        if emb_count > 1:
            self.aud_projection = nn.Linear(emb_count*self.emb_size,
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
                nn.Linear(self.class_h_size, self.n_colors)
            )
            self.shape = nn.Sequential(
                nn.LayerNorm(self.emb_size),
                nn.Linear(self.emb_size, self.class_h_size),
                globals()[self.act_fxn](),
                nn.LayerNorm(self.class_h_size),
                nn.Linear(self.class_h_size, self.n_shapes)
            )

    def reset_h(self, batch_size=1):
        """
        returns an h that is of shape (B,E)
        """
        self.h = self.h_init.repeat(batch_size,1)
        return self.h

    def forward(self, x, h=None, color_idx=None,
                                 shape_idx=None,
                                 number_idx=None):
        """
        x: torch float tensor (B,C,H,W)
        h: optional float tensor (B,E)
        color_idx: long tensor (B,1)
        shape_idx: long tensor (B,1)
        number_idx: long tensor (B,1)
        """
        if h is None:
            h = self.h
        if self.fixed_h:
            h = self.reset_h(len(x))
        cat_arr = [h]
        if self.count_out:
            number_emb = self.number_embs(number_idx)
            cat_arr.append(number_emb.reshape(-1,self.emb_size))
        if self.aud_targs: # Create new h if conditional predictions
            color_emb = self.color_embs(color_idx)
            cat_arr.append(color_emb.reshape(-1,self.emb_size))
            shape_emb = self.shape_embs(shape_idx)
            cat_arr.append(shape_emb.reshape(-1,self.emb_size))
        if len(cat_arr)>1:
            h = torch.cat(cat_arr, axis=-1)
            h = self.aud_projection(h)
        feats = self.cnn(x)
        feats = self.pos_encoder(feats)
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

class CNNBase(nn.Module, CustomModule):
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

class RNNFwdDynamics(LocatorBase):
    """
    This model takes in the current observation and makes a prediction
    of the next state of the game. A separate Decoding model is used
    to constrain these states off of the pixels of the game.
    """
    def __init__(self, deconv_type="SimpleDeconv", rnn_type="GRUCell",
                                                   cnn_type="SimpleCNN",
                                                   aud_targs=False,
                                                   fixed_h=False,
                                                   **kwargs):
        """
        cnn_type: str
            the class of cnn to use for extracting features from the
            image
        deconv_type: str
            the class of deconv to use for creating features from the
            image
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
        self.deconv_type = deconv_type
        self.rnn_type = rnn_type
        self.aud_targs = aud_targs
        self.fixed_h = fixed_h
        if self.fixed_h: print("USING FIXED H VECTOR!!")
        self.cnn = globals()[self.cnn_type](**kwargs)
        self.deconv = globals()[self.deconv_type](**kwargs)
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
        self.number_embs = nn.Embedding(self.n_numbers, self.emb_size)
        emb_count = 2
        if self.aud_targs:
            self.color_embs = nn.Embedding(self.n_colors, self.emb_size)
            self.shape_embs = nn.Embedding(self.n_shapes, self.emb_size)
            emb_count = 4
        self.h_projection = nn.Linear(emb_count*self.emb_size,
                                      self.emb_size)

        # Learned initialization for rnn hidden vector
        self.h_shape = (1,self.emb_size)
        self.h_init = torch.randn(self.h_shape)
        divisor = float(np.sqrt(self.emb_size))
        self.h_init = nn.Parameter(self.h_init/divisor)

        self.rnn = getattr(nn,self.rnn_type)(input_size=self.emb_size,
                                             hidden_size=self.emb_size)

    def reset_h(self, batch_size=1):
        """
        returns an h that is of shape (B,E)
        """
        self.h = self.h_init.repeat(batch_size,1)
        return self.h

    def forward(self, x, h=None, color_idx=None,
                                 shape_idx=None,
                                 number_idx=None):
        """
        x: torch float tensor (B,C,H,W)
        h: optional float tensor (B,E)
        color_idx: long tensor (B,1)
        shape_idx: long tensor (B,1)
        number_idx: long tensor (B,1)
        """
        assert number_idx is not None, "Must have number index"
        if h is None:
            h = self.h
        if self.fixed_h:
            h = self.reset_h(len(x))
        feats = self.cnn(x)
        feats = self.pos_encoder(feats)
        number_emb = self.number_embs(number_idx)
        cat_arr = [h,number_emb.reshape(-1,self.emb_size)]
        if self.aud_targs: # Create new h if conditional predictions
            color_emb=self.color_embs(color_idx)
            cat_arr.append(color_emb.reshape(-1,self.emb_size))
            shape_emb=self.shape_embs(shape_idx)
            cat_arr.append(shape_emb.reshape(-1,self.emb_size))
        h = torch.cat(cat_arr, axis=-1)
        h = self.h_projection(h)
        feat = self.extractor(h.unsqueeze(1), feats)
        h = self.rnn(feat.mean(1),h)
        self.h = h
        pred = self.deconv(h)
        return torch.sigmoid(pred) # (B,C,H,W)

class PooledRNNFwdDynamics(RNNFwdDynamics):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pos_encoder = NullOp()
        self.extractor = Pooler(self.cnn.shapes[-1],
                                emb_size=self.emb_size,
                                ksize=5)
        print("Using PooledRNNFwdDynamics")

class ConcatRNNFwdDynamics(RNNFwdDynamics):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pos_encoder = NullOp()
        self.extractor = Concatenater(self.cnn.shapes[-1],
                                      emb_size=self.emb_size,
                                      ksize=5)
        print("Using ConcatRNNFwdDynamics")

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
    if act_fxn is not None:
        block.append(getattr(nn, act_fxn)())
    return nn.Sequential(*block)


class SimpleDeconv(nn.Module):
    """
    This model is used to make observation predictions. It takes in
    a single state vector and transforms it to an image (C,H,W)
    """
    def __init__(self, emb_size, img_shape, start_shape=(512,7,7),
                                            deconv_bnorm=False,
                                            drop_p=0,
                                            **kwargs):
        """
        start_shape - list like [channel1, height1, width1]
            the initial shape to reshape the embedding inputs
        img_shape - list like [channel2, height2, width2]
            the final shape of the decoded tensor
        emb_size - int
            size of belief vector h
        deconv_bnorm: bool
            determines if batchnorm will be used
        drop_p - float
            dropout probability at each layer
        """
        super().__init__()
        self.start_shape = start_shape
        self.img_shape = img_shape
        self.emb_size = emb_size
        self.drop_p = drop_p
        self.bnorm = deconv_bnorm
        print("deconv using bnorm:", self.bnorm)

        flat_start = int(np.prod(start_shape))
        modules = [
                   nn.LayerNorm(emb_size),
                   nn.Linear(emb_size, flat_start),
                   Reshape((-1, *start_shape))
                   ]
        depth, height, width = start_shape
        first_ksize = 9
        first_stride = 2
        self.sizes = []
        deconv = deconv_block(depth, depth, ksize=first_ksize,
                                            stride=first_stride,
                                            padding=0,
                                            bnorm=self.bnorm,
                                            drop_p=self.drop_p)
        height, width = update_shape((height,width),kernel=first_ksize,
                                                    stride=first_stride,
                                                    op="deconv")
        print("Img shape:", self.img_shape)
        print("Start Shape:", start_shape)
        print("h:", height, "| w:", width)
        self.sizes.append((height, width))
        modules.append(deconv)

        ksizes = [5,5,4,4,4,4,4,4,4]
        padding = 0
        strides = [1,1,1,2,2,2,1,1]
        i = -1
        while height < self.img_shape[-2] and width < self.img_shape[-1]:
            i+=1
            ksize = ksizes[i]
            stride = strides[i]
            modules.append(nn.LayerNorm((depth,height,width)))
            modules.append(deconv_block(depth, max(depth // 2, 16),
                                        ksize=ksize, padding=padding,
                                        stride=stride, bnorm=self.bnorm,
                                        drop_p=drop_p))
            depth = max(depth // 2, 16)
            height, width = update_shape((height,width), kernel=ksize,
                                                         stride=stride,
                                                         padding=padding,
                                                         op="deconv")
            self.sizes.append((height, width))
            print("h:", height, "| w:", width, "| d:", depth)
        
        diff = height-self.img_shape[-2]
        k = diff + 1
        modules.append(nn.Conv2d(depth, self.img_shape[0], k))
        height, width = update_shape((height,width), kernel=k)
        print("decoder:", height, width)
        self.sizes.append((height, width))
        
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        """
        x - torch FloatTensor (B,E)
        """
        return self.sequential(x)

    def extra_repr(self):
        s = "start_shape={}, img_shape={}, bnorm={}, drop_p={}"
        return s.format(self.start_shape, self.img_shape, self.bnorm,
                                                        self.drop_p)

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

