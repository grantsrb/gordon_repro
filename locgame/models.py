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

class LocatorBase(TransformerBase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    #def __init__(self, seq_len=None, n_vocab=None,
    #                                 emb_size=512,
    #                                 enc_slen=None,
    #                                 dec_slen=None,
    #                                 attn_size=64,
    #                                 n_heads=8,
    #                                 enc_layers=3,
    #                                 dec_layers=3,
    #                                 enc_mask=False,
    #                                 class_h_size=4000,
    #                                 class_bnorm=True,
    #                                 class_drop_p=0,
    #                                 act_fxn="ReLU",
    #                                 enc_drop_p=0,
    #                                 dec_drop_p=0,
    #                                 collapse_drop_p=0,
    #                                 expand_drop_p=0,
    #                                 ordered_preds=False,
    #                                 gen_decs=False,
    #                                 init_decs=False,
    #                                 idx_inputs=True,
    #                                 idx_outputs=True,
    #                                 prob_embs=False,
    #                                 prob_attn=False,
    #                                 mask_idx=0,
    #                                 start_idx=None,
    #                                 stop_idx=None,
    #                                 multi_init=False,
    #                                 **kwargs):
    #    """
    #    seq_len: int or None
    #        the maximum length of the sequences to be analyzed. If None,
    #        dec_slen and enc_slen must not be None
    #    enc_slen: int or None
    #        the length of the sequences to be encoded
    #    dec_slen: int or None
    #        the length of the sequences to be decoded
    #    n_vocab: int
    #        the number of words in the vocabulary
    #    emb_size: int
    #        the size of the embeddings
    #    attn_size: int
    #        the size of the projected spaces in the attention layers
    #    n_heads: int
    #        the number of attention heads
    #    enc_layers: int
    #        the number of encoding layers
    #    dec_layers: int
    #        the number of decoding layers
    #    enc_mask: bool
    #        if true, encoder uses a mask
    #    class_h_size: int
    #        the size of the hidden layers in the classifier
    #    class_bnorm: bool
    #        if true, the classifier uses batchnorm
    #    class_drop_p: float
    #        the dropout probability for the classifier
    #    act_fxn: str
    #        the activation function to be used in the MLPs
    #    collapse_type: str
    #        the type of collapsing module for the embedding encoding
    #    expand_type: str
    #        the type of expanding module for the embedding encoding
    #    enc_drop_ps: float or list of floats
    #        the dropout probability for each encoding layer
    #    dec_drop_ps: float or list of floats
    #        the dropout probability for each decoding layer
    #    collapse_drop_p: float
    #        the dropout probability for the collapsing layer
    #    expand_drop_p: float
    #        the dropout probability for the expanding layer
    #    ordered_preds: bool
    #        if true, the decoder will mask the predicted sequence so
    #        that the attention modules will not see the tokens ahead
    #        located further along in the sequence.
    #    gen_decs: bool
    #        if true, decodings are generated individually and used
    #        as the inputs for later decodings. (stands for generate
    #        decodings). This ensures earlier attention values are
    #        completely unaffected by later inputs.
    #    init_decs: bool
    #        if true, an initialization decoding vector is learned as
    #        the initial input to the decoder.
    #    idx_inputs: bool
    #        if true, the inputs are integer (long) indexes that require
    #        an embedding layer. Otherwise it is assumed that the inputs
    #        are feature vectors that do not require an embedding layer
    #    idx_outputs: bool
    #        if true, the output sequence (y) is integer (long) indexes
    #        that require an embedding layer. Otherwise it is assumed
    #        that the outputs are feature vectors that do not require
    #        an embedding layer
    #    prob_embs: bool
    #        if true, all embedding vectors are treated as parameter
    #        vectors for gaussian distributions before being fed into
    #        the transformer architecture
    #    prob_attn: bool
    #        if true, the queries and keys are projected into a mu and
    #        sigma vector and sampled from a gaussian distribution
    #        before the attn mechanism
    #    mask_idx: int
    #        the numeric index of the mask token
    #    start_idx: int
    #        the numeric index of the start token
    #    stop_idx: int
    #        the numeric index of the stop token
    #    multi_init: bool
    #        if true, the initialization vector has a unique value for
    #        each slot in the generated sequence
    #    """

class RNNLocator(LocatorBase):
    def __init__(self, cnn_type="SimpleCNN", **kwargs):
        super().__init__(**kwargs)
        self.cnn_type = cnn_type
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
        self.h_init = torch.randn(1,self.emb_size)
        divisor = float(np.sqrt(self.emb_size))
        self.h_init = nn.Parameter(self.h_init/divisor)

        self.rnn = nn.GRUCell(input_size=self.emb_size,
                              hidden_size=self.emb_size)

        self.locator = nn.Sequential(
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.class_h_size),
            globals()[self.act_fxn](),
            nn.LayerNorm(self.class_h_size),
            nn.Linear(self.class_h_size, 2),
            nn.Tanh()
        )
        self.pavlov = nn.Sequential(
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.class_h_size),
            globals()[self.act_fxn](),
            nn.LayerNorm(self.class_h_size),
            nn.Linear(self.class_h_size, 1)
        )

    def reset_h(self, batch_size=1):
        self.h = self.h_init.repeat(batch_size,1)

    def forward(self, x):
        """
        x: torch float tensor (B,C,H,W)
        """
        feats = self.cnn(x)
        feats = self.pos_encoder(feats)
        feat = self.extractor(self.h.unsqueeze(1), feats)
        self.h = self.rnn(feat.mean(1),self.h)
        loc = self.locator(self.h)
        rew = self.pavlov(self.h)
        return loc,rew

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
        chans = [32,64,128,256,self.emb_size]
        self.chans = chans
        ksize = 3
        stride = 1
        padding = 0
        block = self.get_conv_block(in_chan=self.img_shape[-3],
                                    out_chan=32,
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

class MiddleCNN(CNNBase):
    """
    Middle complexity model
    """
    def __init__(self, emb_size, intm_attn=0, chans=None,
                                              strides=None,
                                              **kwargs):
        """
        emb_size: int
        intm_attn: int
            an integer indicating the number of layers for an attention
            layer in between convolutions
        chans: list of int
            a list of channel depths for each layer
        strides: list of int
            a list of cnn strides corresponding to each layer
        """
        super().__init__(**kwargs)
        self.emb_size = emb_size
        self.intm_attn = intm_attn
        self.conv_blocks = nn.ModuleList([])
        self.intm_attns = nn.ModuleList([])
        self.shapes = []
        shape = self.img_shape[-2:]
        self.shapes.append(shape)
        if chans is None:
            chans = [32,64,128,256,256,512,512,self.emb_size]
        self.chans = chans
        assert self.emb_size == chans[-1]
        if strides is None:
            strides = [1,1,1,2,2,2,2,(1,2)]
        self.strides = strides
        ksize = 3
        padding = 0
        stride = self.strides[0]
        block = self.get_conv_block(in_chan=self.img_shape[-3],
                                    out_chan=chans[0],
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
            stride = self.strides[i+1]
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


