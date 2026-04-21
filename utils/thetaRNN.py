#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:01:17 2022

modeled after: 
https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

@author: dl2820
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch
import torch.jit as jit
from typing import List, Optional, Tuple
from torch import Tensor
import numbers
import numpy as np

#class RNNLayer(jit.ScriptModule):
class thetaRNNLayer(nn.Module):    
    """
    A wrapper for customized RNN layers... inputs should match torch.nn.RNN
    conventions for batch_first=True, with theta taking the place of the batch dimension
    theta x sequence x neuron 
    #in the future, can make batch x sequence x neuron x theta...
    
    forward method inputs:
    input  [1 x sequence x neuron] (optional if internal provided)
    internal [theta x sequence x neuron] (optional if input provided)
    state [DL fill this in] (optional)
    """
    def __init__(self, cell, trunc, *cell_args, defaultTheta=0, continuousTheta=False,
                 hidden_init_sigma=0.1):
        super(thetaRNNLayer, self).__init__()
        self.cell = cell(*cell_args)
        self.trunc = trunc
        self.theta = defaultTheta
        self.continuousTheta = continuousTheta
        self.hidden_init_sigma = hidden_init_sigma
        

    #@jit.script_method
    def forward(self, input: Optional[Tensor] = None,
                internal: Optional[Tensor] = None,
                state: Optional[Tensor] = None,
                theta=None) -> Tuple[Tensor, Tensor]:
        if theta is None:
            theta = self.theta

        if input is None:
            input = torch.tensor([])
        if internal is None:
            internal = torch.tensor([])
        if state is None:
            state = torch.tensor([])

        if theta == 0:
            return self._forward_batched(input=input,
                                         internal=internal,
                                         state=state)
        return self._forward_theta(input=input,
                                   internal=internal,
                                   state=state,
                                   theta=theta)

    def _forward_batched(
        self,
        input: Optional[Tensor] = None,
        internal: Optional[Tensor] = None,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run the recurrent layer with true batch semantics when theta=0."""

        if input is None:
            input = torch.tensor([])
        if internal is None:
            internal = torch.tensor([])
        if state is None:
            state = torch.tensor([])

        if input.size(0) == 0:
            input = torch.zeros(
                internal.size(0),
                internal.size(1),
                self.cell.input_size,
                device=self.cell.weight_hh.device,
            )
        if state.size(0) == 0:
            state = torch.empty(
                1,
                input.size(0),
                self.cell.hidden_size,
                device=self.cell.weight_hh.device,
            ).normal_().mul_(self.hidden_init_sigma)
        if internal.size(0) == 0:
            internal = torch.zeros(
                input.size(0),
                input.size(1),
                self.cell.hidden_size,
                device=self.cell.weight_hh.device,
            )

        inputs = input.unbind(1)
        internals = internal.unbind(1)
        state_tuple = (torch.squeeze(state, 0), 0)
        outputs = []

        for i in range(len(inputs)):
            if i % self.trunc == 0 and i > 0:
                state_tuple = tuple(component.detach() if torch.is_tensor(component) else component for component in state_tuple)
            out, state_tuple = self.cell(inputs[i], internals[i], state_tuple)
            outputs.append(out)

        state_out = torch.unsqueeze(state_tuple[0], 0)
        return torch.stack(outputs, 1), state_out

    def _forward_theta(
        self,
        input: Optional[Tensor] = None,
        internal: Optional[Tensor] = None,
        state: Optional[Tensor] = None,
        theta: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Run the legacy theta rollout path used by the original architecture family."""

        if input is None:
            input = torch.tensor([])
        if internal is None:
            internal = torch.tensor([])
        if state is None:
            state = torch.tensor([])

        if input.size(0)==0:
            input = torch.zeros(internal.size(0),internal.size(1),self.cell.input_size,
                                device=self.cell.weight_hh.device)
        if state.size(0)==0:
            state = torch.zeros(1,1,self.cell.hidden_size,
                                device=self.cell.weight_hh.device)
        if internal.size(0)==0:
            internal = torch.zeros(theta+1,input.size(1),self.cell.hidden_size,
                                   device=self.cell.weight_hh.device)
        
        if input.size(0)<theta+1:
            input = nn.functional.pad(input=input, pad=(0,0,0,0,0,theta), 
                                    mode='constant', value=0)   
        
        inputs = input.unbind(1)
        internals = internal.unbind(1)
        state_tuple = (torch.squeeze(state,0),0)
        outputs = []
        
        for i in range(len(inputs)):
            if i % self.trunc == 0 and i > 0:
                state_tuple = tuple(component.detach() if torch.is_tensor(component) else component for component in state_tuple)
                
            out, state_tuple = self.cell(torch.unsqueeze(inputs[i][0,:],0), 
                                         internals[i][0,:], state_tuple)
                
            state_th = state_tuple
            out = [out]
            for th in range(theta):
                out_th, state_th = self.cell(torch.unsqueeze(inputs[i][th+1,:],0), 
                                             internals[i][th+1,:], state_th)
                out += [out_th]
            out = torch.cat(out,0)
            
            if hasattr(self,'continuousTheta') and self.continuousTheta:
                state_tuple = state_th
                
            outputs += [out]
        
        state_out = torch.unsqueeze(state_tuple[0],0)
        return torch.stack(outputs,1), state_out
        
        
        

#class RNNCell(jit.ScriptModule):
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, musig=None):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #(kaiming he initialization - blows up :'( )
        #std_h = np.sqrt(2.0/hidden_size)
        #std_i = np.sqrt(2.0/input_size)
        #self.weight_ih = Parameter(torch.randn(hidden_size, input_size)*std_i)
        #self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size)*std_h)
        #Pytorch Initalization ("goodbug") with input scaling
        rootk_h = np.sqrt(1./hidden_size)
        rootk_i = np.sqrt(1./input_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk_i-rootk_i)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk_h-rootk_h)
        self.bias = Parameter(torch.zeros(self.hidden_size))
        
        #TODO: Add option for torch.sigmoid or torch.tanh
        self.actfun = torch.nn.ReLU()

    # TODO: with and without history (-h) 
    #@jit.script_method
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tensor:
        hx = state[0]
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())
        x = i_input + h_input
        hy = self.actfun(x + internal + self.bias)
        return hy, (hy,)
    
    
class AdaptingRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, musig=None):
        super(AdaptingRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #Pytorch Initalization ("goodbug") with input scaling
        rootk_h = np.sqrt(1./hidden_size)
        rootk_i = np.sqrt(1./input_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk_i-rootk_i)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk_h-rootk_h)
        self.bias = Parameter(torch.zeros(hidden_size))
        
        self.b = 0.3 #Parameter(torch.ones(1)*0.4)
        self.tau_a = 8. #Parameter(torch.ones(1)*8)
        
        #TODO: Add option for torch.sigmoid or torch.tanh
        self.actfun = torch.nn.ReLU()

    # TODO: with and without history (-h) 
    #@jit.script_method
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tensor:
        hx, ax = state
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())
        x = i_input + h_input
        # TODO check time indices
        ay = ax * (1-1/self.tau_a) + self.b/self.tau_a *hx
        hy = self.actfun(x + internal + self.bias - ax)
        return hy, (hy,ay)



class LayerNormRNNCell(jit.ScriptModule):
    # Change 5: replaced custom LayerNorm with a single fused F.layer_norm call
    #           (one CUDA kernel instead of two separate passes).
    # Change 6: jit.ScriptModule + @jit.script_method → compiled to C++ at class
    #           definition time, eliminating the Python interpreter on every cell call.
    #
    # State convention: Tuple[Tensor, int] where int is an unused placeholder kept
    # for API compatibility with AdaptingLayerNormRNNCell (which stores adapt state).
    def __init__(self, input_size: int, hidden_size: int, musig: List[float] = [0.0, 1.0]):
        super(LayerNormRNNCell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        rootk_h = np.sqrt(1.0 / hidden_size)
        rootk_i = np.sqrt(1.0 / input_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size) * 2 * rootk_i - rootk_i)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size) * 2 * rootk_h - rootk_h)
        # Learnable shift bias for layer-norm (was layernorm.mu).
        # Exposed as self.bias so pRNN's optimizer group still resolves correctly via
        #   model.bias → model.rnn.cell.bias
        self.bias  = Parameter(torch.zeros(hidden_size))
        self.scale = Parameter(torch.ones(hidden_size))

        self.actfun = torch.nn.ReLU()

    @jit.script_method
    def forward(self, input: Tensor, internal: Tensor,
                state: Tuple[Tensor, int]) -> Tuple[Tensor, Tuple[Tensor, int]]:
        hx      = state[0]
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx,    self.weight_hh.t())
        # Single fused kernel: normalize then add learned shift (bias)
        x  = F.layer_norm(i_input + h_input, [self.hidden_size],
                          weight=self.scale, bias=self.bias, eps=1e-4)
        hy = self.actfun(x + internal)
        return hy, (hy, 0)



class AdaptingLayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, musig=[0,1]):
        super(AdaptingLayerNormRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #Pytorch Initalization ("goodbug") with input scaling
        rootk_h = np.sqrt(1./hidden_size)
        rootk_i = np.sqrt(1./input_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk_i-rootk_i)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk_h-rootk_h)
        self.b = 0.3 #Parameter(torch.ones(1)*0.4)
        self.tau_a = 8. #Parameter(torch.ones(1)*8)
        # The layernorms provide learnable biases
        
        self.layernorm = LayerNorm(hidden_size,musig)
        
        self.layernorm.mu = Parameter(torch.zeros(self.hidden_size)+self.layernorm.mu)
        self.bias = self.layernorm.mu
        
        #TODO: Add option for torch.sigmoid or torch.tanh
        self.actfun = torch.nn.ReLU()

    # TODO: with and without history (-h) 
    #@jit.script_method
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tensor:
        hx, ax = state
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())
        x = i_input + h_input
        x = self.layernorm(i_input + h_input)
        # TODO check time indices
        ay = ax * (1-1/self.tau_a) + self.b/self.tau_a *hx
        hy = self.actfun(x + internal - ax)
        return hy, (hy,ay)



#class LayerNorm(jit.ScriptModule):
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, musig):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.mu = musig[0]
        self.sig = musig[1]
        self.normalized_shape = normalized_shape

    #@jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False) + 0.0001
        return mu, sigma

    #@jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.sig + self.mu
    
    
    
#Unit Tests

# F already imported at the top of this file
def test_script_thrnn_layer(seq_len, input_size, hidden_size, trunc, theta):
    inp = torch.randn(1,seq_len,input_size)
    inp = F.pad(inp,(0,0,0,theta))
    state = torch.randn(1, 1, hidden_size)
    internal = torch.zeros(theta+1 ,seq_len+theta, hidden_size)
    rnn = thetaRNNLayer(RNNCell, trunc, input_size, hidden_size)
    out, out_state = rnn(inp, internal, state, theta=theta)

    # Control: pytorch native LSTM
    rnn_ctl = nn.RNN(input_size, hidden_size, 
                     batch_first=True, bias=False, nonlinearity = 'relu')

    for rnn_param, custom_param in zip(rnn_ctl.all_weights[0], rnn.parameters()):
        assert rnn_param.shape == custom_param.shape
        with torch.no_grad():
            rnn_param.copy_(custom_param)
    rnn_out, rnn_out_state = rnn_ctl(inp, state)

    #Check the output matches rnn default for theta=0
    assert (out[0,:,:] - rnn_out).abs().max() < 1e-5
    assert (out_state - rnn_out_state).abs().max() < 1e-5
    
    #Check the theta prediction matches the rnn output when input is withheld
    assert (out[:,-theta-1,0] - rnn_out[0,-theta-1:,0]).abs().max() < 1e-5
    
    return out,rnn_out,inp,rnn

if __name__ == '__main__':
    test_script_thrnn_layer(5, 3, 7, 10, 4)
