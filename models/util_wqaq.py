import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function

# ********************* range_trackers(get range before statistical quantification) *********************
class RangeTracker(nn.Module):
    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':    # A,min/max.shape=(1, 1, 1, 1),layer level
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':  # W,min/max.shape=(N, 1, 1, 1),channel level
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
        self.update_range(min_val, max_val)

#W,min/max.shape=(N, 1, 1, 1),channel level
#Compare this with the previous one to get min/max —— (N, C, W, H)
class GlobalRangeTracker(RangeTracker):  
    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))
# A,min/max.shape=(1, 1, 1, 1),layer level
#get running min/max —— (N, C, W, H)
class AveragedRangeTracker(RangeTracker):  
    def __init__(self, q_level, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        if self.first_a == 0:
            self.first_a.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum)
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)
        
# ********************* quantizers *********************
class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Quantizer(nn.Module):
    def __init__(self, bits, range_tracker):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        # quantization parameters scale 
        self.register_buffer('scale', None)     
        # quantization parameters zero point  
        self.register_buffer('zero_point', None)

    def update_params(self):
        raise NotImplementedError

    #quantize function--floating point number convert to integer number
    def quantize(self, input):
        output = input * self.scale - self.zero_point
        return output

    #the round() function in Eq.3
    def round(self, input):
        output = Round.apply(input)
        return output

    #clamp() function in Eq.4
    def clamp(self, input):
        output = torch.clamp(input, self.min_val, self.max_val)
        return output

    # integer number convert to floating point number
    def dequantize(self, input):
        output = (input + self.zero_point) / self.scale
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            self.range_tracker(input)
            self.update_params()
            output = self.quantize(input)
            output = self.round(output)
            output = self.clamp(output)
            output = self.dequantize(output)
        return output

#SignedQuantizer
class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits - 1)) - 1))
#SymmetricQuantizer,if the n is 8, the range of  SymmetricQuantizer is [-128,+127]
class SymmetricQuantizer(SignedQuantizer):
    def update_params(self):
        #the range after quantization
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))
        #the range before quantization
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))  
        # quantization parameters scale  
        self.scale = quantized_range / float_range    
        # quantization parameters zero point    
        self.zero_point = torch.zeros_like(self.scale)

#UnsignedQuantizer
class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits) - 1))
#AsymmetricQuantizer,if the n is 8, the range of  SymmetricQuantizer is [0,255]
class AsymmetricQuantizer(UnsignedQuantizer):
    def update_params(self):
        quantized_range = self.max_val - self.min_val 
        float_range = self.range_tracker.max_val - self.range_tracker.min_val 
        self.scale = quantized_range / float_range  
        self.zero_point = torch.round(self.range_tracker.min_val * self.scale)

# ********************* Quantized Convolution----1.quantizer A and W 2.convolution operation） *********************
class Conv2d_Q(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,a_bits=8,w_bits=8,q_type=1,first_layer=0,):
        super().__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)
        # Instantiation quantizer（A-layer level，W-channel level）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        #quantizer A and W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(self.weight) 
        # convolution operation
        output = F.conv2d(input=q_input,weight=q_weight,bias=self.bias,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        return output


# ********************* Quantized Fully conection ----1.quantizer A and W 2.Fully conection operation） ***********************
class Linear_Q(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, q_type=1,a_bits=8, w_bits=8):
    super().__init__(in_features=in_features, out_features=out_features, bias=bias)
    if q_type == 0:
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=AveragedRangeTracker(q_level='L'))
    else:
        self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=AveragedRangeTracker(q_level='L'))

  def forward(self, input):
    q_input = self.activation_quantizer(input)
    q_weight = self.weight_quantizer(self.weight) 
    output = F.linear(input=q_input, weight=q_weight, bias=self.bias)
    return output

# ********************* Quantized Max pool ----1.quantizer A  2.Max pool operation）  ***********************
class MaxPool_Q(nn.Linear):
  def __init__(self, in_features, out_features,q_type=1,a_bits=8):
    super().__init__(in_features=in_features, out_features=out_features)
    if q_type == 0:
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
    else:
        self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))

  def forward(self, input):
    q_input = self.activation_quantizer(input)
    output = F.max_pool2d(input=q_input,kernel_size = 2)
    return output

# *********************Quantized Relu ----1.quantizer A  2.Relu operation ***********************
class Relu_Q(nn.Linear):
  def __init__(self, in_features, out_features, q_type=1,a_bits=8):
    super().__init__(in_features=in_features, out_features=out_features)
    if q_type == 0:
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
    else:
        self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))

  def forward(self, input):
    q_input = self.activation_quantizer(input)
    output = F.relu(input=q_input)
    return output

def reshape_to_activation(input):
  return input.reshape(1, -1, 1, 1)

def reshape_to_weight(input):
  return input.reshape(-1, 1, 1, 1)

def reshape_to_bias(input):
  return input.reshape(-1)

# *********************  bn fold quantization convolution（ 1.after bn flod，2.quantizer A and W 3.convolution operation） *********************
class BNFold_Conv2d_Q(Conv2d_Q):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,groups=1,bias=False,eps=1e-5,momentum=0.01, a_bits=8,w_bits=8,q_type=1,first_layer=0,):
        super().__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        # Instantiation quantizer（A-layer level，W-channel level）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        if self.training:
            # Ordinary convolution to get A to get the BN parameter
            output = F.conv2d(input=input,weight=self.weight,bias=self.bias,stride=self.stride,padding=self.padding,)
            # update BN statistical parameter（batch and running）
            dims = [dim for dim in range(4) if dim != 1]
            batch_mean = torch.mean(output, dim=dims)
            batch_var = torch.var(output, dim=dims)
            with torch.no_grad():
                if self.first_bn == 0:
                    self.first_bn.add_(1)
                    self.running_mean.add_(batch_mean)
                    self.running_var.add_(batch_var)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
            # 1.bn flod 
            if self.bias is not None:  
              bias = reshape_to_bias(self.beta + (self.bias -  batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
            else:
              bias = reshape_to_bias(self.beta - batch_mean  * (self.gamma / torch.sqrt(batch_var + self.eps)))
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
        else:
            if self.bias is not None:
              bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
              bias = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps))
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
        # 2.quantizer A and W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(weight) 
        # 3.convolution operation
        if self.training:
          output = F.conv2d(input=q_input,weight=q_weight,bias=self.bias,stride=self.stride,padding=self.padding,)
          output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
          output += reshape_to_activation(bias)
        else:
          output = F.conv2d(input=q_input,weight=q_weight,bias=bias, stride=self.stride,padding=self.padding,)
        return output
