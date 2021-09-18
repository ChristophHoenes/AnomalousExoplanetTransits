import warnings

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class ResBlock(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=None, dilation=1,
                 activation='ReLU', batch_norm=True, pre_activation=True, downsample_first=False, transpose=False):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride=2*stride if downsample_first else stride,
                               padding=padding, dilation=dilation, batch_norm=False, activation=None,
                               transpose=transpose, bias=False)
        self.activation1 = select_activation_function(activation)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               dilation=dilation, batch_norm=False, activation=None, transpose=transpose, bias=False)
        self.activation2 = select_activation_function(activation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.activation = activation
        self.batch_norm = batch_norm
        self._batch_norm_module1 = nn.BatchNorm1d(in_channels) if batch_norm else None
        self._batch_norm_module2 = nn.BatchNorm1d(out_channels) if batch_norm else None
        if downsample_first or out_channels != in_channels:
            self.res_projection = ConvLayer(in_channels, out_channels,
                                            kernel_size=2 if downsample_first else 1,
                                            stride=2*stride if downsample_first else 1,
                                            padding=0, dilation=1, batch_norm=False,
                                            activation=None, transpose=transpose)
        else:
            self.res_projection = None
        self.downsample_first = downsample_first
        self.pre_activation = pre_activation

    def forward(self, x):
        residual = self.res_projection(x) if self.res_projection is not None else x

        if self.pre_activation:
            out = self._batch_norm_module1(x) if self.batch_norm else x
            out = self.activation1(out)
            out = self.conv1(out)
            out = self._batch_norm_module2(out) if self.batch_norm else out
            out = self.activation2(out)
            out = self.conv2(out)
            return out + residual
        else:
            out = self.conv1(x)
            out = self._batch_norm_module1(out) if self.batch_norm else out
            out = self.activation1(out)
            out = self.conv2(out)
            out = self._batch_norm_module2(out) if self.batch_norm else out
            return self.activation2(out + residual)


class ConvLayer(nn.Module):

    def __init__(self, in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=None, padding_mode='zeros',
                 dilation=1, activation='ReLU', batch_norm=True, transpose=False, bias=None):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.activation = activation
        self.batch_norm = batch_norm
        if padding is None:
            padding_amount = int(np.ceil((dilation * (kernel_size - 1))/2))
        else:
            padding_amount = padding
        if transpose:
            modules = [nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_amount,
                                          padding_mode=padding_mode,
                                          output_padding=stride-1 if padding_amount > 0 else 0, dilation=dilation,
                                          bias=not batch_norm if bias is None else bias)]
        else:
            modules = [nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_amount,
                                 padding_mode=padding_mode,
                                 dilation=dilation, bias=not batch_norm if bias is None else bias)]
        if batch_norm:
            modules.append(nn.BatchNorm1d(out_channels))
        if activation is not None:
            modules.append(select_activation_function(activation))
        self._components = nn.Sequential(*modules)

    def forward(self, x):
        return self._components(x)


class LinearLayer(nn.Module):

    def __init__(self, in_channels=16, out_channels=16, dropout=0.0, activation=None, batch_norm=False):
        super(LinearLayer, self).__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
        modules = [nn.Linear(in_channels, out_channels)]
        if self.activation is not None:
            modules.append(select_activation_function(activation))
        if batch_norm:
            modules.append(nn.BatchNorm1d(out_channels))
        if dropout > 0.0:
            modules.append(nn.Dropout(p=self.dropout))
        self._components = nn.Sequential(*modules)

    def forward(self, x):
        return self._components(x)


class PoolingLayer(nn.Module):
    def __init__(self, out_channels=None, pool_type='stride', pool_kernel=2, pool_stride=2, pool_padding=None,
                 map_size=None):
        super(PoolingLayer, self).__init__()
        self.out_channels = out_channels
        self.pool_type = pool_type
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.map_size = map_size
        self._indices = None
        if pool_padding == None:
            pool_padding = (pool_kernel - 1) // 2
        if pool_type == 'max_indices':
            self.pooling = nn.MaxPool1d(pool_kernel, pool_stride, padding=pool_padding, return_indices=True)
        elif pool_type == 'max':
            self.pooling = nn.MaxPool1d(pool_kernel, pool_stride, padding=pool_padding, return_indices=False)
        elif pool_type == 'mean':
            self.pooling = MeanPool1d(pool_kernel, pool_stride, padding=pool_padding, return_indices=False)
        elif pool_type == 'global_max':
            self.pooling = GlobalMaxPool1d(map_size)
        elif pool_type == 'global_avg':
            self.pooling = GlobalAvgPool1d(map_size)
        elif pool_type == 'stride':
            assert out_channels is not None, "Passing the number of channels of the previous layer is required" \
                                             " for pool_type 'stride'!"
            self.pooling = StridePool1d(pool_kernel, pool_stride, out_channels, padding=pool_padding)
        else:
            raise RuntimeWarning("Unknown Pooling function {} encountered!".format(self.pooling))

    @property
    def indices(self):
        return self._indices

    def forward(self, x):
        if self.pool_type == 'max_indices':
            out, indices = self.pooling(x)
            self._indices = indices
            return out
        return self.pooling(x)


class UnPoolingLayer(nn.Module):
    def __init__(self, out_channels, unpool_type='stride', pool_kernel=2, pool_stride=2, pool_padding=None, counterpart=None):
        super(UnPoolingLayer, self).__init__()
        self.unpool_type = unpool_type
        if pool_padding == None:
            pool_padding = (pool_kernel - 1) // 2
        if unpool_type == 'max_indices':
            self.unpooling = MaxUnPool1d(counterpart, pool_kernel, pool_stride, padding=pool_padding)
        elif unpool_type == 'max':
            self.unpooling = Upsample1d(pool_stride)
        elif unpool_type == 'mean':
            self.unpooling = MeanUnool1d(pool_kernel)
        elif unpool_type == 'global_max':
            assert counterpart.map_size is not None, \
                "map_size (size of reduced dimension) was not specified during global maximum pooling. " \
                " --> Operation not reversable!"
            self.unpooling = Upsample1d(counterpart.map_size)
        elif unpool_type == 'global_avg':
            assert counterpart.map_size is not None, \
                "map_size (size of reduced dimension) was not specified during global average pooling. " \
                " --> Operation not reversable!"
            self.unpooling = Upsample1d(counterpart.map_size)
        elif unpool_type == 'stride':
            self.unpooling = StrideUnPool1d(pool_kernel, pool_stride, out_channels, padding=pool_padding)
        else:
            raise RuntimeWarning("Unknown Pooling function {} encountered!".format(self.pooling))

    def forward(self, x, indices=None):
        if self.unpool_type == 'max_indices':
            return self.unpooling(x, indices)
        return self.unpooling(x)


class MaxUnPool1d(nn.Module):

    def __init__(self, counterpart, pool_kernel, pool_stride, padding=None):
        assert isinstance(counterpart, PoolingLayer), "Counterpart of MaxUnPool1d should be MaxPool1d!"
        super(MaxUnPool1d, self).__init__()
        if padding == None:
            padding = (pool_kernel - 1) // 2
        self.module = nn.MaxUnpool1d(kernel_size=pool_kernel, stride=pool_stride, padding=padding)
        self._counterpart = counterpart

    def forward(self, x):
        return self.module(x, self._counterpart.indices)


class Upsample1d(torch.nn.Module):
    def __init__(self, scale_factor):
        super(Upsample1d, self).__init__()
        self._scale_factor = scale_factor
        self._module = nn.UpsamplingNearest2d(scale_factor=scale_factor)

    @property
    def scale_factor(self):
        return self._scale_factor

    def forward(self, x):
        out = torch.unsqueeze(x, 3)
        out = self._module(out)
        return out[:, :, :, 0]


class MeanPool1d(nn.Module):

    def __init__(self, kernel_size, stride, padding=0, return_indices=True):
        super(MeanPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.return_indices = return_indices
        self.padding = padding

    def forward(self, x):
        if self.return_indices:
            return F.avg_pool1d(x, self.kernel_size, stride=self.stride, padding=self.padding), None
        else:
            F.avg_pool1d(x, self.kernel_size, stride=self.stride, padding=self.padding)


class MeanUnool1d(nn.Module):

    def __init__(self, scale):
        super(MeanUnool1d, self).__init__()
        self.factor = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode='linear')


class StridePool1d(nn.Module):

    def __init__(self, pool_kernel, pool_stride, channels, padding=0, activation='ReLU', batch_norm=False):
        super(StridePool1d, self).__init__()
        self.layer = nn.Conv1d(channels, channels, kernel_size=pool_kernel, stride=pool_stride, padding=padding)
        self.activation = select_activation_function(activation)
        self.batch_norm = nn.BatchNorm1d(channels) if batch_norm else None

    def forward(self, x):
        out = self.layer(x) if self.batch_norm is None else self.batch_norm(self.layer(x))
        return out if self.activation is None else self.activation(out)


class StrideUnPool1d(nn.Module):

    def __init__(self, pool_kernel, pool_stride, channels, padding=0, activation='ReLU', batch_norm=False):
        super(StrideUnPool1d, self).__init__()
        self.layer = nn.ConvTranspose1d(channels, channels, kernel_size=pool_kernel, stride=pool_stride, padding=padding)
        self.activation = select_activation_function(activation)
        self.batch_norm = nn.BatchNorm1d(channels) if batch_norm else None

    def forward(self, x):
        out = self.layer(x) if self.batch_norm is None else self.batch_norm(self.layer(x))
        return out if self.activation is None else self.activation(out)


class GlobalMaxPool1d(nn.Module):

    def __init__(self, map_size=None, channel_dim=-1):
        super(GlobalMaxPool1d, self).__init__()
        self.map_size = map_size
        self.channel_dim = channel_dim

    def forward(self, x):
        return x.max(dim=self.channel_dim, keepdim=True).values


class GlobalAvgPool1d(nn.Module):
    def __init__(self, map_size=None, channel_dim=-1):
        super(GlobalAvgPool1d, self).__init__()
        self.map_size = map_size
        self.channel_dim = channel_dim

    def forward(self, x):
        return x.mean(dim=self.channel_dim, keepdim=True)


class UnFlatten(nn.Module):

    def __init__(self, channels):
        super(UnFlatten, self).__init__()
        self.channels = channels

    def forward(self, x):
        return x.unflatten(1, (self.channels, x.size()[1]//self.channels))


class VAE(nn.Module):

    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.param_layer = nn.Linear(encoder.out_size, 2*latent_dim)
        if isinstance(decoder[0], LinearLayer):
            self.resize = nn.Sequential(nn.Linear(latent_dim, encoder.out_size))
        else:
            self.resize = nn.Sequential(nn.Linear(latent_dim, encoder.out_size), UnFlatten(encoder[-1].out_channels))
        self.decoder = decoder
        self.latent_dim = latent_dim

    def sample(self, params):
        assert len(params) > 1, "Latent code dimension is too small! Check your architecture."
        batch_zize = params.size()[0]
        # split code into mean and log std
        split = params.size()[1] // 2
        mean = params[:, :split]
        log_std = params[:, split:2 * split]
        std = torch.exp(log_std)
        # sample code
        code = torch.randn_like(std) * std + mean
        # calculate kl loss
        kl_loss = 0.5 * (mean ** 2 + std ** 2 - 2 * log_std - 1).sum() / (batch_zize * split)
        return code, mean, kl_loss

    def encode(self, x):
        encoded = self.encoder(x)
        flattened = self.flatten(encoded)
        params = self.param_layer(flattened)
        return params[:, :self.latent_dim]

    def decode(self, z):
        resized = self.resize(z)
        return self.decoder(resized)

    def forward(self, x, extended_output=False, sample=True):
        encoded = self.encoder(x)
        flattened = self.flatten(encoded)
        params = self.param_layer(flattened)
        sampled, mean, kl_loss = self.sample(params)
        resized = self.resize(sampled) if sample else self.resize(mean)
        reconstruction = self.decoder(resized)
        if extended_output:
            return reconstruction, sampled, mean, kl_loss
        return reconstruction


class LinearVAE(VAE):

    def __init__(self, input_size, latent_dim=2, dropout=0.0, activation='ReLU', batch_norm=False):
        model_depth = int(math.log(input_size, 2)) - int(math.log(latent_dim*2, 2))
        layers= []
        sizes = []
        current_size = input_size
        for i in range(model_depth-1):
            sizes.append((current_size, current_size//2))
            layers.append(LinearLayer(*sizes[-1], dropout=dropout, activation=activation, batch_norm=batch_norm))
            current_size = current_size//2
        encoder = nn.Sequential(*layers)
        encoder.out_size = current_size
        layers = [nn.Linear(latent_dim, current_size), select_activation_function(activation)]
        for s, size in enumerate(reversed(sizes)):
            if s == len(sizes):
                layers.append(LinearLayer(size[1], size[0], dropout=0.0, activation=None, batch_norm=False))
            else:
                layers.append(LinearLayer(size[1], size[0], dropout=dropout, activation=activation, batch_norm=batch_norm))
        decoder = nn.Sequential(*layers)
        super(LinearVAE, self).__init__(encoder, decoder, latent_dim)


def create_custom_network(config, name):
    net = config[name]
    network_layers = []
    for layer in net:
        if layer.startswith('LinearLayer'):
            for i in range(net[layer]['num_repetitions']):
                network_layers.append(LinearLayer(**net[layer]['params']))
        elif layer.startswith('ConvLayer'):
            for i in range(net[layer]['num_repetitions']):
                network_layers.append(ConvLayer(**net[layer]['params']))
        elif layer.startswith('PoolingLayer'):
            network_layers.append(PoolingLayer(**net[layer]['params']))
        elif layer.startswith('ResBlock'):
            for i in range(net[layer]['num_repetitions']):
                network_layers.append(ResBlock(**net[layer]['params']))
        elif layer.startswith('Flatten'):
            network_layers.append(nn.Flatten(**net[layer]['params']))
        else:
            raise RuntimeError("Unknown layer type {}!".format(layer))
    return network_layers


class CustomVAE(VAE):

    def __init__(self, config, networks=('encoder', 'decoder')):
        self.config = config
        self.network_dict = {}
        for network in networks:
            network_layers = create_custom_network(config, network)
            if network.startswith('decoder') and len(network_layers) == 0:
                network_layers = self._mirror_encoder(self.network_dict['encoder'])
            self.network_dict[network] = nn.Sequential(*network_layers)
        if self.network_dict['encoder'][-1].out_channels is None:
            raise RuntimeError("Argument 'out_channels' must not be None in last encoder layer!")
        self.network_dict['encoder'].out_size = config['encoder_out_size']
        super(CustomVAE, self).__init__(self.network_dict['encoder'], self.network_dict['decoder'], config['latent_dim'])

    def _mirror_encoder(self, encoder):
        module_list = []
        for l, layer in enumerate(reversed(encoder)):
            if isinstance(layer, LinearLayer) or isinstance(layer, ConvLayer) or isinstance(layer, ResBlock):
                activation = layer.activation if l < (len(encoder)-1) else None
                batch_norm = layer.batch_norm if l < (len(encoder)-1) else False
                if isinstance(layer, LinearLayer):
                    module_list.append(LinearLayer(in_channels=layer.out_channels, out_channels=layer.in_channels,
                                                   dropout=layer.dropput, activation=activation,
                                                   batch_norm=batch_norm))
                elif isinstance(layer, ConvLayer):
                    module_list.append(ConvLayer(in_channels=layer.out_channels, out_channels=layer.in_channels,
                                                 kernel_size=layer.kernel_size, stride=layer.stride,
                                                 padding=layer.padding, dilation=layer.dilation, activation=activation,
                                                 batch_norm=batch_norm, transpose=True))
                elif isinstance(layer, ResBlock):
                    module_list.append(ResBlock(in_channels=layer.out_channels, out_channels=layer.in_channels,
                                                kernel_size=layer.kernel_size, stride=layer.stride,
                                                padding=layer.padding, dilation=layer.dilation, activation=activation,
                                                batch_norm=batch_norm, transpose=True,
                                                pre_activation=layer.pre_activation,
                                                downsample_first=layer.downsample_first))
            elif isinstance(layer, PoolingLayer):
                counterpart = layer if layer.pool_type in ['max_indices', 'global_max', 'global_avg'] else None
                module_list.append(UnPoolingLayer(layer.out_channels, unpool_type=layer.pool_type,
                                                  pool_kernel=layer.pool_kernel, pool_stride=layer.pool_stride,
                                                  pool_padding=layer.pool_padding, counterpart=counterpart))

        return nn.Sequential(*module_list)


class VerticalFitter(nn.Module):

    def __init__(self, size):
        super(VerticalFitter, self).__init__()
        self.influence_trend1 = torch.linspace(1., 0., size).unsqueeze(dim=0).unsqueeze(dim=0)
        self.influence_trend2 = torch.ones(size) - self.influence_trend1

    def _apply(self, fn):
        super(VerticalFitter, self)._apply(fn)
        self.influence_trend1 = fn(self.influence_trend1)
        self.influence_trend2 = fn(self.influence_trend2)
        return self

    def forward(self, data, model, scale, trend):
        oot_model = model.max(dim=-1, keepdims=True).values
        model -= oot_model
        model *= (1.-scale.unsqueeze(dim=1).unsqueeze(dim=1))

        model += trend[:, 0].unsqueeze(dim=1).unsqueeze(dim=1) * self.influence_trend1 + \
                 trend[:, 1].unsqueeze(dim=1).unsqueeze(dim=1) * self.influence_trend2

        vertical_loss = F.mse_loss(model, data)

        return model.detach(), vertical_loss


class TransitShapeVAE(CustomVAE):

    def __init__(self, config, mean_transit=None):
        super(TransitShapeVAE, self).__init__(config)
        self.param_layer = nn.Linear(self.encoder.out_size, 2 * self.latent_dim + 5)
        self.mean_transit = mean_transit
        self.vertical_fitter = VerticalFitter(config["input_size"])

    def _apply(self, fn):
        super(TransitShapeVAE, self)._apply(fn)
        if self.mean_transit is not None:
            self.mean_transit = fn(self.mean_transit)
        return self

    def forward(self, x, h_scale_label=None, h_shift_label=None, use_h_labels=False, sample=True):

        # standard VAE forward pass
        encoded = self.encoder(x)
        flattened = self.flatten(encoded)
        params = self.param_layer(flattened)
        sampled, mean, kl_loss = self.sample(params[:, :-5])
        resized = self.resize(sampled) if sample else self.resize(mean)
        reconstruction = self.decoder(resized) if self.mean_transit is None else self.mean_transit + self.decoder(
            resized)

        # fit predicted shape vertically and horizontally to input
        model_to_fit = reconstruction.detach().clone()

        fitting_params = params[:, -5:]
        fitting_loss = [torch.Tensor([0.]), torch.Tensor([0.])]

        # horizontal (x/time-axis) allignment
        if use_h_labels:
            horizontal_scale = h_scale_label
            horizontal_shift = h_shift_label
        else:
            horizontal_scale = torch.sigmoid(fitting_params[:, 0]) * 3.
            horizontal_shift = torch.tanh(fitting_params[:, 1])
            if h_scale_label is not None:
                fitting_loss[0] = F.mse_loss(horizontal_scale, h_scale_label)
            if h_shift_label is not None:
                fitting_loss[1] = F.mse_loss(horizontal_shift, h_shift_label)
        model_to_fit = self._horizontal_scale_n_shift(model_to_fit, horizontal_scale, horizontal_shift)


        # vertical (y/flux-axis) alignment
        vertical_scale = torch.sigmoid(fitting_params[:, 2])
        vertical_trend = torch.tanh(fitting_params[:, 3:])
        reconstruction_fitted, vertical_loss = self.vertical_fitter(x, model_to_fit, vertical_scale, vertical_trend)
        fitting_loss.append(vertical_loss)
        fitting_loss = tuple(fitting_loss)

        return reconstruction, reconstruction_fitted, sampled, mean, kl_loss, fitting_loss

    def _horizontal_scale_n_shift(self, model_to_fit, scale_params, shift_params):
        input_size = model_to_fit.size()[-1]
        corrected_models = torch.ones_like(model_to_fit) * model_to_fit.max(dim=-1, keepdims=True).values
        new_sizes = (scale_params * input_size).round().int().tolist()
        shifts = (shift_params * input_size//2).round().int().tolist()
        for s, (new_size, shift) in enumerate(zip(new_sizes, shifts)):
            mid = input_size//2
            if new_size <= 2:
                continue
            resized = F.interpolate(model_to_fit[s].unsqueeze(dim=0), new_size, mode='linear', align_corners=False)
            if new_size > input_size:
                mid_new = new_size // 2
                start = mid_new - mid
                corrected_models[s, :, :] = resized[:, :, start:start+input_size]
            else:
                left_size = new_size//2
                right_size = new_size - left_size
                corrected_models[s, :, mid-left_size:mid+right_size] = resized
            corrected_models[s] = corrected_models[s].roll(shifts=shift, dims=-1)
        return corrected_models


def select_activation_function(activation):
    if activation in ['ReLU', 'relu']:
        return nn.ReLU()
    elif activation.startswith('LeakyReLU') or activation.startswith('leaky_relu'):
        try:
            act_strip = activation.strip('LeakyReLU')
            act_strip = act_strip.strip('leaky_relu')
            act_strip = act_strip.strip('_')
            return nn.LeakyReLU(negative_slope=float(act_strip))
        except ValueError:
            return nn.LeakyReLU()
    elif activation in ['PReLU', 'prelu']:
        return nn.PReLU()
    elif activation in ['GELU', 'gelu']:
        return nn.GELU()
    elif activation in ['ELU', 'elu']:
        return nn.ELU()
    elif activation in ['SELU', 'selu']:
        return nn.SELU()
    elif activation in ['Tanh', 'tanh']:
        return nn.Tanh()
    elif activation in ['Sigmoid', 'sigmoid']:
        return nn.Sigmoid()
    elif activation in ['Softplus', 'softplus']:
        return nn.Softplus
    else:
        warnings.warn('Activation function "{}" is unknown! No activation will be used.'.format(activation), RuntimeWarning)
        return None
