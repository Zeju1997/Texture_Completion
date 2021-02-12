import torch
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet.models import decoder

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder,
    'main_decoder': decoder.MainLocalDecoder
}


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, main_decoder=None, main_encoder=None, device=None):
        super().__init__()

        '''
        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None
        '''

        if main_decoder is not None:
            self.main_decoder = main_decoder.to(device)
        else:
            self.main_decoder = None

        if main_encoder is not None:
            self.main_encoder = main_encoder.to(device)
        else:
            self.main_encoder = None

        self._device = device

    def forward(self, p, inputs, inputs_colors, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        '''
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)

        main_c = self.encode_inputs(inputs)
        main_p_r = self.decode(p, main_c, **kwargs)
        '''

        c = self.main_encode_inputs(inputs, inputs_colors)
        p_r = self.main_decode(p, c, **kwargs)

        main_c = self.main_encode_inputs(inputs, inputs_colors)
        main_p_r = self.main_decode(p, main_c, **kwargs)

        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def main_encode_inputs(self, inputs, inputs_colors):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.main_encoder is not None:
            c = self.main_encoder(inputs, inputs_colors)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def main_decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits = self.main_decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        # p_col = dist.Bernoulli(logits=logits_col)
        return p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
