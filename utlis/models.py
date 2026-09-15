#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
# import gym
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.preprocessing import get_action_dim#, is_image_space, maybe_transpose, preprocess_obs
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from utlis.config import get_config

class ExtractFeaturesNetwork(nn.Module):

    def __init__(self):
        super(ExtractFeaturesNetwork, self).__init__()
        self.layer1 = nn.Sequential()

    def forward(self, x):
        out = self.layer1(x)
        return out


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 128, #*# GEANT, Xspedius, SwitchL3, Unnet时128;Abilene 64
            last_layer_dim_vf: int = 128, #*# GEANT, Xspedius, SwitchL3, Unnet时128;Abilene 64

            device: str = "cpu"
    ):
        super(CustomNetwork, self).__init__()

        # # Shared network
        # self.shared_net = nn.Sequential(
        #     nn.Linear(feature_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        # ).to(device)

        self.shared_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),#*#对比实验DRL-TE和ScaleDRL输入维度为 1，其他为3
            # nn.BatchNorm1d(16, momentum=0.8, affine=True),
            nn.ReLU(),
            # nn.MaxPool2d(2), #*# Unnet时使用，其他时注释掉

            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm1d(32, momentum=0.8, affine=True),
            nn.ReLU(),
            # nn.MaxPool2d(2), #*# Xspedius, SwitchL3时使用，Abilene和GEANT时注释掉

            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm1d(32, momentum=0.8, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm1d(32, momentum=0.8, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # nn.AdaptiveMaxPool1d(5)
            nn.Flatten(),
            # nn.Linear(32 * 3 * 3, 64),  #*# Abilene 时输入维度 torch.Size([1, 4, 12, 12])#drlte,
            # nn.Linear(32 * 3 * 3, 128),  #*# Unnet 时输入维度 torch.Size([1, 4, 12, 12])
            nn.Linear(32 * 5 * 5, 128), # 64),  #*# GEANT 时输入维度 torch.Size([1, 4, 23, 23])#drlte,1280 ##!R1_Scal remove节点数>=4且<=7时更为 32 * 4 * 4
            # nn.Linear(32 * 4 * 4, 128),  # 64),  # Xspedius 时输入维度 torch.Size([1, 4, 34, 34])#drlte,1280
            # nn.Linear(32 * 5 * 5, 128),  # 64),  # SwitchL3 时输入维度 torch.Size([1, 4, 42, 42])#drlte,1280
            nn.ReLU(),
        ).to(device)
        # Policy network
        self.policy_net = nn.Sequential(
            # nn.Linear(128, last_layer_dim_pi),
            # nn.ReLU()
        ).to(device)
        # Value network
        self.value_net = nn.Sequential(
            # nn.Linear(128, last_layer_dim_vf),
            # nn.ReLU()
        ).to(device)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # return self.policy_net(features), self.value_net(features)

        # shared_latent = self.shared_net(features)
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared_net(features))
    # def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
    #     return self.policy_net(features)
    #
    # def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
    #     return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        # observation_space: gym.spaces.Space,
        # action_space: gym.spaces.Space,
        # lr_schedule: Schedule,
        # net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        # activation_fn: Type[nn.Module] = nn.Tanh,
        # *args,
        # **kwargs,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        # # Disable orthogonal initialization
        # self.ortho_init = False

        ##重写父类ActorCriticPolicy的属性与方法
        self.action_net = nn.Sequential(nn.Linear(self.mlp_extractor.latent_dim_pi, get_action_dim(self.action_space)),
                                        # nn.Softmax(dim=-1)
                                        # nn.Softplus()
                                        # nn.Sigmoid()
                                        nn.Tanh()
                                        # nn.ReLU()
                                        # nn.LeakyReLU()
                                        # https://blog.csdn.net/DIPDWC/article/details/112686489
                                        )
        self.features_extractor = ExtractFeaturesNetwork()
        self.pi_features_extractor = ExtractFeaturesNetwork()
        self.vf_features_extractor = ExtractFeaturesNetwork()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(feature_dim=self.features_dim, device=get_config().device)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed

        features = self.extract_features(obs)##self.extract_features可认为是个输入特征预处理的函数，可自定义重写覆盖它(ExtractFeaturesNetwork())
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy



class TCNModel(nn.Module):
    def __init__(self, args):
        super(TCNModel, self).__init__()
        self.transformer_module = TransformerModule(data_length = args.data_length,
                                                    len_patch=args.len_patch,
                                                    ninp=args.ninp,
                                                    nhead=args.nhead,
                                                    nhid=args.nhid,
                                                    nlayers=args.nlayers,
                                                    dropout=args.dropout)
        self.CNN_layer = CNNLayer(data_length=args.data_length,
                                  len_patch=args.len_patch)
        self.classifier_layer = ClassifierLayer(ninp_classifier=64 * 25,
                                                num_classes=args.num_ways)

    def forward(self, x):
        out = self.transformer_module(x)
        return self.classifier_layer(self.CNN_layer(out))


class TransformerModule(nn.Module):
    """Container module with a encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, data_length = 1024, len_patch=32, ninp=200, nhead=4, nhid=200, nlayers=2, dropout=0.2, activation = "relu"):
        super(TransformerModule, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp = ninp
        self.len_patch = len_patch
        self.data_length = data_length # BN #！！！！！！未使用
        self.linear_projection = nn.Linear(len_patch, ninp)
        self.BN_layer = nn.BatchNorm1d(self.data_length // self.len_patch)#data_length // len_patch#！！！！！！未使用
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation)
        # encoder_norm = nn.LayerNorm(ninp)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # self.encoder = nn.Embedding(ntoken, ninp)
        # self.decoder = nn.Linear(ninp, ntoken)
        self.Dropout1 = nn.Dropout(dropout)
        # self.Dropout2 = nn.Dropout(dropout)

        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.weight)
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def src_patch(self,src):
        # src(input) => (N, C, L)
        # output => (C, N, L(len_patch))
        if not isinstance(src, torch.Tensor):
            try:
                src = torch.Tensor(src)
            except:
                raise TypeError("The input of data type for the model should be 'torch.Tensor'.")

        assert src.shape[1] == 1, "The model type (MODELTYPE) for the model should be '1d'."
        assert len(src.shape) == 3,"The dimension of src should be 3."
        # src=>(N, L), where N is the batch size, L is the length of each sample for the model input.
        return src[:, :, :src.shape[-1] // self.len_patch * self.len_patch]\
                    .reshape(src.shape[0],-1, self.len_patch)\
                    .transpose(1,0)

    def data_norm(self, src, norm_type='ZeroMean'):
        if norm_type == 'L2':
            return torch.div(src, torch.norm(src, p=2, dim=-1).unsqueeze(1))
        elif norm_type == 'MinMax':
            return torch.div(src - torch.min(src, dim=-1).values.unsqueeze(1),
                             (torch.max(src, dim=-1).values - torch.min(src, dim=-1).values).unsqueeze(1))
        elif norm_type == 'ZeroMean':
            return torch.div(src - torch.mean(src, dim=-1).unsqueeze(1),
                             torch.std(src, dim=-1).unsqueeze(1))
        else:
            return src

    def forward(self, src, has_mask=True):
        ############ preparation #############
        # Linear Projection of all patches for the signal sample
        # src(input) => (N, C, L), where N is the batch size, C is the number of channels (if 'modeltype' is '1d', C is 1),
        #               L is the length of each sample for the model input (1024).
        # output => (N, S, ninp), where S is the source sequence length.

        ##Input data normalization processing

        #######！！！！！！！！！！！！！！！
        # src = self.data_norm(src,norm_type='ZeroMean')
        ## norm_type: 'L2', 'MinMax', 'ZeroMean'

        if not src.dtype == torch.float32:
            src = src.to(torch.float32)

        src = self.src_patch(src) # (C, N, L(len_patch))
        ## src = self.Dropout1(self.linear_projection(self.src_patch(src)) * math.sqrt(self.ninp))
        src = self.linear_projection(src)
        # src = self.BN_layer(src.transpose(1,0)).transpose(1,0)
        # src = src * math.sqrt(self.ninp)
        ## src = self.encoder(src) * math.sqrt(self.ninp)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.pos_encoder(src)
        # return src.transpose(1, 0)
        # ############ encoder #############
        output = self.transformer_encoder(src, self.src_mask)
        # # ############ decoder #############
        # # output = self.decoder(output)
        # ############ output #############
        return output.transpose(1, 0)


class CNNLayer(nn.Module):

    def __init__(self, data_length, len_patch):
        super(CNNLayer, self).__init__()

        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(data_length // len_patch, 32, kernel_size=5, padding=1, stride=2),
        #     nn.BatchNorm1d(16, momentum=1, affine=True),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv1d(16, 16, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm1d(16, momentum=1, affine=True),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2))

        self.layer1 = nn.Sequential(
            nn.Conv1d(data_length // len_patch, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            # nn.MaxPool1d(2)
        )#!!
        # self.layer4 = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm1d(64, momentum=1, affine=True),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(2)
        # )#!!

        self.admp = nn.AdaptiveMaxPool1d(25)
        # self.admp = nn.AdaptiveMaxPool1d(8)

    def forward(self, x):
        # input => (N, C, L), where N is the batch size, C is the number of channel, L is the length of features.
        # output => (N, L_out), where L_out is the length after flattened.
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.admp(out)
        out = out.view(out.size(0), -1)
        return out


class ClassifierLayer(nn.Module):

    def __init__(self, ninp_classifier=32 * 10, num_classes=5):#128
        super(ClassifierLayer, self).__init__()
        self.ninp_classifier = ninp_classifier
        self.classifier = nn.Linear(ninp_classifier, num_classes)

    def forward(self, x):
        # input => (N, C, L), where N is the batch size, C is the number of channel, L is the length of features.
        # output => (N, L_out), where L_out is equal to num_classes.
        assert self.ninp_classifier == x.shape[-1],\
            "The input data is not equal to the input dimension of ClassifierLayer."
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
