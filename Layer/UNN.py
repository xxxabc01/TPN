import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class BNN_layer(Module):
    r"""
    Applies Bayesian Linear

    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.


    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        self.uncertainty = nn.Linear(in_features, 1)

        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def kl_loss(self, mu_0, log_sigma_0, mu_1, log_sigma_1):
        """
        An method for calculating KL divergence between two Normal distribtuion.

        Arguments:
            mu_0 (Float) : mean of normal distribution.
            log_sigma_0 (Float): log(standard deviation of normal distribution).
            mu_1 (Float): mean of normal distribution.
            log_sigma_1 (Float): log(standard deviation of normal distribution).

        """
        kl = log_sigma_1 - log_sigma_0 + \
             (torch.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2) / (2 * math.exp(log_sigma_1) ** 2) - 0.5
        return kl.sum()

    def bayesian_kl_loss(self, reduction='mean', last_layer_only=False):
        """
        An method for calculating KL divergence of whole layers in the model.


        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'mean'``: the sum of the output will be divided by the number of
                elements of the output.
                ``'sum'``: the output will be summed.
            last_layer_only (Bool): True for return only the last layer's KL divergence.

        """
        # kl_sum = torch.FloatTensor([0]).to(device)
        # n = torch.FloatTensor([0]).to(device)
        kl_sum = 0
        n = 0


        kl = self.kl_loss(self.weight_mu, self.weight_log_sigma, self.prior_mu, self.prior_log_sigma)
        kl_sum += kl
        n += len(self.weight_mu.view(-1))

        if self.bias:
            kl = self.kl_loss(self.bias_mu, self.bias_log_sigma, self.prior_mu, self.prior_log_sigma)
            kl_sum += kl
            n += len(self.bias_mu.view(-1))

        if last_layer_only or n == 0:
            return kl

        if reduction == 'mean':
            return kl_sum / n
        elif reduction == 'sum':
            return kl_sum
        else:
            raise ValueError(reduction + " is not valid")

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps

        if self.bias:
            if self.bias_eps is None:
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else:
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else:
            bias = None

        self.lamda = torch.sigmoid(self.uncertainty(torch.exp(self.weight_log_sigma)))
        # print(self.lamda)
        return self.lamda.flatten(), F.linear(input, weight, bias)

        # self.lamda = torch.mean(torch.exp(self.weight_log_sigma))
        # print(self.lamda)
        # return self.lamda, F.linear(input, weight, bias)


    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu,
                                                                                              self.prior_sigma,
                                                                                              self.in_features,
                                                                                              self.out_features,
                                                                                              self.bias is not None)

