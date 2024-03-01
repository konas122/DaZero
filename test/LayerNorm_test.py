# To test this, you should install `PyTorch` first
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print('To test this, you should install `PyTorch` first')
    os._exit(0)

import dazero.layers as L
import dazero.functions as F
from dazero import Model, Tensor


class Net(Model):
    def __init__(self, normalized_shape, gamma=None, beta=None):
        super().__init__()
        self.layer = L.LayerNorm(normalized_shape, gamma=gamma, beta=beta)

    def forward(self, inputs):
        return self.layer(inputs)


def torch_compare_layernorm(normalized_shape, inputs, gamma, beta, delta=''):
    network = nn.LayerNorm(normalized_shape, elementwise_affine=True).requires_grad_(True)
    network.double()
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(gamma)
            i.retain_grad = True
        else:
            i.data = torch.from_numpy(beta)
            i.retain_grad = True
        cnt += 1
    inputs = torch.tensor(inputs, requires_grad=True, dtype=torch.float64)
    output = network(inputs)
    delta = torch.tensor(delta)
    output.backward(delta)
    # sum = torch.sum(output) # make sure the gradient is 1
    # kk = sum.backward()
    grad_gamma = 0
    grad_beta   = 0
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            grad_gamma = i.grad
        else:
            grad_beta = i.grad
        cnt += 1
    inputs.retain_grad()
    output.retain_grad()
    k = inputs.grad
    return network, output, k, grad_gamma, grad_beta


if __name__ == "__main__":
    inputs = np.random.rand(100, 100, 30, 30).astype(np.float64)
    normalized_shape = (100, 30, 30)
    gamma = np.random.rand(100, 30, 30).astype(np.float64)
    beta = np.random.rand(100, 30, 30).astype(np.float64)

    x = Tensor(inputs)
    layernorm = Net(normalized_shape, gamma=gamma, beta=beta)
    output = layernorm(x)
    delta = np.ones(inputs.shape).astype(np.float64)
    output.backward()

    network, output_torch, gx_torch, grad_gamma_torch, grad_beta_torch = torch_compare_layernorm(normalized_shape, inputs, gamma, beta, delta)

    beta = layernorm.layer.beta.data
    gamma = layernorm.layer.gamma.data

    output, gx = output.data, x.grad.data
    gamma_delta, beta_delta = layernorm.layer.gamma.grad.data, layernorm.layer.beta.grad.data

    assert np.mean(np.abs(beta - network.bias.cpu().detach().numpy())) < 1e-6
    assert np.mean(np.abs(gamma - network.weight.cpu().detach().numpy())) < 1e-6

    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(gx - gx_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(gx - gx_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(gamma_delta - grad_gamma_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(gamma_delta - grad_gamma_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(beta_delta - grad_beta_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(beta_delta - grad_beta_torch.cpu().detach().numpy()))
    print("success")

    x = Tensor(inputs)
    output = F.layer_norm(x, normalized_shape, gamma, beta)
    output.backward()
    output, gx = output.data, x.grad.data
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(gx - gx_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(gx - gx_torch.cpu().detach().numpy()))
    print("success")

    x = Tensor(inputs)
    layernorm = Net(normalized_shape)
    output = layernorm(x)
    delta = np.ones(inputs.shape).astype(np.float64)
    output.backward()
    print("success")
