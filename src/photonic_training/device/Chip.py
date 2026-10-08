import math
from abc import ABC
from collections.abc import Iterable
from copy import deepcopy

import torch
import torch.nn as nn


def complex_mm(mat1, mat2):
    return torch.stack((mat1[0] @ mat2[0] - mat1[1] @ mat2[1],
                        mat1[1] @ mat2[0] + mat1[0] @ mat2[1]))


class Buffer(torch.Tensor):
    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __repr__(self):
        return "Buffer containing:\n" + super(Buffer, self).__repr__()


class BoundedParameter(nn.Parameter):
    def __new__(cls, data=None, bounds=(0, 1), requires_grad=True):
        if data is None:
            data = torch.Tensor()
        if not torch.is_tensor(data):
            raise TypeError("argument 'data' must be Tensor, not %s" % type(data).__name__)
        # check bounds
        try:
            a, b = bounds
        except ValueError:
            raise ValueError("bounds should be a tuple with length 2")
        if b <= a:
            raise ValueError("bounds should be a tuple with length 2 and with bounds[0] < bounds[1]")
        if (data < a).any() or (data > b).any():
            raise ValueError("some of your data is outside the specified bounds: [%i, %i]" % (a, b))
        new = torch.Tensor._make_subclass(cls, cls._inverse_sigmoid(data.data, (a, b)), requires_grad)
        new.bounds = (float(a), float(b))
        return new

    @staticmethod
    def _sigmoid(weights, bounds):
        a, b = bounds
        scaled_data = torch.sigmoid(weights)
        data = (b - a) * scaled_data + a
        return data

    @staticmethod
    def _inverse_sigmoid(data, bounds):
        a, b = bounds
        scaled_data = (data - a) / (b - a)
        weights = -torch.log(1 / scaled_data - 1)
        return weights

    def __repr__(self):
        tensor_repr = torch.Tensor.__repr__(self._sigmoid(self.data, self.bounds))
        return "BoundedParameter in [%.2f, %.2f] representing:\n" % self.bounds + tensor_repr

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = torch.Tensor._make_subclass(cls, self.data.clone(), self.requires_grad)
        new.bounds = deepcopy(self.bounds, memo)
        return new


class Module(nn.Module, ABC):
    def __init__(self):
        super(Module, self).__init__()
        self.device = torch.device("cpu")

    def __setattr__(self, attr, value):
        if isinstance(value, Buffer):
            self.register_buffer(attr, value)
        if isinstance(value, BoundedParameter):
            _attr = "_" + attr
            self.register_parameter(_attr, value)
            value_property = property(lambda self: self._parameters[_attr]._sigmoid(
                self._parameters[_attr], self._parameters[_attr].bounds))
            self.__class__ = type(self.__class__.__name__, (self.__class__,), {attr: value_property})
        else:
            super(Module, self).__setattr__(attr, value)

    @property
    def is_cuda(self):
        return False if self.device.type == "cpu" else True

    def to(self, *args, **kwargs):
        new = super(Module, self).to(*args, **kwargs)
        for k, v in self._modules.items():
            self._modules[k] = v.to(*args, **kwargs)
        try:  # torch < 1.5.1
            new.device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        except ValueError:  # torch >= 1.5.1
            new.device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        return new

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self, device=None):
        if device is None:
            device = "cuda:0"
        elif isinstance(device, int):
            device = "cuda:%i" % device
        return self.to(device=device)


class Component(Module, ABC):
    def __init__(self, in_dim, out_dim, vol_num=0):
        super(Component, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.vol_num = vol_num

    def set_s(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.set_s(*args, **kwargs)

    def __repr__(self):
        return '{}({}, {}, vol_num={})'.format(self._get_name(), self.in_dim, self.out_dim, self.vol_num)


class Network(Module, ABC):
    def __init__(self):
        super(Network, self).__init__()

    def set_s(self, vol, mode='phase'):
        out_channel, in_channel, vol_num = vol.shape
        assert vol_num == self.vol_num
        first = last = next(self.children())
        s = first.set_s() if first.vol_num == 0 else first.set_s(vol[:, :, :first.vol_num], mode=mode)
        vol_cnt = first.vol_num
        for last in self.children():
            if last is first:
                continue
            temp = last.set_s() if last.vol_num == 0 else last.set_s(vol[:, :, vol_cnt:vol_cnt + last.vol_num],
                                                                     mode=mode)
            s = complex_mm(temp, s)
            vol_cnt += last.vol_num
        return s

    def forward(self, *args, **kwargs):
        return self.set_s(*args, **kwargs)

    @property
    def in_dim(self):
        return next(self.children()).in_dim

    @property
    def out_dim(self):
        last = next(self.children())
        for last in self.children():
            pass
        return last.out_dim

    @property
    def vol_num(self):
        return sum(module.vol_num for module in self.children())


class Chip(Network, ABC):
    def __init__(self):
        super(Chip, self).__init__()
        # MZI1
        self.mzi1_phi = PS(32, range(0, 32, 1))
        self.mzi1_bs1 = BS(32, range(0, 32, 2))
        self.mzi1_theta = PS(32, range(0, 32, 2))
        self.mzi1_bs2 = BS(32, range(0, 32, 2))
        self.cross1 = FFTCross(32, 1)
        # MZI2
        self.mzi2_phi = PS(32, range(0, 32, 2))
        self.mzi2_bs1 = BS(32, range(0, 32, 2))
        self.mzi2_theta = PS(32, range(0, 32, 2))
        self.mzi2_bs2 = BS(32, range(0, 32, 2))
        self.cross2 = FFTCross(32, 2)
        # MZI3
        self.mzi3_phi = PS(32, range(0, 32, 2))
        self.mzi3_bs1 = BS(32, range(0, 32, 2))
        self.mzi3_theta = PS(32, range(0, 32, 2))
        self.mzi3_bs2 = BS(32, range(0, 32, 2))
        self.cross3 = FFTCross(32, 3)
        # MZI4
        self.mzi4_phi = PS(32, range(0, 32, 2))
        self.mzi4_bs1 = BS(32, range(0, 32, 2))
        self.mzi4_theta = PS(32, range(0, 32, 2))
        self.mzi4_bs2 = BS(32, range(0, 32, 2))
        self.cross4 = FFTCross(32, 4)
        # MZI5
        self.mzi5_phi = PS(32, range(0, 32, 2))
        self.mzi5_bs1 = BS(32, range(0, 32, 2))
        self.mzi5_theta = PS(32, range(0, 32, 2))
        self.mzi5_bs2 = BS(32, range(0, 32, 2))
        self.ps_out = PS(32, range(0, 32, 1))


class BS(Component, ABC):
    def __init__(self, dim, bs_list, coupling=0.5, loss=0.):
        assert isinstance(bs_list, Iterable)
        bs_num = sum(1 for _ in bs_list)
        assert min(i for i in bs_list) >= 0 and max(i for i in bs_list) <= dim - 1
        super(BS, self).__init__(dim, dim, 0)
        self.bs_list = bs_list
        self.coupling = BoundedParameter(
            torch.FloatTensor(coupling * torch.ones(bs_num)),
            bounds=(0.4, 0.6),
            requires_grad=False)
        self.loss = BoundedParameter(
            torch.FloatTensor(loss * torch.ones(bs_num)),
            bounds=(0, 1),
            requires_grad=False)

    def set_s(self):
        t = (1 - self.coupling) ** 0.5 * 10 ** (-0.05 * self.loss)
        k = self.coupling ** 0.5 * 10 ** (-0.05 * self.loss)
        s = torch.zeros(2, 1, 1, self.out_dim, self.in_dim, dtype=torch.float32, device=self.device)
        s[0, :, :, range(self.out_dim), range(self.in_dim)] = 1
        s[0, :, :, self.bs_list, self.bs_list] \
            = s[0, :, :, [i + 1 for i in self.bs_list], [i + 1 for i in self.bs_list]] = t
        s[1, :, :, [i + 1 for i in self.bs_list], self.bs_list] \
            = s[1, :, :, self.bs_list, [i + 1 for i in self.bs_list]] = k
        return s


class PS(Component, ABC):
    def __init__(self, dim, ps_list, gamma=0.15, phase0=0.):
        assert isinstance(ps_list, Iterable)
        ps_num = sum(1 for _ in ps_list)
        assert min(i for i in ps_list) >= 0 and max(i for i in ps_list) <= dim - 1
        assert min(i - j for i, j in zip(ps_list[1:], ps_list[:-1])) >= 1
        super(PS, self).__init__(dim, dim, ps_num)
        self.ps_list = ps_list
        self.gamma = nn.Parameter(
            torch.FloatTensor(gamma * torch.ones(ps_num)),
            requires_grad=False)
        self.phase0 = nn.Parameter(
            torch.FloatTensor(phase0 * torch.ones(ps_num)),
            requires_grad=False)

    def set_s(self, vol, mode='nophase'):
        out_channel, in_channel, vol_num = vol.shape
        assert vol_num == self.vol_num
        phase = torch.zeros(out_channel, in_channel, self.in_dim, dtype=torch.float32, device=self.device)
        if mode != 'phase':
            vol = self.gamma * vol ** 2 + self.phase0
            # print("yes,no pahse")
        phase[:, :, self.ps_list] += vol
        s = torch.zeros(2, out_channel, in_channel, self.out_dim, self.in_dim, dtype=torch.float32, device=self.device)
        s[0, :, :, range(self.out_dim), range(self.in_dim)] = torch.cos(phase)
        s[1, :, :, range(self.out_dim), range(self.in_dim)] = torch.sin(phase)
        return s


class FFTCross(Component, ABC):
    def __init__(self, dim, layer, loss=0.):
        assert 1 <= layer <= math.log2(dim) - 1
        super(FFTCross, self).__init__(dim, dim, 0)
        self.layer = layer
        self.loss = BoundedParameter(
            torch.FloatTensor(loss * torch.ones(dim)),
            bounds=(0, 1),
            requires_grad=False)

    def set_s(self):
        t = 10 ** (-0.05 * self.loss)
        s = torch.zeros(2, 1, 1, self.out_dim, self.in_dim, dtype=torch.float32, device=self.device)
        out_list = []
        for i in range(self.out_dim // 2 ** (self.layer + 1)):
            out_list += range(2 ** (self.layer + 1) * i, 2 ** (self.layer + 1) * (i + 1), 2)
            out_list += range(2 ** (self.layer + 1) * i + 1, 2 ** (self.layer + 1) * (i + 1), 2)
        s[0, :, :, out_list, range(self.in_dim)] = t
        return s
