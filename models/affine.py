# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from supar.modules.mlp import MLP


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        n_proj (Optional[int]):
            If specified, applies MLP layers to reduce vector dimensions. Default: ``None``.
        dropout (Optional[float]):
            If specified, applies a :class:`SharedDropout` layer with the ratio on MLP outputs. Default: 0.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
        decompose (bool):
            If ``True``, represents the weight as the product of 2 independent matrices. Default: ``False``.
        init (Callable):
            Callable initialization method. Default: `nn.init.zeros_`.
    """

    def __init__(
        self,
        n_in: int,                                      # 输入特征的维度。
        n_out: int = 1,                                 # 输出的通道数（默认为1）。
        n_proj: Optional[int] = None,                   # 可选参数，如果指定，则对输入特征进行 MLP 层处理以降低维度。
        dropout: Optional[float] = 0,                   # 可选参数，如果指定，则在 MLP 层的输出上应用 dropout。   
        scale: int = 0,                                 # 一个缩放因子，默认为0。
        bias_x: bool = True,                            # 可选参数，如果为 True，则为输入张量 x 和 y 添加偏置项。
        bias_y: bool = True,
        decompose: bool = False,                        # 可选参数，如果为 True，则表示权重矩阵为三个独立矩阵的乘积。
        init: Callable = nn.init.zeros_                 # 初始化权重的方法，默认为 nn.init.zeros_。
    ) -> Biaffine:
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_proj = n_proj
        self.dropout = dropout
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.decompose = decompose
        self.init = init

        if n_proj is not None:
            self.mlp_x, self.mlp_y = MLP(n_in, n_proj, dropout), MLP(n_in, n_proj, dropout)
        self.n_model = n_proj or n_in
        if not decompose:
            self.weight = nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x, self.n_model + bias_y))
        else:
            self.weight = nn.ParameterList((nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x)),
                                            nn.Parameter(torch.Tensor(n_out, self.n_model + bias_y))))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.n_proj is not None:
            s += f", n_proj={self.n_proj}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.decompose:
            s += f", decompose={self.decompose}"
        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        if self.decompose:
            for i in self.weight:
                self.init(i)
        else:
            self.init(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if hasattr(self, 'mlp_x'):
            x, y = self.mlp_x(x), self.mlp_y(y)
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        if self.decompose:
            wx = torch.einsum('bxi,oi->box', x, self.weight[0])
            wy = torch.einsum('byj,oj->boy', y, self.weight[1])
            s = torch.einsum('box,boy->boxy', wx, wy)
        else:
            s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        return s.squeeze(1) / self.n_in ** self.scale


class Triaffine(nn.Module):
    r"""
    Triaffine layer for second-order scoring :cite:`zhang-etal-2020-efficient,wang-etal-2019-second`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        n_proj (Optional[int]):
            If specified, applies MLP layers to reduce vector dimensions. Default: ``None``.
        dropout (Optional[float]):
            If specified, applies a :class:`SharedDropout` layer with the ratio on MLP outputs. Default: 0.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.
        decompose (bool):
            If ``True``, represents the weight as the product of 3 independent matrices. Default: ``False``.
        init (Callable):
            Callable initialization method. Default: `nn.init.zeros_`.
    """

    def __init__(
        self,
        n_in: int,                                                  # 输入特征的维度。
        n_out: int = 1,                                             # 输出的通道数（默认为1）。
        n_proj: Optional[int] = 400,                               # 可选参数，如果指定，则对输入特征进行 MLP 层处理以降低维度。
        dropout: Optional[float] = 0.33,                               # 可选参数，如果指定，则在 MLP 层的输出上应用 dropout。
        scale: int = 0,                                             # 一个缩放因子，默认为0。
        bias_x: bool = True,                                       # 可选参数，如果为 True，则为输入张量 x 和 y 添加偏置项。
        bias_y: bool = True,
        decompose: bool = True,                                    # 可选参数，如果为 True，则表示权重矩阵为三个独立矩阵的乘积。
        init: Callable = nn.init.zeros_
    ) -> Triaffine:
        super().__init__()

        r"""
        __init__ 方法初始化了 Triaffine 模型的参数和结构。如果指定了 n_proj,它将创建三个 MLP 层(self.mlp_x、self.mlp_y 和 self.mlp_z)来处理输入特征。
        然后，模型根据 decompose 参数来选择权重的表示方式。如果 decompose 为 True,则表示权重为三个独立矩阵;否则,权重为一个张量。
        这些参数和权重将在模型的前向传播中使用。
        """

        self.n_in = n_in
        self.n_out = n_out
        self.n_proj = n_proj
        self.dropout = dropout
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.decompose = decompose
        self.init = init

        if n_proj is not None:
            self.mlp_x = MLP(n_in, n_proj, dropout)
            self.mlp_y = MLP(n_in, n_proj, dropout)
            self.mlp_z = MLP(n_in, n_proj, dropout)
        self.n_model = n_proj or n_in
        if not decompose:
            self.weight = nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x, self.n_model, self.n_model + bias_y))
        else:
            self.weight = nn.ParameterList((nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x)),
                                            nn.Parameter(torch.Tensor(n_out, self.n_model)),
                                            nn.Parameter(torch.Tensor(n_out, self.n_model + bias_y))))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.n_proj is not None:
            s += f", n_proj={self.n_proj}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.decompose:
            s += f", decompose={self.decompose}"
        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        if self.decompose:
            for i in self.weight:
                self.init(i)
        else:
            self.init(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if hasattr(self, 'mlp_x'):
            x, y, z = self.mlp_x(x), self.mlp_y(y), self.mlp_z(y)
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len, seq_len]
        if self.decompose:
            wx = torch.einsum('bxi,oi->box', x, self.weight[0])
            wz = torch.einsum('bzk,ok->boz', z, self.weight[1])
            wy = torch.einsum('byj,oj->boy', y, self.weight[2])
            s = torch.einsum('box,boz,boy->bozxy', wx, wz, wy)
        else:
            w = torch.einsum('bzk,oikj->bozij', z, self.weight)
            s = torch.einsum('bxi,bozij,byj->bozxy', x, w, y)
        return s.squeeze(1) / self.n_in ** self.scale
