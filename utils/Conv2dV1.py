import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import functional as F
from torch import __init__
from torch.nn.modules import Module

class Conv2dV1(_ConvNd):
    __doc__ = r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """ + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or an int / a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        {depthwise_separable_note}

    Note:
        {cudnn_reproducibility_note}

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """.format(**reproducibility_notes, **convolution_notes) + r"""

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_: _size_2_t,
        kernel_size_max: _size_2_t,
        n_sizes: 2,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)




    def convolution(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            stride: List[int],
            padding: List[int],
            dilation: List[int],
            transposed: bool,
            output_padding: List[int],
            groups: int,
    ):
        stride = tuple(stride)
        padding = tuple(padding)
        dilation = tuple(dilation)
        output_padding = tuple(output_padding)
        assert isinstance(groups, int)
        kwargs: ConvLayoutParams = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "transposed": transposed,
            "output_padding": output_padding,
            "groups": groups,
        }

        if len(x.get_size()) == len(weight.get_size()) - 1:
            # add batch dimension to simplify rest of function
            return L[aten.squeeze](
                convolution(L[aten.expand](x, [1, *x.get_size()]), weight, bias,
                            **kwargs),
                dim=0,
            )

        out_chan, in_chan, *kernel_shape = V.graph.sizevars.evaluate_static_shapes(
            weight.get_size()
        )
        ndim = len(kernel_shape)
        stride = pad_listlike(stride, ndim)
        padding = pad_listlike(padding, ndim)
        dilation = pad_listlike(dilation, ndim)
        output_padding = pad_listlike(output_padding, ndim)

        def channels_last_conv():
            if V.graph.layout_opt and ndim == 2:
                return True

            layout = conv_layout(x, weight, None, **kwargs)
            req_stride_order = ir.get_stride_order(
                V.graph.sizevars.size_hints(layout.stride)
            )
            return req_stride_order == ir.NHWC_STRIDE_ORDER

        autotuning_gemm = config.max_autotune or config.max_autotune_gemm

        if (
                (config.conv_1x1_as_mm or (
                        autotuning_gemm and channels_last_conv()))
                and is_ones(kernel_shape)
                and is_ones(stride)
                and is_zeros(padding)
                and is_ones(dilation)
                and not transposed
                and is_zeros(output_padding)
                and groups == 1
        ):
            return convert_1x1_conv_to_mm(x, weight, bias)

        if bias is not None and ir.get_device_type(x) != "cpu":
            # peel off the bias, cudnn is slower with it
            result = convolution(x, weight, None, **kwargs)
            return L[aten.add](
                result, L[aten.view](bias, [result.get_size()[1]] + ndim * [1])
            )

        x.realize()
        weight.realize()

        # ndim can be 1 for convolution in models such as demucs
        # TODO: check if it's beneficial to convert Conv1d to Conv2d and then
        # apply channels last.
        if V.graph.layout_opt and ndim == 2:
            V.graph.num_channels_last_conv += 1
            x = ir.ExternKernel.require_channels_last(x)
            # TODO maybe we can convert weights to channels last just once before
            # running the model.
            weight = ir.ExternKernel.require_channels_last(weight)
            layout = conv_layout(x, weight, None, **kwargs)
        else:
            layout = conv_layout(x, weight, None, **kwargs)
            req_stride_order = ir.get_stride_order(
                V.graph.sizevars.size_hints(layout.stride)
            )
            x = ir.ExternKernel.require_stride_order(x, req_stride_order)
            weight = ir.ExternKernel.require_stride_order(weight,
                                                          req_stride_order)

        ordered_kwargs_for_cpp_kernel = [
            "stride",
            "padding",
            "dilation",
            "transposed",
            "output_padding",
            "groups",
        ]
        if bias is None:
            args = [x, weight]
            kwargs["bias"] = None  # type: ignore[typeddict-unknown-key]
            ordered_kwargs_for_cpp_kernel.insert(0, "bias")
        else:
            args = [x, weight, bias]
            bias.realize()
            bias.freeze_layout()
            V.graph.sizevars.evaluate_static_shapes(bias.get_size())

        choices = [
            aten_convolution.bind(args, layout, ordered_kwargs_for_cpp_kernel,
                                  **kwargs)
        ]
        if (
                use_triton_template(layout)
                # templates only support these:
                and ndim == 2
                and is_ones(dilation)
                and not transposed
                and is_zeros(output_padding)
                # there are some odd models where this check fails (e.g. shufflenet_v2_x1_0)
                and V.graph.sizevars.statically_known_equals(in_chan,
                                                             x.get_size()[1])
        ):
            if (
                    is_ones(kernel_shape)
                    and is_ones(stride)
                    and is_zeros(padding)
                    and groups == 1
            ):
                choices.append(aten_conv1x1_via_mm.bind(args, layout))

            for cfg in conv_configs(
                    sympy_product([x.get_size()[0], *x.get_size()[2:]]),
                    out_chan,
                    in_chan,
            ):
                conv2d_template.maybe_append_choice(
                    choices,
                    (x, weight),
                    layout,
                    KERNEL_H=kernel_shape[0],
                    KERNEL_W=kernel_shape[1],
                    STRIDE_H=stride[0],
                    STRIDE_W=stride[1],
                    PADDING_H=padding[0],
                    PADDING_W=padding[1],
                    GROUPS=groups,
                    # TODO(jansel): try unroll for bigger kernels once fixed:
                    #               https://github.com/openai/triton/issues/1254
                    UNROLL=is_ones(kernel_shape),
                    ALLOW_TF32=torch.backends.cudnn.allow_tf32,
                    num_stages=cfg.num_stages,
                    num_warps=cfg.num_warps,
                    **cfg.kwargs,
                )

        return autotune_select_algorithm("convolution", choices, args, layout)
class Conv2dV1(nn.Conv2d):

    def __init__(self, image_size, in_channels, out_channels, kernel_sizes,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)

        self.image_size = image_size
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups


    def decompose_image(self, image):

        # decompose image into patches
        center_x1 = image_size // 4
        center_x2 = image_size // 4 * 3
        center_y1 = image_size // 4
        center_y2 = image_size // 4 * 3

        center = image[..., center_x1:center_x2, center_y1:center_y2]
        patches = image.unfold(2, self.kernel_sizes[0], self.stride).unfold(3, self.kernel_sizes[1], self.stride)

        # reshape patches into columns
        cols = patches.reshape(patches.shape[0], patches.shape[1], -1)

        return cols
