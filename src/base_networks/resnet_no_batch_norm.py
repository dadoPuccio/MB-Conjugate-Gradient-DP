import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

def signal_propagation_metrics(activation, dim=[0, 1, 2, 3]):
    with torch.no_grad():
        mean = activation.mean(dim=dim).pow(2).mean()
        var = activation.var(dim=dim).mean()
    return mean.item(), var.item()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conf_params, stride=1, batch_norm=False, fan="fan_in", hooks=False):
        super(BasicBlock, self).__init__()

        self.conf_params = conf_params

        self.bn1 = nn.BatchNorm2d(in_planes) if batch_norm else nn.Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0, bias=False)

        self.dummy = nn.Identity()

        self.pad = torch.nn.ReplicationPad2d(padding=1)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.forward_hooks = []
        self.backward_hooks = []
        if hooks:
            # forward
            self.conv2.register_forward_hook(lambda m, i, o: self.m_fw_hook(m, i, o))
            self.shortcut.register_forward_hook(lambda m, i, o: self.m_fw_hook(m, i, o))
            self.dummy.register_forward_hook(lambda m, i, o: self.m_fw_hook(m, i, o))
            # backward
            self.dummy.register_full_backward_hook(lambda m, i, o: self.m_bw_hook(m, i, o))

        # initialization
        kaiming_normal_residual_conv_(self.conv1.weight, mode=fan, coeff=self.conf_params['c'], dim=self.conf_params["init"], fan_expansion=1.0)
        kaiming_normal_residual_conv_(self.conv2.weight, mode=fan, coeff=self.conf_params['c'], dim=self.conf_params["init"], fan_expansion=1.0)
        if isinstance(self.shortcut, nn.Conv2d):
            xavier_normal_residual_conv_(self.shortcut.weight, mode=fan, dim=self.conf_params["init"], fan_expansion=1.0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(x))
        out = self.conv1(self.pad(out))
        out = F.relu(self.bn2(out))
        out = self.conv2(self.pad(out))
        activation = self.dummy(out + shortcut)
        return activation

    def m_fw_hook(self, module, tensor_in, tensor_out, t=""):
        out_mean, out_var = signal_propagation_metrics(tensor_out, dim=self.conf_params["spp"])
        self.forward_hooks.append({"module": str(module).split('(')[0],
                                   "mean": out_mean,
                                   "var": out_var})

    def m_bw_hook(self, module, grad_in, grad_out):
        out_mean, out_var = signal_propagation_metrics(grad_out[0], dim=self.conf_params["spp"])
        self.backward_hooks.append({"module": str(module).split('(')[0],
                                    "mean": out_mean,
                                    "var": out_var})

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conf_params, stride=1, batch_norm=False, fan="fan_in", hooks=False):
        super(Bottleneck, self).__init__()

        self.conf_params = conf_params

        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm2d(in_planes) if self.batch_norm else nn.Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes) if self.batch_norm else nn.Identity()
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.dummy = nn.Identity()

        self.pad = torch.nn.ReplicationPad2d(padding=1)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.forward_hooks = []
        self.backward_hooks = []
        if hooks:
            # forward
            self.conv3.register_forward_hook(lambda m, i, o: self.m_fw_hook(m, i, o))
            self.shortcut.register_forward_hook(lambda m, i, o: self.m_fw_hook(m, i, o))
            self.dummy.register_forward_hook(lambda m, i, o: self.m_fw_hook(m, i, o))
            # backward
            self.dummy.register_full_backward_hook(lambda m, i, o: self.m_bw_hook(m, i, o))

        # initialization
        kaiming_normal_residual_conv_(self.conv1.weight, mode=fan, coeff=self.conf_params['c'], dim=self.conf_params["init"], fan_expansion=1.0)
        kaiming_normal_residual_conv_(self.conv2.weight, mode=fan, coeff=self.conf_params['c'], dim=self.conf_params["init"], fan_expansion=1.0)
        kaiming_normal_residual_conv_(self.conv3.weight, mode=fan, coeff=self.conf_params['c'], dim=self.conf_params["init"], fan_expansion=1.0)
        if isinstance(self.shortcut, nn.Conv2d):
            xavier_normal_residual_conv_(self.shortcut.weight, mode=fan, dim=self.conf_params["init"], fan_expansion=1.0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(self.pad(out))
        out = F.relu(self.bn3(out))
        out = self.conv3(out)
        activation = self.dummy(out + shortcut)
        return activation

    def m_fw_hook(self, module, tensor_in, tensor_out):
        out_mean, out_var = signal_propagation_metrics(tensor_out, dim=self.conf_params["spp"])
        self.forward_hooks.append({"module": str(module).split('(')[0],
                                   "mean": out_mean,
                                   "var": out_var})

    def m_bw_hook(self, module, grad_in, grad_out):
        out_mean, out_var = signal_propagation_metrics(grad_out[0], dim=self.conf_params["spp"])
        self.backward_hooks.append({"module": str(module).split('(')[0],
                                   "mean": out_mean,
                                   "var": out_var})


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, conf_params, num_classes=10, batch_norm=False, fan="fan_in", hooks=False):
        super(ResNet, self).__init__()

        self.in_planes = 64
        self.batch_norm = batch_norm
        self.conf_params = conf_params

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, conf_params=conf_params, fan=fan, hooks=hooks)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, conf_params=conf_params, fan=fan, hooks=hooks)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, conf_params=conf_params, fan=fan, hooks=hooks)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, conf_params=conf_params, fan=fan, hooks=hooks)
        self.classifier = nn.Linear(512 * block.expansion, num_classes, bias=True)

        self.pad = torch.nn.ReplicationPad2d(padding=1)

        self.forward_hooks = []
        self.backward_hooks = []
        if hooks:
            # forward
            self.conv1.register_forward_hook(lambda m, i, o: self.m_fw_hook(m, i, o))
            # backward
            self.conv1.register_full_backward_hook(lambda m, i, o: self.m_bw_hook(m, i, o))

        # initialization
        xavier_normal_residual_conv_(self.conv1.weight, mode=fan, dim=conf_params["init"], fan_expansion=1.0)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _make_layer(self, block, planes, num_blocks, stride, conf_params, fan, hooks):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride,
                                conf_params=conf_params,
                                batch_norm=self.batch_norm,
                                fan=fan, hooks=hooks))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(self.pad(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.layer4(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def m_fw_hook(self, module, tensor_in, tensor_out):
        out_mean, out_var = signal_propagation_metrics(tensor_out, dim=self.conf_params["spp"])
        self.forward_hooks.append({"module": str(module).split('(')[0],
                                   "mean": out_mean,
                                   "var": out_var})

    def m_bw_hook(self, module, grad_in, grad_out):
        out_mean, out_var = signal_propagation_metrics(grad_out[0], dim=self.conf_params["spp"])
        self.backward_hooks.append({"module": str(module).split('(')[0],
                                   "mean": out_mean,
                                   "var": out_var})

    def get_hooks(self):
        forward = self.forward_hooks
        backward = self.backward_hooks
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                forward.extend(block.forward_hooks)
                backward.extend(block.backward_hooks)
        return {"fw": forward, "bw": backward}

def ResNet18_bn(num_classes, conf_params, fan="fan_in", hooks=False, batch_norm=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, batch_norm=batch_norm, conf_params=conf_params, hooks=hooks, fan=fan)

# def ResNet34(num_classes, conf_params, fan="fan_in", hooks=False, batch_norm=None):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, batch_norm=batch_norm, conf_params=conf_params, hooks=hooks, fan=fan)

# def ResNet50(num_classes, conf_params, fan="fan_in", hooks=False, batch_norm=None):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, batch_norm=batch_norm, conf_params=conf_params, hooks=hooks, fan=fan)

# def ResNet101(num_classes, conf_params, fan="fan_in", hooks=False, batch_norm=None):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, batch_norm=batch_norm, conf_params=conf_params, hooks=hooks, fan=fan)

# def ResNet152(num_classes, conf_params, fan="fan_in", hooks=False, batch_norm=None):
#     return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, batch_norm=batch_norm, conf_params=conf_params, hooks=hooks, fan=fan)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_normal_residual_conv_(tensor, mode="fan_in", fan_expansion=2.0, coeff=2, dim=[0, 1, 2, 3]):
    fan = _calculate_correct_fan(tensor, mode)
    std = coeff / (fan * fan_expansion)
    std = math.sqrt(std)
    with torch.no_grad():
        weights = tensor.normal_(0, std)
        weights = tensor.div_(weights.std(dim=dim, keepdim=True))
        weights = tensor.mul_(std)
        weights = tensor.sub_(weights.mean(dim=dim, keepdim=True))
    return weights, fan

def xavier_normal_residual_conv_(tensor, mode="fan_in", fan_expansion=2.0, dim=[0, 1, 2, 3]):
    fan = _calculate_correct_fan(tensor, mode)
    std = 1.0 / (fan * fan_expansion)
    std = math.sqrt(std)
    with torch.no_grad():
        weights = tensor.normal_(0, std)
        weights = tensor.div_(weights.std(dim=dim, keepdim=True))
        weights = tensor.mul_(std)
        weights = tensor.sub_(weights.mean(dim=dim, keepdim=True))
    return weights, fan