import torch
from torch import nn
from torch.nn import functional as F

from .base_networks.resnet_no_batch_norm import ResNet18_bn


def get_model(model_name, train_set=None, model_args=None, features_dim=None):
    if model_name in ["softmax"]:
        model = Mlp_model(input_size=train_set[0][0].shape[0], hidden_sizes=[], n_classes=2, bias=False)

    if model_name in ["linear", "logistic"]:
        model = LinearRegression(input_dim=train_set[0][0].shape[0], output_dim=1, bias=False)

    if model_name == "mlp":
        model = Mlp(n_classes=10, dropout=False)
    
    elif model_name == "mlp_double":
        model = Mlp(n_classes=10, hidden_sizes=[512, 512, 256], dropout=False)
    
    elif model_name == "cnn":
        model = Cnn(input_channels=1, n_classes=10)
    
    elif model_name == "resnet18_bn":
        model = ResNet18_bn(num_classes=10, batch_norm=False, conf_params={"c": 2.0, "init": [0, 1, 2, 3], "spp": [0, 1, 2, 3]}, fan="fan_out", hooks=False)

    return model

# =====================================================
# Linear Network
class LinearNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True):
        super().__init__()

        # iterate averaging:
        self._prediction_params = None

        self.input_size = input_size
        if output_size:
            self.output_size = output_size
            self.squeeze_output = False
        else :
            self.output_size = 1
            self.squeeze_output = True

        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size, bias=bias)
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size, bias=bias)

    def forward(self, x):
        '''
            x: The input patterns/features.
        '''
        x = x.view(-1, self.input_size)
        out = x

        for layer in self.hidden_layers:
            Z = layer(out)
            # no activation in linear network.
            out = Z

        logits = self.output_layer(out)
        if self.squeeze_output:
            logits = torch.squeeze(logits)

        return logits

def Mlp_model(input_size=784, hidden_sizes=[512, 256], n_classes=10, bias=True, dropout=False):
    modules = []
    if len(hidden_sizes) == 0:
        modules.append(nn.Linear(input_size, n_classes, bias=bias))
    else:
        for i, layer in enumerate(hidden_sizes):
            if i == 0:
                modules.append(nn.Linear(input_size, layer, bias=bias))
            else:
                modules.append(nn.Linear(hidden_sizes[i-1], layer, bias=bias))

            modules.append(nn.ReLU())
            if dropout:
                modules.append(nn.Dropout(p=0.5))

        modules.append(nn.Linear(hidden_sizes[-1], n_classes, bias=bias))

    return nn.Sequential(*modules)

# =====================================================
# Logistic
class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

# =====================================================
# MLP
class Mlp(nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[512, 256],
                 n_classes=10,
                 bias=True, dropout=False):
        super().__init__()

        self.dropout=dropout
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], n_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = F.relu(Z)

            if self.dropout:
                out = F.dropout(out, p=0.5)

        logits = self.output_layer(out)

        return logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class Cnn(nn.Module):

    def __init__(self, input_channels=1, n_classes=10, activation="relu", n_conv_filters=32, fc_size=512):

        assert activation in ["relu", "sigmoid", "elu", "silu"]

        super().__init__()

        self.weights = nn.ParameterList()

        # CNN 1
        w1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((n_conv_filters, input_channels, 5, 5))))
        b1 = nn.Parameter(torch.zeros((n_conv_filters)))

        self.weights.append(w1)
        self.weights.append(b1)

        # CNN 2
        w2 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((n_conv_filters, n_conv_filters, 5, 5))))
        b2 = nn.Parameter(torch.zeros(n_conv_filters))

        self.weights.append(w2)
        self.weights.append(b2)

        # CNN 3
        w3 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((n_conv_filters, n_conv_filters, 5, 5))))
        b3 = nn.Parameter(torch.zeros(n_conv_filters))

        self.weights.append(w3)
        self.weights.append(b3)

        # FC
        w4 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(fc_size, n_conv_filters * 36)))
        b4 = nn.Parameter(torch.zeros(fc_size))
        
        self.weights.append(w4)
        self.weights.append(b4)

        # Linear
        w5 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(n_classes, fc_size)))
        b5 = nn.Parameter(torch.zeros(1, n_classes))

        self.weights.append(w5)
        self.weights.append(b5)

        if activation == "relu":
            self.activation = nn.functional.relu
            
        elif activation == "elu":
            self.activation = nn.functional.elu
            
        elif activation == "silu":
            self.activation = nn.functional.silu

        elif activation == "sigmoid":
            self.activation = nn.functional.sigmoid        


    def forward(self, x, weights=None):

        if weights is None:
            weights = self.weights
        
        # CNN 1
        x = nn.functional.conv2d(x, weights[0], weights[1], stride=1)
        x = self.activation(x)

        # CNN 2
        x = nn.functional.conv2d(x, weights[2], weights[3], stride=2)
        x = self.activation(x)

        # CNN 3
        x = nn.functional.conv2d(x, weights[4], weights[5], stride=1)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)

        # FC
        x = nn.functional.linear(x, weights[6], weights[7])
        x = self.activation(x)

        # Linear
        x = nn.functional.linear(x, weights[8], weights[9])

        return x
    
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

