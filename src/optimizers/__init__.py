import torch

from src.optimizers.bibatch_methods.ConjugateGradient import ConjugateGradient
from src.optimizers.SGDOverlapTest import SGDOverlapTest


def get_optimizer(opt, params, n_batches_per_epoch=None, train_set_len=None):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    if isinstance(opt, dict):
        opt_name = opt["name"]
        opt_dict = opt
    else:
        opt_name = opt
        opt_dict = {}

    n_batches_per_epoch = opt_dict.get("n_batches_per_epoch") or n_batches_per_epoch

    if opt_name == 'conjugate_gradient':
        opt = ConjugateGradient(
            params,
            **{key: value for key, value in opt_dict.items() if key != "name"}
        )

    elif opt_name == 'gd_overlap_test':
        opt = SGDOverlapTest(
            params,
            **{key: value for key, value in opt_dict.items() if key != "name"}
        )
    
    elif opt_name == 'adam':
        opt = torch.optim.Adam(params, **{key: value for key, value in opt_dict.items() if key != "name"})

    elif opt_name == 'sgd':
        opt = torch.optim.SGD(params, **{key: value for key, value in opt_dict.items() if key != "name"})

    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt
