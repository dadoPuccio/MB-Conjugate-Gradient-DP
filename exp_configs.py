from Utils import logs_utils

opt_bibatch_ls = ["conjugate_gradient"]

opt_overlap_test = ["gd_overlap_test"]

custom_opt_list = opt_bibatch_ls + opt_overlap_test

non_convex_exp = [
        {"name": "sgd", "lr": 1e-3, "momentum": 0.9}, # SGD + Momentum
        {"name": "adam", "lr": 1e-5}, # Adam
]

# non_convex_exp = []
for cg_mode in ['FR']: # 'PPR', 'HS'
    for eta_mode in ['polyak']: # 'constant', 'vaswani'
        for dir_recovery_mode in ['clip']: # 'grad
            for max_eta in [10]: # 10e3, 10e6
                for c_p in [1.]:
                    
                    non_convex_exp.append({"name": "conjugate_gradient", "cg_mode": cg_mode,
                                            "use_backtrack_heuristic": False,
                                            "use_line_search": True, "zhang_xi": 1, "eps": 0., "eta_0": 0.1,
                                            "c_p": c_p, "max_eta": max_eta, "eta_mode": eta_mode,
                                            "max_beta": 1.5, "dir_recovery_mode": dir_recovery_mode})
                            
convex_exp = [
        {"name": "sgd", "lr": 1e-2, "momentum": 0.9}, # SGD + Momentum
        {"name": "adam", "lr": 1e-3}, # Adam
]

for cg_mode in ['FR']: # 'HS', 'PPR'
    for eta_mode in ['polyak']: # 'constant', 'vaswani'
        for dir_recovery_mode in ['clip']: # 'grad', 'inv', 'qps'
            for c_p in [1.]:
                for zhang_xi in [1]:
                    
                    convex_exp.append({"name": "conjugate_gradient", "cg_mode": cg_mode, "c": 0.5, 
                                        "use_backtrack_heuristic": False,
                                        "use_line_search": True, "zhang_xi": zhang_xi, "eps": 0., "eta_0": 0.1,
                                        "c_p": c_p, "max_eta": 1e4, "eta_mode": eta_mode,
                                        "max_beta": 1.5, "dir_recovery_mode": dir_recovery_mode})

mnist_overlap_exp = [
    {'name': 'gd_overlap_test', 'lr': 0.01, 'overlap_percentages': [0, 25, 50, 75, 100], 'beta': 0.9, 'mode': 'past_info'},
]

ijcnn_overlap_exp = [
    {'name': 'gd_overlap_test', 'lr': 1000, 'overlap_percentages': [0, 25, 50, 75, 100], 'beta': 0.9, 'mode': 'past_info'},
]


long_run = 10
batch_size = [128, 512] 

batch_size_convex = [128, 512] 
convex_run = 10

many_runs = [0, 1]
overlap_batches = [False, True]

# Experiments definition
EXP_GROUPS = {

    "mnist_mlp_overlap_test": {
                  "dataset":["mnist"],
                  "model":["mlp"],
                  "not_save_pth": True,
                  "loss_func": ["softmax_loss"],
                  "opt": mnist_overlap_exp,
                  "acc_func":["softmax_accuracy"],
                  "batch_size": batch_size,
                  "max_epoch":[100],
                  "runs": [0],
                  "overlap_batches": True, 
    },

    "ijcnn_overlap_test": {
                "dataset": ["ijcnn"],
                "model": ["logistic"],
                "loss_func": ["logistic_loss"],
                "acc_func": ["logistic_accuracy"],
                "opt": ijcnn_overlap_exp,
                "batch_size": batch_size_convex,
                "max_epoch": [100],
                "runs": [0],
                "overlap_batches": True, 
    },

    "mnist_mlp": {"dataset":["mnist"],
                  "model":["mlp"],
                  "not_save_pth": True,
                  "loss_func": ["softmax_loss"],
                  "opt": non_convex_exp,
                  "acc_func":["softmax_accuracy"],
                  "batch_size": batch_size,
                  "max_epoch":[long_run],
                  "overlap_batches": overlap_batches, 
                  "runs":many_runs},

    "fashion_cnn":{"dataset":["fashion"],
                "model":["cnn"],
                "loss_func": ["softmax_loss"],
                "opt": non_convex_exp,
                "acc_func":["softmax_accuracy"],
                "batch_size": batch_size,
                "max_epoch":[long_run],
                "overlap_batches": overlap_batches, 
                "runs": many_runs},

    "cifar10_resnet18_bn":{
                "dataset":["cifar10"],
                "model":["resnet18_bn"],
                "loss_func": ["softmax_loss"],
                "opt": non_convex_exp,
                "acc_func":["softmax_accuracy"],
                "batch_size": batch_size,
                "max_epoch":[long_run],
                "overlap_batches": overlap_batches,
                "runs": many_runs
    },

    "mushrooms": {"dataset": ["mushrooms"],
                    "model": ["logistic"],
                    "loss_func": ['logistic_loss'],
                    "acc_func": ["logistic_accuracy"],
                    "opt": convex_exp,
                    "batch_size": batch_size_convex,
                    "max_epoch": [convex_run],
                    "overlap_batches": overlap_batches, 
                    "runs": many_runs},

    "ijcnn": {"dataset": ["ijcnn"],
                "model": ["logistic"],
                "loss_func": ['logistic_loss'],
                "acc_func": ["logistic_accuracy"],
                "opt": convex_exp,
                "batch_size": batch_size_convex,
                "max_epoch": [convex_run],
                "overlap_batches": overlap_batches, 
                "runs": many_runs},

    "rcv1": {"dataset": ['rcv1'],
            "model": ["logistic"],
            "loss_func": ['logistic_loss'],
            "acc_func": ["logistic_accuracy"],
            "opt": convex_exp,
            "batch_size": batch_size_convex,
            "max_epoch": [convex_run],
            "overlap_batches": overlap_batches, 
            "runs": many_runs},

}

EXP_GROUPS = {k:logs_utils.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}
