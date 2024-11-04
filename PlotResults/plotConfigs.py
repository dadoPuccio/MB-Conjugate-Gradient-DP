
def configs_wrapper(plot_id, problem_name, batch_size):

    zhang_xi = 1

    if plot_id == 0:

        custom_filter_convex = {
                    "conjugate_gradient": {"zhang_xi": zhang_xi, 
                                        "eta_mode": "polyak", # !!!
                                        "cg_mode": ['FR', 'PPR', 'HS'],
                                        "use_backtrack_heuristic": False,
                                        "dir_recovery_mode": "clip" # !!!
                                        },
                    }

        custom_filter_non_convex = {
                        "conjugate_gradient": {"zhang_xi": zhang_xi, 
                                                "eta_mode": "polyak", # !!!
                                                "cg_mode": ['FR', 'HS', 'PPR'],
                                                "use_backtrack_heuristic": False,
                                                "dir_recovery_mode": "clip", # !!!
                                            },
                    }
        
        colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
        markers = ['.', ',', 'o'] 

        
    elif plot_id == 1:
        custom_filter_convex = {
                    "conjugate_gradient": {"zhang_xi": zhang_xi, 
                                        "eta_mode": ["polyak", "vaswani", "constant"], 
                                        "cg_mode": 'FR', # !!!
                                        "use_backtrack_heuristic": False,
                                        "dir_recovery_mode": "clip" # !!!
                                        },
                    }

        custom_filter_non_convex = {
                        "conjugate_gradient": {"zhang_xi": zhang_xi, 
                                            "eta_mode": ["polyak", "vaswani", "constant"], 
                                            "cg_mode": 'FR', # !!!
                                            "use_backtrack_heuristic": False,
                                            "dir_recovery_mode": "clip", # !!!
                                            },
                    }
        
        colors = ['#d62728', '#2ca02c', '#9467bd']
        markers = ['v', '.', '^']

    
    elif plot_id == 2:
        custom_filter_convex = {
                    "conjugate_gradient": {"zhang_xi": zhang_xi, 
                                        "eta_mode": "polyak",
                                        "cg_mode": 'FR', #['FR', 'PPR', 'HS'],
                                        "use_backtrack_heuristic": False,
                                        "dir_recovery_mode": ["grad", "inv", "qps", "clip"]
                                        },
                    }

        custom_filter_non_convex = {
                        "conjugate_gradient": {"zhang_xi": zhang_xi, 
                                            "eta_mode": "polyak",
                                            "cg_mode": 'FR', #['FR', 'PPR', 'HS'],
                                            "use_backtrack_heuristic": False,
                                            "dir_recovery_mode": ["grad", "inv", "qps", "clip"]
                                            },
                    }
        
        colors = ['#2ca02c','#8c564b', '#e377c2', '#ffbf00']
        markers = ['.', '<', '>', '1']

    
    elif plot_id == 3:
        
        sgd_overlap_value = True
        adam_overlap_value = True 

        custom_filter_convex = {
                    "sgd": {'lr': 0.01, 'momentum': 0.9, "overlap_batches": sgd_overlap_value},
                    "adam": {"lr": 0.001, "overlap_batches": adam_overlap_value},
                    "conjugate_gradient": {"zhang_xi": 1, 
                                           "eta_mode": "polyak", 
                                           "cg_mode": 'FR', 
                                           "use_backtrack_heuristic": False,
                                           "dir_recovery_mode": "clip"
                                        },
                    }

        custom_filter_non_convex = {
                    "sgd": {'lr': 0.001, 'momentum': 0.9, "overlap_batches": sgd_overlap_value}, 
                    "adam": {"lr": 0.00001, "overlap_batches": adam_overlap_value},
                    "conjugate_gradient": {"zhang_xi": 1, 
                                    "eta_mode": "polyak", 
                                    "cg_mode": 'FR',#  'HS'],
                                    "use_backtrack_heuristic": False,
                                    "dir_recovery_mode": "clip"
                                    },
                    }
        
        colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
                ]


        markers = ['.', ',', 'o', 'v', '^',
                '<', '>', '1', '2', '3',
                '4', 's', 'p', '*', 'h',
                'H', '+', 'x', 'D', 'd']
        
        
    
    return custom_filter_convex, custom_filter_non_convex, colors, markers