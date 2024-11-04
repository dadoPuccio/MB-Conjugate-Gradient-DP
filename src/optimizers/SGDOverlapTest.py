import time
import copy
import torch

from src.optimizers.LineSearchOptimizer import LineSearchOptimizer

from Utils.opt_utils import random_seed_torch

class SGDOverlapTest(LineSearchOptimizer):
    """
    Used to detect how many times the sgd + momentum update causes the direction not to be a descent direction.
    Updates are performed with Vanilla SGD.
    """

    def __init__(self,
                 params,

                 lr = 0.1,
                 overlap_percentages = [0, 25, 50, 75, 100],
                 beta = 0.9,
                 ref_angle = 90,
                 mode = 'past_info' 

                 ):
        
        assert mode in ['past_info', # evaluates the angle between current Grad dir & the dir (x^{k} - x^{k-1}) or -(old_mag)
                        'curr_dir']  # evaluates the angle between current Grad dir & the dir (-g_k + beta (x^{k} - x^{k-1})) or -(g_k + beta * old_mag)

        params = list(params)
        super().__init__(params, {})

        self.lr = lr
        self.overlap_percentages = overlap_percentages
        self.beta = beta

        self.ref_angle = ref_angle
        self.mode = mode

        self.mag = [torch.zeros_like(p) for p in self.params]
        # self.mag_alt = [torch.zeros_like(p) for p in self.params]
        self.old_params = [copy.deepcopy(p) for p in self.params]
    
        self.new_epoch()


    def step(self, closure):

        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()
       
        # get loss and compute gradients
        loss = closure_deterministic()
        loss.backward()

        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = self.get_grad_list(self.params)

        grad_norm = self.compute_norm(grad_current)

        direction = [-g for g in grad_current]

        self.mag = [g + self.beta * p_mag for g, p_mag in zip(grad_current, self.mag)]
       
        self.update_no_line_search(self.lr, loss, params_current, direction)

        self.old_params = [copy.deepcopy(p) for p in params_current]

        self.state['step'] += 1

        self.state['all_grad_norm'].append(grad_norm.item())
        
        return loss
        
    
    def compute_stats_overlap(self, closure_overlap, overlap_percentage: int, seed):

        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure_overlap()

        self.zero_grad()
        loss_next = closure_deterministic()
        loss_next.backward()

        grad_next = self.get_grad_list(self.params)
        grad_next_norm = self.compute_norm(grad_next)
        
        if self.mode == 'curr_dir':
            mag_direction = [- (g + self.beta * p_mag) for g, p_mag in zip(grad_next, self.mag)]
        elif self.mode == 'past_info':
            mag_direction = [- p_mag for p_mag in self.mag]
        
        scalar_product_mag = self.compute_scalar_product([-g for g in grad_next], mag_direction)
        mag_norm = self.compute_norm(mag_direction)

        cos_angle_mag = torch.clamp(scalar_product_mag / (grad_next_norm * mag_norm), -1.0, 1.0)
        angle_mag = torch.rad2deg(torch.acos(cos_angle_mag))

        self.state['all_mag_angle_' + str(overlap_percentage)].append(angle_mag.item())

        if self.mode == 'curr_dir':
            sgdm_direction = [- self.lr * g  + self.beta * (p - p_old) for g, p, p_old in zip(grad_next, self.params, self.old_params)]
        elif self.mode == 'past_info':
            sgdm_direction = [(p - p_old) for p, p_old in zip(self.params, self.old_params)]
            
        scalar_product_sgdm = self.compute_scalar_product([-g for g in grad_next], sgdm_direction)
        sgdm_norm = self.compute_norm(sgdm_direction)

        cos_angle_sgdm = torch.clamp(scalar_product_sgdm / (grad_next_norm * sgdm_norm), -1.0, 1.0)
        angle_sgdm = torch.rad2deg(torch.acos(cos_angle_sgdm))

        self.state['all_sgdm_angle_' + str(overlap_percentage)].append(angle_sgdm.item())

        # print(angle_mag, angle_sgdm, self.ref_angle, overlap_percentage)

        if angle_mag > self.ref_angle:
            self.state['mag_count_' + str(overlap_percentage)] += 1

        if angle_sgdm > self.ref_angle:
            self.state['sgdm_count_' + str(overlap_percentage)] += 1

        # if self.compute_scalar_product(grad_next, [- (g + self.beta * p_mag) for g, p_mag in zip(grad_next, self.mag)]) > 0:
        #     self.state['mag_count_' + str(overlap_percentage)] += 1

        # if self.compute_scalar_product(grad_next, [- self.lr * g  + self.beta * (p - p_old) for g, p, p_old in zip(grad_next, self.params, self.old_params)]) > 0:
        #     self.state['sgdm_count_' + str(overlap_percentage)] += 1

        
    
      
    def new_epoch(self):

        super().new_epoch()

        self.state['all_grad_norm'] = []

        for overlap_percentage in self.overlap_percentages:
            self.state['mag_count_' + str(overlap_percentage)] = 0
            self.state['sgdm_count_' + str(overlap_percentage)] = 0

            self.state['all_mag_angle_' + str(overlap_percentage)] = []
            self.state['all_sgdm_angle_' + str(overlap_percentage)] = []
