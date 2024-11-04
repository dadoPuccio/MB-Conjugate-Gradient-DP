import time
import copy
import torch

from src.optimizers.LineSearchOptimizer import LineSearchOptimizer
from Utils.opt_utils import random_seed_torch, maybe_torch

class ConjugateGradient(LineSearchOptimizer):
    """
        Note that ConjugateGradient is like LBFGS from the LBFGS optimizer from pytorch,
        the step needs to be called like in the following:
        >> closure = lambda: loss_function(model, images, labels, backwards=False)
        >> opt.step(closure)
    """

    def __init__(self,
                 params,
                 
                 # Line Search parameters
                 use_line_search=True,
                 c=0.5,
                 delta=0.5,
                 zhang_xi=1,
                 use_backtrack_heuristic = False,
                 max_lk = 5,
                 
                 # Conjugate Gradient parameters
                 cg_mode = 'FR',
                 eps = 1e-6,
                 max_beta = 1.5,

                 dir_recovery_mode = 'clip',
                 beta_damping=0.5,

                 eta_mode = 'polyak',
                 min_eta=1e-6,
                 max_eta=1e6,

                 # Polyak step-size
                 c_p = 1,
                 f_star = 0,

                 # Constant step-size
                 eta_0=1, # (used also by Vaswani)

                 # Vaswani step-size
                 gamma=2.0,
                 n_batches_per_epoch=8,
                 
                 ):
        
        assert cg_mode in ['PPR', 'FR', 'HS']
        assert dir_recovery_mode in ['clip', 'grad', 'qps', 'inv']
        assert eta_mode in ['polyak', 'constant', 'vaswani']

        params = list(params)
        super().__init__(params, c=c, delta=delta, zhang_xi=zhang_xi, max_lk=max_lk)

        # Conjugate-Gradient init
        self.cg_mode = cg_mode
        self.eps = eps
        self.beta_damping = beta_damping
        self.max_beta = max_beta

        self.eta_mode = eta_mode

        self.dir_recovery_mode = dir_recovery_mode

        self.min_eta = min_eta
        self.max_eta = max_eta

        # Polyak step-size params
        self.c_p = c_p
        self.f_star = f_star

        self.eta = eta_0

        # Vaswani step-size params
        self.gamma=gamma
        self.n_batches_per_epoch=n_batches_per_epoch

        self.use_backtrack_heuristic = use_backtrack_heuristic
        self.use_line_search = use_line_search
       
        self.grad_old = None
        self.direction_old = None

        self.new_epoch()


    def step(self, loss, grad_current_1, grad_current_2, grad_tot, closure_line_search, loss_2, closure_2):

        seed = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure_line_search()
            
        def closure_2_deterministic(): # needed when using QPS recover strategy (since we need to sample in x^{k-1} on the second part of the mini-batch)
            with random_seed_torch(int(seed)):
                return closure_2()

        params_current = copy.deepcopy(self.params)
        
        if self.direction_old is None:
            self.state['all_orig_beta'].append(0)
        else:
            beta = torch.tensor(self.compute_cg_beta(grad_current_1, self.grad_old))
            self.state['all_orig_beta'].append(beta.item())

        self.grad_old = grad_current_2

        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        if self.direction_old is None:
            direction = [-g for g in grad_tot]
            self.state['all_beta'].append(0)
        else:
            direction = [-g + beta * d_old for g, d_old in zip(grad_tot, self.direction_old)] 

            scalar_prod = self.compute_scalar_product(grad_tot, direction)

            if scalar_prod >= 0:
                self.state['d_not_descent'] += 1
                direction, scalar_prod = self.recover_descent_direction(params_current, grad_tot, direction, beta, scalar_prod,
                                                                        loss, closure_deterministic, closure_2_deterministic)

                assert scalar_prod < 0
            else:
                self.state['all_beta'].append(beta.item())
            

        # Compute sufficient decrement info for line search 
        dir_norm = self.compute_norm(direction)

        if self.direction_old is None:
            suff_dec = dir_norm**2
        else: 
            suff_dec = -scalar_prod


        # Compute initial step-size along the direction
        if self.eta_mode == "polyak":
            eta = max(min((loss - self.f_star)/(self.c_p * dir_norm**2 + 1e-8), self.max_eta), self.min_eta)
        elif self.eta_mode == "constant":
            eta = self.eta
        elif self.eta_mode == "vaswani":
            coeff = self.gamma ** (1. / self.n_batches_per_epoch)
            eta = max(min(self.eta * coeff, self.max_eta), self.min_eta)
        else:
            raise NotImplementedError("Check the input for eta_mode")

        if self.use_backtrack_heuristic:
            eta = max(min(eta * (self.delta ** self.lk), self.max_eta), self.min_eta)


        # Line search 
        if self.use_line_search:
            self.line_search(eta, loss, params_current, direction, suff_dec, closure_deterministic)
        else:
            self.update_no_line_search(eta, loss, params_current, direction, closure_deterministic)

        self.eta = self.state['all_step'][-1]

        self.params_old = params_current
        self.old_loss_2 = loss_2
        self.direction_old = [d for d in direction] # original formulation
        # self.direction_old = [self.state['all_step'][-1] * d for d in direction]

        self.state['step'] += 1
        self.state['all_dir_norm'].append(maybe_torch(dir_norm))
 
        return loss
    

    @torch.no_grad()
    def compute_cg_beta(self, grad_curr, grad_old):
        
        if self.cg_mode == "PPR": # Polyak-Polak-Ribiére
            
            g_dot_y = 0.
            g_old_dot_g_old = 0.
            
            for g, g_old in zip(grad_curr, grad_old): 
                
                g_dot_y += torch.sum(torch.mul(g, g - g_old))
                g_old_dot_g_old += torch.sum(torch.mul(g_old, g_old))

            beta = (g_dot_y / g_old_dot_g_old).item()
            return max(0, min(beta, self.max_beta))
        
        elif self.cg_mode == "FR": # Fletcher-Reeves
            
            g_dot_g = 0.
            g_old_dot_g_old = 0.
            
            for g, g_old in zip(grad_curr, grad_old): 

                g_dot_g += torch.sum(torch.mul(g, g))
                g_old_dot_g_old += torch.sum(torch.mul(g_old, g_old))

            beta = (g_dot_g / g_old_dot_g_old).item()
            return max(0, min(beta, self.max_beta)) 
        
        elif self.cg_mode == "HS": # Hestenes-Stiefel
            
            g_dot_y = 0.
            d_old_dot_y = 0.
            
            for g, g_old, d_old in zip(grad_curr, grad_old, self.direction_old): 

                g_dot_y += torch.sum(torch.mul(g, g - g_old))
                d_old_dot_y += torch.sum(torch.mul(d_old, g - g_old))

            beta = (g_dot_y / d_old_dot_y).item()
            return max(0, min(beta, self.max_beta)) 
        
        else:
            raise NotImplementedError
        
    @torch.no_grad()
    def recover_descent_direction(self, params_current, grad_tot, direction, beta, scalar_prod,
                                   loss_current, closure_deterministic, closure_2_deterministic):

        if self.dir_recovery_mode == 'clip':

            damping_factor = beta
            while scalar_prod >= 0:
                damping_factor = damping_factor * self.beta_damping
                direction = [-g + damping_factor * d_old for g, d_old in zip(grad_tot, self.direction_old)] 
                scalar_prod = self.compute_scalar_product(grad_tot, direction)

            self.state['all_beta'].append(damping_factor.item())
        
        elif self.dir_recovery_mode == 'inv':
            direction = [-d for d in direction] 
            scalar_prod = self.compute_scalar_product(grad_tot, direction)

            self.state['all_beta'].append(beta.item())

        elif self.dir_recovery_mode == 'grad':
            direction = [-g for g in grad_tot] 
            scalar_prod = self.compute_scalar_product(grad_tot, direction)

            self.state['all_beta'].append(0)

        elif self.dir_recovery_mode == 'qps':
            self.try_update(self.params, 0, self.params_old , [0 for _ in grad_tot]) # sample in x^{k-1}
            l0 = self.old_loss_2 + closure_2_deterministic()

            grad_norm = self.compute_norm(grad_tot) # Polyak step size along the negative-gradient direction (where we wish to sample)
            eta_1_polyak = max(min((loss_current - self.f_star)/(self.c_p * grad_norm**2 + 1e-8), self.max_eta), self.min_eta)

            self.try_update(self.params, - eta_1_polyak, params_current, grad_tot)
            l1 = closure_deterministic()

            dir_norm = self.compute_norm(direction) # Polyak step size along the (negative) direction of CG
            eta_2_polyak = max(min((loss_current - self.f_star)/(self.c_p * dir_norm**2 + 1e-8), self.max_eta), self.min_eta)
            self.try_update(self.params, - eta_2_polyak, params_current, direction)
            l2 = closure_deterministic()

            # Interpolation linear system
            A = 0.5 * torch.tensor([[1, 0, 0],
                                    [0, 0, eta_1_polyak**2],
                                    [(eta_2_polyak * beta)**2, 2 * (eta_2_polyak **2) * beta, eta_2_polyak**2]])
            
            grad_phi_1 = self.compute_scalar_product(grad_tot, [p - p_old for p, p_old in zip(params_current, self.params_old)])
            grad_phi_2 = - grad_norm ** 2
            
            b = torch.tensor([l0 - loss_current + grad_phi_1,
                              l1 - loss_current - eta_1_polyak * grad_phi_2,
                              l2 - loss_current + (eta_2_polyak * beta) * grad_phi_1 + eta_2_polyak * grad_phi_2]) 
            
            eigenvals = torch.linalg.eigvals(A).real

            if torch.any(torch.abs(eigenvals) < self.eps):
                B = torch.zeros((2, 2))
            else:
                B_flattened = torch.linalg.solve(A, b)

                B = torch.tensor([[B_flattened[0], B_flattened[1]],
                                  [B_flattened[1], B_flattened[2]]])
            
            eigenvals = torch.linalg.eigvals(B).real

            if torch.any(torch.abs(eigenvals) < self.eps): # gradient direction when B is bad-conditioned
                direction = [-g for g in grad_tot] 
                scalar_prod = - grad_norm ** 2

                self.state['all_alpha_qps'].append(1)
                self.state['all_beta_qps'].append(0)
        
            else:
                steps = torch.linalg.solve(B, -torch.tensor((grad_phi_1, grad_phi_2)))

                direction = [steps[0] * (p - p_old) + steps[1] * (- g) for p, p_old, g in zip(params_current, self.params_old, grad_tot)]
                scalar_prod = self.compute_scalar_product(grad_tot, direction)

                if scalar_prod >= 0:
                    direction = [-g for g in grad_tot]
                    scalar_prod = - grad_norm ** 2

                    self.state['all_alpha_qps'].append(1)
                    self.state['all_beta_qps'].append(0)

                    self.state['all_beta'].append(0)

                else:
                    self.state['all_alpha_qps'].append(steps[1].item())# alpha is the step along the gradient
                    self.state['all_beta_qps'].append(steps[0].item()) # beta is the step along momentum

                    self.state['all_beta'].append(steps[0].item())
            
        return direction, scalar_prod
   
    def new_epoch(self):
        super().new_epoch()

        self.state['all_dir_norm'] = []
        self.state['all_orig_beta'] = []
        self.state['all_beta'] = []
        
        if self.dir_recovery_mode == 'qps':
            self.state['all_alpha_qps'] = []
            self.state['all_beta_qps'] = []

        self.state['d_not_descent'] = 0