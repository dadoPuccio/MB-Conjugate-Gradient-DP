import time
import torch
from abc import abstractmethod

from Utils.opt_utils import random_seed_torch, maybe_torch

class LineSearchOptimizer(torch.optim.Optimizer):
    
    def __init__(self,
                 params,
                         
                 # Line Search parameters
                 c=0.5, 
                 delta=0.5,
                 zhang_xi=1,
                 max_lk=5
                 ):
        
        assert None not in [c, delta, zhang_xi]

        params = list(params)
        super().__init__(params, {})

        self.params = params 

        self.device = params[0].get_device()

        # Line Search parameters
        self.c = c
        self.delta = delta
        self.zhang_xi = zhang_xi

        self.Q_k = 0
        self.C_k = 0

        self.lk = 0
        self.max_lk = max_lk
        
        # Stats
        self.state['step'] = 0
        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0


    def new_epoch(self):
        self.state['all_step'] = []
        self.state['all_orig_step'] = []
        self.state['all_loss'] = []
        self.state['all_dec'] = []

        self.state['backtracks'] = 0

    @abstractmethod
    def step(self, closure):
        seed = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()       
            
        raise NotImplementedError("")
    
    @abstractmethod
    def step(self, closure_1, closure_2, n_1, n_2):
        seed = time.time()
        def closure_1_deterministic():
            with random_seed_torch(int(seed)):
                return closure_1()
        def closure_2_deterministic():
            with random_seed_torch(int(seed)):
                return closure_2()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return (n_1 * closure_1() + n_2 * closure_2()) / (n_1 + n_2)  
            
        raise NotImplementedError("Implement this method to extend a method that uses splitted mini-batches")
    
    @abstractmethod
    def step(self, closure_1, closure_2, closure_all, n_1, n_2):
        seed = time.time()
        def closure_1_deterministic():
            with random_seed_torch(int(seed)):
                return closure_1()
        def closure_2_deterministic():
            with random_seed_torch(int(seed)):
                return closure_2()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure_all()      
            
        raise NotImplementedError("Implement this method to extend a method that uses splitted mini-batches")
        
    @torch.no_grad()
    def line_search(self,
                    orig_step_size: torch.tensor, loss: torch.tensor, 
                    params_current: list, direction: list, suff_dec: torch.tensor, 
                    closure_deterministic, eta_k = 0):
        
        assert suff_dec > 0

        # compute nonmonotone terms for the Zhang&Hager line search
        q_kplus1 = self.zhang_xi * self.Q_k + 1
        self.C_k = (self.zhang_xi * self.Q_k * self.C_k + loss.item()) / q_kplus1
        self.Q_k = q_kplus1

        step_size = orig_step_size
        
        # TODO: controllo da fare prima di chiamare line_search
        if loss.item() > 1e-8:
            
            found = 0

            for e in range(100):
                # try a prospective step
                self.try_update(self.params, step_size, params_current, direction)

                # compute the loss at the next step; no need to compute gradients.
                loss_next = closure_deterministic()
                self.state['n_forwards'] += 1

                ref_value = max(self.C_k, loss.item())
                found, step_size = self.check_armijo_conditions(step_size=step_size,
                                                                loss=ref_value,
                                                                suff_dec=suff_dec,
                                                                loss_next=loss_next,
                                                                c=self.c,
                                                                beta_b=self.delta,
                                                                eta_k=eta_k)
            

                if found == 1:
                    # assert loss.item() > loss_next
                    break

            # if line search exceeds 100 internal iterations
            if found == 0:
                step_size = torch.tensor(data=1e-6)
                self.try_update(self.params, step_size, params_current, direction)

            self.state['backtracks'] += e
            self.lk = min(max(self.lk + e - 1, 0), self.max_lk)

        else: 
            # print("Loss is below threshold")
            step_size = torch.tensor(0.)
            loss_next = torch.tensor(loss.item())

        self.state['all_step'].append(maybe_torch(step_size))
        self.state['all_orig_step'].append(maybe_torch(orig_step_size))
        self.state['all_loss'].append(maybe_torch(loss))
        self.state['all_dec'].append(max(maybe_torch(loss) - maybe_torch(loss_next), 0))

    @torch.no_grad()
    def update_no_line_search(self, orig_step_size: torch.tensor, loss: torch.tensor,
                              params_current: list, direction: list, closure_deterministic = None):
        
        self.try_update(self.params, orig_step_size, params_current, direction)

        if closure_deterministic is not None:
            loss_next = closure_deterministic()
            self.state['n_forwards'] += 1
            self.state['all_dec'].append(max(maybe_torch(loss) - maybe_torch(loss_next), 0))
        else:
            self.state['all_dec'].append(torch.inf)

        self.state['all_step'].append(maybe_torch(orig_step_size))
        self.state['all_orig_step'].append(maybe_torch(orig_step_size))
        self.state['all_loss'].append(maybe_torch(loss))
        

    # Armijo line search
    @torch.no_grad()
    def check_armijo_conditions(self, step_size, loss, suff_dec,
                                loss_next, c, beta_b, eta_k = 0):
        found = 0
        sufficient_decrease = step_size * c * suff_dec
        rhs = loss - sufficient_decrease + eta_k
        break_condition = loss_next - rhs
        if (break_condition <= 0):
            found = 1
        else:
            step_size = step_size * beta_b

        return found, step_size
    

    @torch.no_grad()
    def try_update(self, params, step_size, params_current, direction):

        for p_next, p_current, d in zip(params, params_current, direction):
            p_next.data = p_current + step_size * d 
        
    @torch.no_grad()
    def compute_norm(self, direction):

        res = 0.
        for d in direction: 
            
            res += torch.sum(torch.mul(d, d)) # linearized Scalar products

        return torch.sqrt(res)
    
    @torch.no_grad()
    def compute_scalar_product(self, direction_1, direction_2):
        res = 0.
        for d_1, d_2 in zip(direction_1, direction_2): 
            res += torch.sum(torch.mul(d_1, d_2)) # linearized Scalar products
        return res
    
    @torch.no_grad()
    def get_grad_list(self, params):
        return [p.grad for p in params]


    
