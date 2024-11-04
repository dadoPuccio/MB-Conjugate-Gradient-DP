import os
import argparse
import torch 
import numpy as np
import time
import exp_configs
from src import datasets, models, optimizers, metrics

from Utils.logs_utils import init_logs_folder, init_exp_log_folder, save_json, init_csv, append_row_csv, append_rows_csv

NUM_WORKERS = 6
    
print("Num. Workers:", NUM_WORKERS)

TERMINATION_LOSS = 1e-5
MAX_TIME = 10000

def trainval(exp_dict, savedir, exp_fields_logs, datadir):

    # Create Experiments Log Dir
    expdir = init_exp_log_folder(savedir, exp_dict, exp_fields_logs)
    save_json(os.path.join(expdir, 'params.json'), exp_dict)
   
    print('Experiment saved in %s' % expdir)

    # set seed
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # -----------
    # Load Train Dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if exp_dict.get("multiple_gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict,
                                     device=device)
    
    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   device=device)
    
    print("Number of training samples:", len(train_set))

    # Model
    # -----------
    model = models.get_model(exp_dict["model"], train_set=train_set)
    if exp_dict.get("multiple_gpu"):
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if exp_dict.get("half"):
        model = model.half()
        print("Using half-precision model")
    elif exp_dict.get("double"):
        model = model.double()

    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # Load Optimizer
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch =n_batches_per_epoch,
                                   train_set_len=len(train_set))


    start_epoch = 0
    print('Starting experiment at epoch %d/%d' % (start_epoch, exp_dict['max_epoch']))

    excluded_fields = ["step", "time",  "batch_size", "loss", "all_lipschitz", "special_count,step_size", "grad_norm", "new_loss", "dec",
                       "Q_k", "C_k", "lk", "lipschitz", "losses", "relative_dec", "all_relative_dec", "all_lip_smooth", "suff_dec", "lip_smooth",
                       "sharp", "zero_steps", "numerical_error", "sufficient_dec", "loss_history", "sgc", "n_backtr", "all_suff_dec", "all_sharp",
                       "prev_alpha", "prev_beta"] 
    
    # Train Loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                                drop_last=False,
                                                shuffle=True,
                                                batch_size=exp_dict["batch_size"], num_workers=NUM_WORKERS, pin_memory=True)
    
    first_iter = True
    iterations = 0
    full_time = 0 
    for epoch in range(start_epoch, exp_dict['max_epoch']):

        epoch_stats = {}

        epoch_stats['epoch'] = epoch

        if exp_dict["opt"]["name"] in exp_configs.custom_opt_list:
            opt.new_epoch()

        # Set seed
        np.random.seed(exp_dict['runs']+epoch)
        torch.manual_seed(exp_dict['runs']+epoch)
        torch.cuda.manual_seed_all(exp_dict['runs']+epoch)

        # Compute metrics on whole Training dataset and the Validation dataset
        all_dataset_metrics_train = metrics.compute_metric_on_dataset(model, train_set, metric_names=[exp_dict["loss_func"], exp_dict["acc_func"]], device=device)

        for metric_name in all_dataset_metrics_train.keys():
            epoch_stats["train_" + metric_name.split("_")[1]] = all_dataset_metrics_train[metric_name]

        all_dataset_metrics_val = metrics.compute_metric_on_dataset(model, val_set, metric_names=[exp_dict["loss_func"], exp_dict["acc_func"]], device=device)

        for metric_name in all_dataset_metrics_val.keys():
            epoch_stats["val_" + metric_name.split("_")[1]] = all_dataset_metrics_val[metric_name]

        # 3. Train over train loader
        model.train()
        s_time = time.time()

        iterator = train_loader.__iter__()

        if exp_dict["opt"]["name"] in exp_configs.opt_bibatch_ls:
            if first_iter:
                buffer_images = None
                buffer_labels = None
                first_iter = False
            opt, model, mini_batch_stats, buffer_images, buffer_labels = train_loop_bibatch(n_batches_per_epoch, iterator, opt, loss_function, model, 
                                                                                        iterations, device, buffer_images, buffer_labels, exp_dict['loss_func'])
            
        elif exp_dict['opt']["name"] in exp_configs.opt_overlap_test:
            if first_iter:
                buffer_images = None
                buffer_labels = None
                first_iter = False
            opt, model, mini_batch_stats, buffer_images, buffer_labels = train_loop_custom_overlap(n_batches_per_epoch, iterator, opt, loss_function, model,
                                                                                                    iterations, device, buffer_images, buffer_labels, exp_dict['opt']['overlap_percentages'])

        elif exp_dict.get("overlap_batches"):
            if first_iter:
                buffer_images = None
                buffer_labels = None
                first_iter = False
            opt, model, mini_batch_stats, buffer_images, buffer_labels =  train_loop_overlapped(n_batches_per_epoch, iterator, opt, loss_function, model,
                                                                                                iterations, device, buffer_images, buffer_labels)
        
        else: 
            opt, model, mini_batch_stats =  train_loop_standard(n_batches_per_epoch, iterator, opt, loss_function, model, iterations, device)
      
        e_time = time.time() 

        iterations = mini_batch_stats['iter'][-1]

        epoch_stats["train_epoch_time"] = e_time - s_time
        full_time += (e_time - s_time)
        epoch_stats["time"] = full_time

        # Record metrics
        if exp_dict["opt"]["name"] in exp_configs.custom_opt_list:
            
            for stat in opt.state.keys():
                if stat not in excluded_fields:

                    if "all_" in stat:
                      
                        mini_batch_stats[stat] = opt.state[stat]

                        avg_stat_name = 'avg_' + stat.split("all_")[1]

                        epoch_stats[avg_stat_name] = sum(opt.state[stat]) / max(len(opt.state[stat]), 1)

                    else:
                        epoch_stats[stat] = opt.state[stat]

        if epoch == start_epoch:
            log_string = ""
            for stat in epoch_stats.keys():
                log_string += stat + " " * max(10 - len(stat), 0) 

            print(log_string)

            init_csv(expdir, "epoch_stats.csv", [stat for stat in epoch_stats.keys()])
            init_csv(expdir, "minibatch_stats.csv", [stat for stat in mini_batch_stats.keys()])

        log_string = ""
        for stat in epoch_stats.keys():
            val = epoch_stats[stat]
            if type(val) == int:
                val = "{:6d}".format(val)
            else:
                val = "{:12.5f}".format(val)
            
            log_string += val + " " * max(10 - len(val), 0) 

        print(log_string)

        append_row_csv(expdir, "epoch_stats.csv", [val for val in epoch_stats.values()])
        append_rows_csv(expdir, "minibatch_stats.csv", list(zip(*[val for val in mini_batch_stats.values()])))

        if epoch_stats["train_loss"] < TERMINATION_LOSS:
            print('Very Small Loss')
            break
        
        # print(full_time)
        if full_time > MAX_TIME:
            print('Max Time Reached')
            break

    print('Experiment completed')
    

def train_loop_standard(n_batches_per_epoch, iterator, opt, loss_function, model, iterations, device):
    
    mini_batch_stats = {}
    mini_batch_stats['iter'] = []

    if exp_dict["opt"]["name"] not in exp_configs.custom_opt_list:
        mini_batch_stats['all_loss'] = []
    
    for _ in range(int(np.ceil(n_batches_per_epoch))):
        images, labels, _ = next(iterator)
        images, labels = images.to(device), labels.to(device)

        if exp_dict.get("half"):
            images = images.half()
        elif exp_dict.get("double"):
            images = images.double()

        opt.zero_grad()

        if exp_dict["opt"]["name"] in exp_configs.custom_opt_list:
            closure = lambda: loss_function(model, images, labels, backwards=False)
            opt.step(closure)
        else:
            loss = loss_function(model, images, labels)
            loss.backward()
            opt.step()

            mini_batch_stats["all_loss"].append(loss.item())

        iterations += 1
        mini_batch_stats['iter'].append(iterations)

    return opt, model, mini_batch_stats


def train_loop_overlapped(n_batches_per_epoch, iterator, opt, loss_function, model,
                          iterations, device, buffer_images, buffer_labels):
    
    mini_batch_stats = {}
    mini_batch_stats['iter'] = []

    if exp_dict["opt"]["name"] not in exp_configs.custom_opt_list:
        mini_batch_stats['all_loss'] = []

    for _ in range(int(np.ceil(n_batches_per_epoch))):
        batch_images, batch_labels, _ = next(iterator)
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        c1_images, c2_images = torch.chunk(batch_images, 2, dim=0)
        c1_labels, c2_labels = torch.chunk(batch_labels, 2, dim=0)

        if buffer_images is None:
            zipped_batch_data = zip([batch_images], [batch_labels])
            buffer_images = c2_images
            buffer_labels = c2_labels

        else:
            zipped_batch_data = zip([torch.cat((buffer_images, c1_images), dim=0), batch_images],
                                    [torch.cat((buffer_labels, c1_labels), dim=0), batch_labels])
            buffer_images = c2_images
            buffer_labels = c2_labels
        
        for images, labels in zipped_batch_data:

            if exp_dict.get("half"):
                images = images.half()
            elif exp_dict.get("double"):
                images = images.double()

            opt.zero_grad()

            if exp_dict["opt"]["name"] in exp_configs.custom_opt_list:
                closure = lambda: loss_function(model, images, labels, backwards=False)
                opt.step(closure) 
            else:
                loss = loss_function(model, images, labels)
                loss.backward()
                opt.step()

                mini_batch_stats["all_loss"].append(loss.item())

            iterations += 1
            mini_batch_stats['iter'].append(iterations)

    return opt, model, mini_batch_stats, buffer_images, buffer_labels


def train_loop_bibatch(n_batches_per_epoch, iterator, opt, loss_function, model,
                       iterations, device, buffer_images, buffer_labels, loss_func_name):
    
    mini_batch_stats = {}
    mini_batch_stats['iter'] = []

    for _ in range(int(np.ceil(n_batches_per_epoch))):
        batch_images, batch_labels, _ = next(iterator)
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        c1_images, c2_images = torch.chunk(batch_images, 2, dim=0)
        c1_labels, c2_labels = torch.chunk(batch_labels, 2, dim=0)

        if buffer_images is None:
            zipped_batch_data = zip([[c1_images, c2_images]], [[c1_labels, c2_labels]])
            buffer_images = c2_images
            buffer_labels = c2_labels
            
        else:
            zipped_batch_data = zip([[buffer_images, c1_images], [c1_images, c2_images]],
                                     [[buffer_labels, c1_labels], [c1_labels, c2_labels]])            
            buffer_images = c2_images
            buffer_labels = c2_labels

        for images, labels in zipped_batch_data:
            
            if exp_dict.get("half"):
                images[0] = images[0].half()
                images[1] = images[1].half()
            elif exp_dict.get("double"):
                images[0] = images[0].double()
                images[1] = images[1].double()

            opt.zero_grad()

            # loss_1, loss_2, loss_tot, grad_1, grad_2, grad_tot = compute_grads_bibatch(model, images, labels, loss_function, loss_func_name)
            # print(loss_1.item(), loss_2.item(), loss_tot.item(), torch.linalg.norm(grad_1[0]).item(), torch.linalg.norm(grad_1[0]).item(), torch.linalg.norm(grad_tot[0]).item())
            # opt.zero_grad()
            loss_1, loss_2, loss_tot, grad_1, grad_2, grad_tot = compute_grads_bibatch_std(model, images, labels, loss_function, loss_func_name)
            # print(loss_1.item(), loss_2.item(), loss_tot.item(), torch.linalg.norm(grad_1[0]).item(), torch.linalg.norm(grad_1[0]).item(), torch.linalg.norm(grad_tot[0]).item())
            # opt.zero_grad()
            # loss_1, loss_2, loss_tot, grad_1, grad_2, grad_tot = compute_grads_bibatch_vmap(model, images, labels, loss_function, loss_func_name)
            # print(loss_1.item(), loss_2.item(), loss_tot.item(), torch.linalg.norm(grad_1[0]).item(), torch.linalg.norm(grad_1[0]).item(), torch.linalg.norm(grad_tot[0]).item())
            opt.state['n_forwards'] += 1
            opt.state['n_backwards'] += 1
            
            closure_line_search = lambda: loss_function(model, torch.cat(images, dim=0), torch.cat(labels, dim=0), backwards=False)

            if exp_dict["opt"]["name"] == 'conjugate_gradient' and opt.dir_recovery_mode == 'qps':

                closure_2 = lambda: loss_function(model, images[1], labels[1], backwards=False)

            else:
                closure_2 = None

            opt.step(loss_tot, grad_1, grad_2, grad_tot, closure_line_search, loss_2, closure_2)

            iterations += 1
            mini_batch_stats['iter'].append(iterations)

    return opt, model, mini_batch_stats, buffer_images, buffer_labels


def compute_grads_bibatch(model, images, labels, loss_function, loss_func_name):

    all_images = torch.cat(images, dim=0)
    all_labels = torch.cat(labels, dim=0)

    if loss_func_name == 'softmax_loss':
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        all_logits = model(all_images)
        criterion_loss = criterion(all_logits, all_labels.view(-1))
    elif loss_func_name == 'logistic_loss':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        all_logits = model(all_images)
        criterion_loss = criterion(all_logits.view(-1), all_labels.float().view(-1))
    
    
    cl_1, cl_2 = criterion_loss[:images[0].shape[0]], criterion_loss[images[0].shape[0]:]

    loss_1 = torch.mean(cl_1)
    loss_1.backward(retain_graph=True)

    grad_1 = [p.grad.detach().clone() for p in model.parameters()]

    model.zero_grad()

    loss_2 = torch.mean(cl_2) 
    loss_2.backward()

    grad_2 = [p.grad for p in model.parameters()]

    loss_tot = (loss_1 * images[0].shape[0] + loss_2 * images[1].shape[0]) / all_images.shape[0]
    grad_tot = [(g_1 * images[0].shape[0] + g_2 * images[1].shape[0]) / all_images.shape[0] for g_1, g_2 in zip(grad_1, grad_2)]


    return loss_1, loss_2, loss_tot, grad_1, grad_2, grad_tot

def compute_grads_bibatch_std(model, images, labels, loss_function, loss_func_name):
    
    loss_1 = loss_function(model, images[0], labels[0])
    loss_1.backward(retain_graph=True)

    grad_1 = [p.grad.detach().clone() for p in model.parameters()]

    model.zero_grad()

    loss_2 = loss_function(model, images[1], labels[1])
    loss_2.backward()

    grad_2 = [p.grad for p in model.parameters()]

    loss_tot = (loss_1 * images[0].shape[0] + loss_2 * images[1].shape[0]) / (images[0].shape[0] + images[1].shape[0])
    grad_tot = [(g_1 * images[0].shape[0] + g_2 * images[1].shape[0]) / (images[0].shape[0] + images[1].shape[0]) for g_1, g_2 in zip(grad_1, grad_2)]

    return loss_1, loss_2, loss_tot, grad_1, grad_2, grad_tot

from torch.func import functional_call, vmap, grad 

def compute_grads_bibatch_vmap(model, images, labels, loss_function, loss_func_name):

    if images[0].shape != images[1].shape:
        return compute_grads_bibatch_std(model, images, labels, loss_function, loss_func_name)

    images = torch.stack(images)
    labels = torch.stack(labels)

    params = {k: v.detach() for k, v in model.named_parameters()}

    if loss_func_name == 'softmax_loss':
        def loss_fn(predictions, targets):
            return torch.nn.functional.cross_entropy(predictions, targets, reduction='mean')
        
    elif loss_func_name == 'logistic_loss':
        def loss_fn(predictions, targets):
            return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction='mean')
    
    def compute_loss(params, samples, targets):
        predictions = functional_call(model, params, (samples,))
        loss = loss_fn(predictions, targets)
        return loss, loss

    compute_grad = grad(compute_loss, has_aux=True)

    compute_grad_vmap = vmap(compute_grad, in_dims=(None, 0, 0))

    per_part_grads, per_part_losses = compute_grad_vmap(params, images, labels)
    
    grad_1 = []
    grad_2 = []
    grad_tot = []
    for v in per_part_grads.values():
        grad_1.append(v[0])
        grad_2.append(v[1])
        grad_tot.append(torch.mean(v, dim=0))

    return per_part_losses[0], per_part_losses[0], torch.mean(per_part_losses), grad_1, grad_2, grad_tot


def train_loop_custom_overlap(n_batches_per_epoch, iterator, opt, loss_function, model,
                              iterations, device, buffer_images, buffer_labels, overlap_percentages = [0, 25, 50, 75, 100]):
    
    mini_batch_stats = {}
    mini_batch_stats['iter'] = []

    for _ in range(int(np.ceil(n_batches_per_epoch))):
        images, labels, _ = next(iterator)
        images, labels = images.to(device), labels.to(device)

        if exp_dict.get("half"):
            images = images.half()
        elif exp_dict.get("double"):
            images = images.double()

        if buffer_images is None:
            buffer_images = images
            buffer_labels = labels

        else:
            seed = time.time()
            for overlap_percentage in overlap_percentages:
                n_elements_old = int(buffer_images.shape[0] * overlap_percentage / 100)
                n_elements_new = int(images.shape[0] * (100 - overlap_percentage) / 100)
            
                if n_elements_old == 0:
                    custom_overlap_images = images
                    custom_overlap_labels = labels
                elif n_elements_new == 0:
                    custom_overlap_images = buffer_images
                    custom_overlap_labels = buffer_labels
                else:
                    custom_overlap_images = torch.cat((buffer_images[-n_elements_old:], images[:n_elements_new]), dim=0)
                    custom_overlap_labels = torch.cat((buffer_labels[-n_elements_old:], labels[:n_elements_new]), dim=0)

                opt.zero_grad()

                closure_overlap = lambda: loss_function(model, custom_overlap_images, custom_overlap_labels, backwards=False)
                opt.compute_stats_overlap(closure_overlap, overlap_percentage, seed)

            buffer_images = images
            buffer_labels = labels

        opt.zero_grad()

        closure = lambda: loss_function(model, images, labels, backwards=False)
        opt.step(closure)

        iterations += 1
        mini_batch_stats['iter'].append(iterations)
           
    return opt, model, mini_batch_stats, buffer_images, buffer_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)

    args = parser.parse_args()
    
    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    savedir = init_logs_folder(args.savedir_base)

    exp_fields_logs = ["dataset", "opt.name"]

    overall_fields = {}
    for exp_dict in exp_list:
        for field in exp_dict:
            if field == 'opt':

                for opt_field in exp_dict[field]:
                    key = field + "." + opt_field
                    if key not in overall_fields:
                        overall_fields[key] = []
                    overall_fields[key].append(exp_dict[field][opt_field])

            else:
                if field not in overall_fields:
                    overall_fields[field] = []
                overall_fields[field].append(exp_dict[field])

    for key, val in overall_fields.items():
        try:
            set(val)
            if len(set(val)) > 1 and key not in exp_fields_logs:
                exp_fields_logs.append(key)
        except:
            print("Unable to use " + str(key) + " to name log folders")
        

    # Run experiments
    # ----------------------------
    for exp_dict in exp_list:
        if exp_dict['overlap_batches'] == False and exp_dict['opt']['name'] == 'conjugate_gradient':
            pass
        else:
            trainval(exp_dict=exp_dict,
                    savedir=savedir,
                    exp_fields_logs=exp_fields_logs,
                    datadir=args.datadir)
