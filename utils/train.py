import os
import time
import json
from tqdm import tqdm
import torch
import torch.distributed as dist

from torch import nn


def kernel(x, y, sigma):
    dist = torch.cdist(x, y, p=2)
    k = torch.exp(- dist / (2 * sigma * sigma))
    return k


def train_single_epoch(rank, model, loader, optimizer, scheduler, loss, dataset_name, train_config, epoch_index, tag, subgraphs, world_size,):
    if world_size > 1:
        model_name = model.module.__class__.__name__
    else:
        model_name = model.__class__.__name__
    backprop = True if tag == 'train' else False
    if backprop:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    result = {'loss': torch.tensor(0., device=rank), 'counter': torch.tensor(0., device=rank), }

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f'Epoch {epoch_index} - {tag.capitalize()} [GPU {rank}]',
        position=rank,
        leave=False
    )

    for step, data in pbar:
        node_feat, node_loc, node_vel, node_attr = data.x, data.pos, data.vel, data.attr
        batch = data.batch
        loc_mean, loc_target = data.loc_mean, data.target

        # edge_index, edge_attr = get_edges_mini_batch_multigpu(data, rank)
        edge_index, edge_attr = data.edge_index, data.edge_attr

        batch_size = batch.max().item() + 1

        # Transfer to GPU
        node_feat, node_loc, node_vel, node_attr, edge_index, edge_attr, batch = \
            node_feat.to(rank), node_loc.to(rank), node_vel.to(rank), node_attr.to(rank), edge_index.to(rank), edge_attr.to(rank), batch.to(rank)
        loc_mean, loc_target = loc_mean.to(rank), loc_target.to(rank)

        # check train loader consistence
        if world_size > 1:
            loc_mean_list = [torch.zeros_like(loc_mean) for _ in range(world_size)]
            dist.all_gather(loc_mean_list, loc_mean)
            if rank == 0:
                for i in range(1, world_size):
                    if not torch.allclose(loc_mean_list[0], loc_mean_list[i], atol=1e-6):
                        assert False

        def model_forward(model_name):
            if model_name in ['FastEGNN', 'FastTFN', 'FastSchNet']:
                node_attr_nf = model.node_attr_nf if world_size == 1 else model.module.node_attr_nf
                node_attr_input = None if node_attr_nf == 0 else node_attr
                if model_name == 'FastTFN':
                    loc_pred, virtual_node_loc = model(node_feat, node_loc, node_vel, loc_mean, edge_index, batch, node_attr, edge_attr, node_attr_input)
                else:
                    loc_pred, virtual_node_loc = model(node_feat, node_loc, node_vel, loc_mean, edge_index, batch, edge_attr, node_attr_input)
                return loc_pred, virtual_node_loc
            elif model_name == 'FastRF':
                loc_pred, virtual_node_loc = model(node_loc, node_vel, loc_mean, edge_index, batch, edge_attr)
                return loc_pred, virtual_node_loc
            elif model_name == 'SchNet':
                loc_pred = model(z=node_feat, pos=node_loc, batch=batch, edge_index=edge_index)
                return loc_pred, None
            elif model_name == 'EGNN':
                loc_pred, _, _ = model(node_loc, node_feat, edge_index, edge_attr, node_vel)
                return loc_pred, None
            elif model_name == 'RF_vel':
                loc_pred = model(torch.norm(node_vel, dim=-1, keepdim=True), node_loc, edge_index, node_vel, edge_attr)
                return loc_pred, None
            elif model_name == 'OurDynamics':  # TFN
                loc_pred = model(node_loc, node_vel, node_attr, edge_index)
                return loc_pred, None
            elif model_name == 'Linear_dynamics':
                loc_pred = model(node_loc, node_vel)
                return loc_pred, None
            raise NotImplementedError(f'{model_name} not implemented!')

        if backprop:
            loc_pred, virtual_node_loc = model_forward(model_name)
        else:
            with torch.no_grad():
                loc_pred, virtual_node_loc = model_forward(model_name)

        loss_loc = loss(loc_pred, loc_target)  # loss in a single card

        node_cnt = torch.tensor(node_loc.size(0), device='cuda', dtype=torch.float32)  # node count in a card
        
        total_node_cnt = torch.tensor(node_loc.size(0), device='cuda', dtype=torch.float32)  # total node count
        if world_size > 1:
            dist.all_reduce(total_node_cnt, op=dist.ReduceOp.SUM)
        
        loss_loc = node_cnt / total_node_cnt * loss_loc  # loss of single device
        total_loss_loc = loss_loc.clone().detach()
        if world_size > 1:
            dist.all_reduce(total_loss_loc, op=dist.ReduceOp.SUM)  # loss to log
        loss_loc = world_size * loss_loc  # loss to compute gradient (ddp automatically **average** the gradient. But we need to **sum** the gradient)
        
        batch_size = batch.max() + 1
        result['loss'] += total_loss_loc * batch_size
        result['counter'] += batch_size

        pbar.set_postfix({'Epoch': epoch_index, 'MSE': total_loss_loc.item()})

        # Add MMD
        if model_name.startswith('Fast'):
            virtual_node_loc = virtual_node_loc.permute(0, 2, 1)  # [B, subgraphs, 3]
            l_vv, l_rv = 0., 0.
            num_sample = train_config.mmd.samples * subgraphs

            for i in range(batch_size):
                node_loc_i = loc_target[batch == i, ...]
                virtual_node_loc_i = virtual_node_loc[i, ...]

                num_node = node_loc_i.size(0)

                # Random sample real nodes
                sample_idx = torch.randperm(num_node)[:num_sample]
                node_loc_i = node_loc_i[sample_idx]  # [num_sample, 3]

                # calc_kernel
                k_vv = kernel(virtual_node_loc_i, virtual_node_loc_i, train_config.mmd.sigma)
                k_rv = kernel(node_loc_i, virtual_node_loc_i, train_config.mmd.sigma)

                l_vv += torch.sum(k_vv)
                l_rv += torch.sum(k_rv)
            
            # average between mini-batch
            l_vv = l_vv / batch_size / subgraphs / subgraphs
            l_rv = 2 * l_rv / batch_size / num_sample / subgraphs

            loss_mmd = l_vv - l_rv

            loss_loc = loss_loc + train_config.mmd.weight * world_size * node_cnt / total_node_cnt * loss_mmd

        if backprop:
            loss_loc = loss_loc / float(train_config.accumulation_steps)  # accumulate gradient
            loss_loc.backward()
            if (step + 1) % train_config.accumulation_steps == 0:
                if (world_size > 1 or dataset_name == 'LargeFluid') and model_name == 'FastEGNN':
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""

    if rank == 0:
        print(f'{prefix + tag} epoch: {epoch_index}, avg loss: {result["loss"] / result["counter"] :.5f}')

    return (result['loss'] / result['counter']).item()


def train(rank, model, optimizer, scheduler, loader_train, loader_valid, loader_test, train_config, log_config, config, start_epoch):
    # setup(rank, config.data.world_size)

    print(f'{train_config=}')
    print(f'{log_config=}')
    world_size = config.data.world_size

    loss_mse = nn.MSELoss()

    # Preparations before training
    if rank == 0:
        log_dict = {'epochs': [], 'loss': [], 'loss_train': []}
        best_log_dict = {'epoch_index': 0, 'loss_valid': 1e8, 'loss_test': 1e8, 'loss_train': 1e8,}

    if rank == 0 and log_config.wandb.enable == True:
        wandb_dir = os.path.join(log_config.log_dir, log_config.exp_name)
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ['WANDB_API_KEY'] = log_config.wandb.api_key  # API key
        if log_config.wandb.offline:
            os.environ['WANDB_MODE'] = 'offline'
        import wandb
        wandb.init(config=config,
            project=log_config.wandb.project,
            entity=log_config.wandb.entity,
            group=f'{config.data.dataset_name}',
            name=log_config.exp_name,
            dir=wandb_dir,
            reinit=True)
        
    if rank == 0: 
        log_dir = os.path.join(log_config.log_dir, log_config.exp_name, 'log')
        os.makedirs(log_dir, exist_ok=True)
        state_dict_dir = os.path.join(log_config.log_dir, log_config.exp_name, 'state_dict')
        os.makedirs(state_dict_dir, exist_ok=True)
        start =time.perf_counter()
    
    early_stop_flag = torch.tensor(0, device=rank)


    for epoch_index in range(1 + start_epoch, train_config.epochs + 1):
        if early_stop_flag == 1:
            print(f'Device {rank} stop succeed!')
            break

        loss_train = train_single_epoch(rank, model, loader_train, optimizer, scheduler, loss_mse, config.data.dataset_name, train_config, epoch_index, 
                                        tag='train', subgraphs=config.model.virtual_channels, world_size=world_size)
        
        if rank == 0:
            log_dict['loss_train'].append(loss_train)
        
        if epoch_index % log_config.test_interval == 0:
            loss_valid = train_single_epoch(rank, model, loader_valid, optimizer, scheduler, loss_mse, config.data.dataset_name, train_config, epoch_index, 
                                            tag='valid', subgraphs=config.model.virtual_channels, world_size=world_size)
            loss_test = train_single_epoch(rank, model, loader_test, optimizer, scheduler, loss_mse, config.data.dataset_name, train_config, epoch_index, 
                                           tag='test', subgraphs=config.model.virtual_channels, world_size=world_size)
            
            if rank == 0:
                log_dict['epochs'].append(epoch_index)
                log_dict['loss'].append(loss_test)
            
                if loss_valid < best_log_dict['loss_valid']:
                    best_log_dict = {'epoch_index': epoch_index, 'loss_valid': loss_valid, 'loss_test': loss_test, 'loss_train': loss_train,}

                    # Save best model
                    state_dict = {
                        'epoch': epoch_index,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
                        'loss_train': loss_train, 'loss_valid': loss_valid, 'loss_test': loss_test,
                        'config': config,
                    }
                    torch.save(state_dict, os.path.join(state_dict_dir, 'best_model.pth'))
                
                if log_config.wandb.enable:
                    wandb.log({'loss_train': loss_train, 'loss_valid': loss_valid, 'loss_test': loss_test,}, step=epoch_index)

                print(f'*** Best Valid Loss: {best_log_dict["loss_valid"] :.5f} | Best Test Loss: {best_log_dict["loss_test"] :.5f} | Best Epoch Index: {best_log_dict["epoch_index"]}')
                
                # save last model
                state_dict = {
                    'epoch': epoch_index,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
                    'loss_train': loss_train, 'loss_valid': loss_valid, 'loss_test': loss_test,
                    'config': config,
                }
                torch.save(state_dict, os.path.join(state_dict_dir, 'last_model.pth'))
            
            if rank == 0 and epoch_index - best_log_dict['epoch_index'] >= train_config.early_stop:
                best_log_dict['early_stop'] = epoch_index
                print(f'Early stopped! Epoch: {epoch_index}')
                early_stop_flag.fill_(1)

            if world_size > 1:
                dist.all_reduce(early_stop_flag, op=dist.ReduceOp.MAX)  # early stop! broadcast!

        elif rank == 0:
            if log_config.wandb.enable:
                wandb.log({'loss_train': loss_train}, step=epoch_index)

        if rank == 0:
            end = time.perf_counter() 
            time_cost = end - start
            best_log_dict['time_cost'] = time_cost
            
            json_object = json.dumps([best_log_dict, log_dict, config], indent=4)
            with open(os.path.join(log_dir, 'log.json'), "w") as outfile:
                outfile.write(json_object)
    
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        if log_config.wandb.enable:
            wandb.log({'best_test_loss': best_log_dict['loss_test'],})
        return best_log_dict, log_dict
