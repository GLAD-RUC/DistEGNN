import os
import math
import copy
import time
import yaml
import torch
import random
import datetime
import argparse
import torch.distributed as dist

from easydict import EasyDict
from functools import partial

from torch import optim
from torch.utils.data import RandomSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader

from utils.train import train
from utils.seed import fix_seed

from models.basic import EGNN, RF_vel, Linear_dynamics
from models.SchNet import SchNet
from models.FastRF import FastRF
from models.FastTFN import FastTFN
from models.FastEGNN import FastEGNN
from models.FastSchNet import FastSchNet

from datasets.process_dataset import process_dataset_edge_cutoff, process_dataset_distribute, DatasetWrapper

torch.multiprocessing.set_sharing_strategy('file_system')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_parameters(model, local_rank, world_size):
    model_params = [param.data.flatten() for param in model.parameters()]
    concatenated_params = torch.cat([param for param in model_params])

    reference_params = concatenated_params.clone().detach()

    dist.broadcast(reference_params, src=0)

    is_consistent = torch.allclose(concatenated_params, reference_params, atol=1e-6)

    if local_rank == 0:
        print(f"Rank {local_rank}: Model parameters consistency check passed!")
    else:
        print(f"Rank {local_rank}: Model parameters consistency: {is_consistent}")
    
    return is_consistent


def get_model(model_config, world_size, dataset_name):
    print(f'{model_config=}')
    if model_config.model_name == 'FastEGNN':
        return FastEGNN(node_feat_nf=model_config.node_feat_nf, node_attr_nf=model_config.node_attr_nf, edge_attr_nf=model_config.edge_attr_nf, normalize=model_config.normalize,
                        hidden_nf=model_config.hidden_nf, n_layers=model_config.n_layers, virtual_channels=model_config.virtual_channels, gravity=None, world_size=world_size)
    elif model_config.model_name == 'FastRF':
        return FastRF(edge_attr_nf=model_config.edge_attr_nf, hidden_nf=model_config.hidden_nf, n_layers=model_config.n_layers, virtual_channels=model_config.virtual_channels, world_size=world_size)
    elif model_config.model_name == 'FastTFN':
        return FastTFN(node_feat_nf=model_config.node_feat_nf, node_attr_nf=model_config.node_attr_nf, edge_attr_nf=model_config.edge_attr_nf, hidden_nf=model_config.hidden_nf, 
                       virtual_channels=model_config.virtual_channels, n_layers=model_config.n_layers, normalize=model_config.normalize, gravity=None)
    elif model_config.model_name == 'FastSchNet' or model_config.model_name == 'SchNet':
        if dataset_name == 'nbody_100':
            interatomic_cutoff = 1
        elif dataset_name == 'protein':
            interatomic_cutoff = 10
        elif dataset_name == 'Water-3D':
            interatomic_cutoff = 0.035
        else:
            assert False
        if model_config.model_name == 'FastSchNet':
            return FastSchNet(node_feat_nf=model_config.node_feat_nf, node_attr_nf=model_config.node_attr_nf, edge_attr_nf=model_config.edge_attr_nf, hidden_nf=model_config.hidden_nf, 
                            virtual_channels=model_config.virtual_channels, n_layers=model_config.n_layers, normalize=model_config.normalize, gravity=None, cutoff=interatomic_cutoff)
        else:
            return SchNet(hidden_channels=model_config.hidden_nf, max_num_neighbors=200000, cutoff=interatomic_cutoff)
    elif model_config.model_name == 'EGNN':
        return EGNN(n_layers=model_config.n_layers, in_node_nf=model_config.node_feat_nf, in_edge_nf=model_config.edge_attr_nf, 
                    hidden_nf=model_config.hidden_nf, with_v=True)
    elif model_config.model_name == 'RF':
        return RF_vel(hidden_nf=model_config.hidden_nf, edge_attr_nf=model_config.edge_attr_nf, n_layers=model_config.n_layers)
    elif model_config.model_name == 'TFN':
        from models.se3_dynamics.dynamics import OurDynamics as SE3_Transformer
        return SE3_Transformer(nf=model_config.hidden_nf // 2, n_layers=model_config.n_layers, model='tfn', num_degrees=2, div=1)
    elif model_config.model_name == 'Linear':
        return Linear_dynamics()
    raise NotImplementedError(f'Model {model_config.model_name} Not Implemented')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='path to config yaml file')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--split_mode', type=str, default=None)
    parser.add_argument('--early_stop', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--cutoff_rate', type=float, default=None)
    parser.add_argument('--outer_radius', type=float, default=None)
    parser.add_argument('--inner_radius', type=float, default=None)
    parser.add_argument('--virtual_channels', type=int, default=None)
    args = parser.parse_args()

    # Get configs
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    
    if args.wandb == True:
        config.log.wandb.offline = False
    if args.seed is not None:
        config.seed = args.seed
    if args.lr is not None:
        config.train.learning_rate = args.lr
    if args.model_name is not None:
        config.model.model_name = args.model_name
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.split_mode is not None:
        config.data.split_mode = args.split_mode
    if args.early_stop is not None:
        config.train.early_stop = args.early_stop
    if args.checkpoint is not None:
        config.model.checkpoint = args.checkpoint
    if args.cutoff_rate is not None:
        config.data.cutoff_rate = args.cutoff_rate
    if args.outer_radius is not None:
        config.data.outer_radius = args.outer_radius
    if args.inner_radius is not None:
        config.data.inner_radius = args.inner_radius   
    if args.virtual_channels is not None:
        config.model.virtual_channels = args.virtual_channels

    # Multiple GPUs
    world_size = torch.cuda.device_count()
    config.data.world_size = world_size
    print(f'Use {world_size} GPUs!')

    log_time_suffix = str(time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())))
    if config.data.accelerate_mode == 'distribute':
        if config.model.model_name.startswith('Fast'):
            config.log.exp_name = f'{config.data.dataset_name}_{config.data.split_mode}_{config.model.model_name}_{config.data.outer_radius}_{config.data.inner_radius}_{world_size}_{config.model.virtual_channels}_{log_time_suffix}'
        else:
            config.log.exp_name = f'{config.data.dataset_name}_{config.data.split_mode}_{config.model.model_name}_{config.data.outer_radius}_{config.data.inner_radius}_{world_size}_{log_time_suffix}'
    else:
        if config.model.model_name.startswith('Fast'):
            config.log.exp_name = f'{config.data.dataset_name}_{config.model.model_name}_{config.data.radius}_{config.data.cutoff_rate :.3f}_{config.model.virtual_channels}_{world_size}_{log_time_suffix}'
        else:
            config.log.exp_name = f'{config.data.dataset_name}_{config.model.model_name}_{config.data.radius}_{config.data.cutoff_rate :.3f}_{world_size}_{log_time_suffix}'

    if world_size > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        timeout = datetime.timedelta(seconds=18000)
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size, timeout=timeout)
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    fix_seed(config.seed)
    
    # Get dataset
    if config.data.accelerate_mode == 'distribute':
        processed_data_files = process_dataset_distribute(local_rank, world_size, config.data)
    elif config.data.accelerate_mode == 'cutoff_edges':
        assert world_size == 1
        processed_data_files = process_dataset_edge_cutoff(config.data)
    else:
        raise NotImplementedError(f'accelerate_mode {config.data.accelerate_mode} not implemented')
    if world_size > 1: dist.barrier()

    fix_seed(config.seed)

    # Get dataset
    dataset_train, dataset_valid, dataset_test = DatasetWrapper(processed_data_files[0]), DatasetWrapper(processed_data_files[1]), DatasetWrapper(processed_data_files[2])
    print(f'Device [{local_rank}]: Data get!')
    loader = partial(DataLoader, batch_size=config.data.batch_size, drop_last=True, num_workers=4, pin_memory=False)
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(config.seed)
    sampler = RandomSampler(dataset_train, replacement=False, generator=sampler_generator)
    loader_train = loader(dataset=dataset_train, follow_batch=['edge_index'], sampler=sampler)  # Use RandomSampler(with same generator) to maintain same graph across devices!
    loader_valid = loader(dataset=dataset_valid, follow_batch=['edge_index'], shuffle=False)
    loader_test  = loader(dataset=dataset_test, follow_batch=['edge_index'], shuffle=False)

    # Get model
    model = get_model(config.model, world_size, config.data.dataset_name)
    model = model.to(local_rank)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)

    if config.train.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs * len(loader_train) // config.train.accumulation_steps, eta_min=1e-8, verbose=False)
    else:
        scheduler = None

    if local_rank == 0:
        print(model)
        print(count_parameters(model))

    # Load checkpoint
    start_epoch = 0
    if args.checkpoint is not None:
        map_location = torch.device(f'cuda:{local_rank}')
        checkpoint = torch.load(args.checkpoint, map_location=map_location)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f'GPU[{local_rank}]: Checkpoint loaded!')
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        is_consistent = check_model_parameters(model, local_rank, world_size)
        assert is_consistent
        dist.barrier()

    # Training
    # Multi Card
    train(local_rank, model, optimizer, scheduler, loader_train, loader_valid, loader_test, config.train, config.log, config, start_epoch)
