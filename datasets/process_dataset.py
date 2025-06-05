import os
import math
import h5py
import random
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import numpy as np
import pickle as pkl
import zstandard as zstd
import MDAnalysis
import MDAnalysisData

from scipy.sparse import coo_matrix
from MDAnalysis import transformations
from MDAnalysis.analysis import distances

from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph

from datasets.distribute_graphs import split_large_graph_random, split_large_graph_metis
from utils.rotate import random_rotate, random_rotate_y


def process_dataset_edge_cutoff(data_config):
    dataset_name = data_config.dataset_name

    if dataset_name.startswith('nbody'):
        return process_nbody_cutoff(data_config.data_dir, data_config.dataset_name, data_config.max_samples, 
                                    data_config.radius, data_config.frame_0, data_config.frame_T, data_config.cutoff_rate)
    elif dataset_name.startswith('protein'):
        return process_protein_cutoff(data_config.data_dir, data_config.dataset_name, data_config.max_samples, data_config.radius, 
                                      data_config.delta_t, data_config.cutoff_rate, data_config.backbone, data_config.test_rot, data_config.test_trans)
    elif dataset_name.startswith('Water-3D'):
        return process_water3d_cutoff(data_config.data_dir, data_config.dataset_name, data_config.max_samples, data_config.radius, 
                                      data_config.delta_t, data_config.cutoff_rate)
    else:
        raise NotImplementedError(f'{dataset_name} not implemented in {__file__}')


def process_dataset_distribute(rank, world_size, data_config):
    dataset_name = data_config.dataset_name

    if dataset_name.startswith('Water-3D'):
        return process_water_3d_dist(rank, world_size, data_config.data_dir, data_config.dataset_name, data_config.outer_radius, 
                                     data_config.inner_radius, data_config.max_samples, data_config.split_mode, data_config.delta_t)
    elif dataset_name == "Fluid113K":
        return process_large_fluid_dist(rank, world_size, data_config.data_dir, data_config.dataset_name, data_config.outer_radius, 
                                        data_config.inner_radius, data_config.max_samples, data_config.split_mode, data_config.delta_t)
    else:
        raise NotImplementedError(f'{dataset_name} not implemented in {__file__}')


def process_nbody_cutoff(data_dir, dataset_name, max_samples, radius, frame_0, frame_T, cutoff_rate):
    processed_dir = os.path.join(data_dir, dataset_name, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    processed_file_names = []
    for partition in ['train', 'valid', 'test']:
        processed_file_name = f'{dataset_name}_{partition}_{radius}_{cutoff_rate :.3f}_{max_samples}_{frame_0}_{frame_T}.pt'
        processed_file_name = os.path.join(processed_dir, processed_file_name)
        processed_file_names.append(processed_file_name)

        if os.path.exists(processed_file_name):
            print(f'{processed_file_name} exists!')
            continue

        suffix = f'{partition}_charged100_0_0_1'
        loc = np.load(os.path.join(data_dir, dataset_name, f'loc_{suffix}.npy'))
        vel = np.load(os.path.join(data_dir, dataset_name, f'vel_{suffix}.npy'))
        charges = np.load(os.path.join(data_dir, dataset_name, f'charges_{suffix}.npy'))
        loc, vel, charges = torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(charges)

        loc, vel = loc[0:max_samples, :, :, :], vel[0:max_samples, :, :, :]
        loc_0, loc_t = loc[:, frame_0, :, :], loc[:, frame_T, :, :]  # [num_systems, num_node_r, 3]
        vel_0, vel_t = vel[:, frame_0, :, :], vel[:, frame_T, :, :]  # [num_systems, num_node_r, 3]
        charges = charges[0:max_samples]  # [num_systems, num_node_r, 1]


        def process_key(loc_0, vel_0, charges, loc_t, data):
            # if partition == 'test':
            #     rotate_matrix = random_rotate()
            #     rotate_matrix = rotate_matrix.to(loc_0).to(torch.float)
            #     loc_0 = loc_0 @ rotate_matrix
            #     loc_t = loc_t @ rotate_matrix
            #     vel_0 = vel_0 @ rotate_matrix

            # Edge
            num_nodes = loc_0.size(0)
            if radius == -1:
                edge_index = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
                edge_index = torch.tensor(edge_index).T
            else:
                edge_index = radius_graph(loc_0, r=radius, max_num_neighbors=num_nodes)
            
            edge_index = cutoff_edge(edge_index, loc_0, cutoff_rate)
            edge_attr = torch.norm(loc_0[edge_index[0], :] - loc_0[edge_index[1], :], p=2, dim=1).unsqueeze(-1).repeat(1, 2)

            # Node Feat
            feat_node_velocity = torch.sqrt(torch.sum(vel_0 ** 2, dim=1)).unsqueeze(1)
            feat_node_charge = charges
            node_feat = torch.cat([feat_node_velocity, feat_node_charge / feat_node_charge.max()], dim=1)

            # Virtual node loc = mean
            loc_mean = torch.mean(loc_0, dim=0).unsqueeze(0)  # [1, 3]

            data.append(Data(x=node_feat, pos=loc_0, vel=vel_0, attr=charges, target=loc_t, loc_mean=loc_mean, 
                        edge_index=edge_index, edge_attr=edge_attr, special_nodes=torch.ones((loc_0.size(0), ), dtype=torch.bool, device=loc_0.device)))

        data = []
        num_systems = charges.size(0)
        for key in tqdm(range(num_systems)):
            process_key(loc_0[key, :, :], vel_0[key, :, :], charges[key, :, :], loc_t[key, :, :], data)
        
        torch.save(data, processed_file_name)
        print(f'{processed_file_name} processed!')

    return processed_file_names


def process_protein_cutoff(data_dir, dataset_name, max_samples, radius, delta_t, cutoff_rate, backbone=True, test_rot=False, test_trans=False):
    processed_dir = os.path.join(data_dir, dataset_name, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    processed_file_names = []
    for partition in ['train', 'valid', 'test']:
        processed_file_name = f'{dataset_name}_{partition}_{radius}_{cutoff_rate :.3f}_{max_samples}_{delta_t}.pt'
        processed_file_name = os.path.join(processed_dir, processed_file_name)
        processed_file_names.append(processed_file_name)

        if os.path.exists(processed_file_name):
            print(f'{processed_file_name} exists!')
            continue

        def process_key(adk_data, t):
            if backbone:
                ag = adk_data.select_atoms('backbone')
            else:
                ag = adk_data.atoms
            charges = torch.tensor(adk_data.atoms[ag.ix].charges).float().unsqueeze(-1)

            frame_0, frame_t = t, t + delta_t

            # [deep] copy to avoid multithread change
            ts_0 = adk_data.trajectory[frame_0].copy()
            ts_1 = adk_data.trajectory[frame_0 + 1].copy()
            ts_t = adk_data.trajectory[frame_t].copy()

            # transfer to pytorch tensor
            loc_0 = torch.tensor(ts_0.positions[ag.ix])
            vel_0 = torch.tensor(ts_1.positions[ag.ix]) - loc_0
            loc_t = torch.tensor(ts_t.positions[ag.ix])

            # test rotation equivariant
            if test_rot and partition == 'test':
                rotate_matrix = random_rotate()
                rotate_matrix = rotate_matrix.to(loc_0.device).to(loc_0.dtype)
                loc_0 = loc_0 @ rotate_matrix
                vel_0 = vel_0 @ rotate_matrix
                loc_t = loc_t @ rotate_matrix

            # test translation equivariant
            if test_trans and partition == 'test':
                trans = np.random.randn(3) * ts_0.dimensions[:3] / 2
                trans = torch.from_numpy(trans).unsqueeze(0).to(loc_0.device).to(loc_0.dtype)
                loc_0 = loc_0 + trans
                loc_t = loc_t + trans
    
            # Edges
            edge_index = coo_matrix(distances.contact_matrix(loc_0.detach().numpy(), cutoff=radius, returntype="sparse"))
            edge_index.setdiag(False)
            edge_index.eliminate_zeros()
            edge_index = torch.stack([torch.tensor(edge_index.row, dtype=torch.long),
                                    torch.tensor(edge_index.col, dtype=torch.long)], dim=0)
            
            # Cutoff edges
            edge_index = cutoff_edge(edge_index, loc_0, cutoff_rate)

            # Edge attr
            edge_attr = torch.norm(loc_0[edge_index[0], :] - loc_0[edge_index[1], :], p=2, dim=1).unsqueeze(-1).repeat(1, 2)

            # Node Feat
            feat_node_velocity = torch.sqrt(torch.sum(vel_0 ** 2, dim=1)).unsqueeze(1)
            feat_node_charge = charges
            node_feat = torch.cat([feat_node_velocity, feat_node_charge / feat_node_charge.max()], dim=1)

            # Virtual node loc = mean
            loc_mean = torch.mean(loc_0, dim=0).unsqueeze(0)  # [1, 3]

            return Data(x=node_feat, pos=loc_0, vel=vel_0, attr=charges, target=loc_t, loc_mean=loc_mean, 
                        edge_index=edge_index, edge_attr=edge_attr, special_nodes=torch.ones((loc_0.size(0), ), dtype=torch.bool, device=loc_0.device))


        adk = MDAnalysisData.datasets.fetch_adk_equilibrium(data_home=data_dir)
        adk_data = MDAnalysis.Universe(adk.topology, adk.trajectory)

        data = []
        train_valid_test_split = {
            'train': (0, 2481),
            'valid': (2481, 2481 + 827),
            'test': (2481 + 827, 2481 + 827 + 863),
        }

        data = Parallel(n_jobs=10)(delayed(process_key)(adk_data, i) for i in tqdm(range(train_valid_test_split[partition][0], train_valid_test_split[partition][1])))

        # with ThreadPoolExecutor(max_workers=25) as executor:
        #     future_to_key = {executor.submit(process_key, adk_data, t, data): t for t in tqdm(range(train_valid_test_split[partition][0], train_valid_test_split[partition][1]), desc='Preparing keys')}

        #     for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc='Processing keys'):
        #         future.result()
        
        torch.save(data, processed_file_name)
        print(f'{processed_file_name} processed!')

    return processed_file_names


def process_water3d_cutoff(data_dir, dataset_name, max_samples, radius, delta_t, cutoff_rate):
    processed_dir = os.path.join(data_dir, dataset_name, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    processed_file_names = []
    for partition in ['train', 'valid', 'test']:
        processed_file_name = f'{dataset_name}_{partition}_{radius}_{cutoff_rate :.3f}_{max_samples}_{delta_t}.pt'
        processed_file_name = os.path.join(processed_dir, processed_file_name)
        processed_file_names.append(processed_file_name)

        if os.path.exists(processed_file_name):
            print(f'{processed_file_name} exists!')
            continue

        def process_key(file, key, data, num_frames):
            particle_type = file[key]['particle_type']
            particle_type = torch.tensor(np.array(particle_type)).float().unsqueeze(-1)
            postion = file[key]['position']
            postion = torch.tensor(np.array(postion)).float()

            frames = [random.randint(0, 250) for _ in range(min(num_frames, max_samples - len(data)))]  # Select 15 frames from former 150 frames
            
            if len(frames) == 0:
                return
            
            for frame in frames:
                loc_0, loc_t = postion[frame, :, :], postion[frame + delta_t, :, :]
                vel_0 = postion[frame + 1, :, :] - postion[frame, :, :]
                node_type = particle_type

                # roteta_matrix = random_rotate_y()
                # roteta_matrix = roteta_matrix.to(loc_0.device).to(torch.float)

                # if partition == 'test':
                #     loc_0 = loc_0 @ roteta_matrix
                #     loc_t = loc_t @ roteta_matrix
                #     vel_0 = vel_0 @ roteta_matrix

                # Edge
                edge_index = radius_graph(loc_0, r=0.035, max_num_neighbors=100000)
                edge_index = cutoff_edge(edge_index, loc_0, cutoff_rate)
                edge_attr = torch.norm(loc_0[edge_index[0], :] - loc_0[edge_index[1], :], p=2, dim=1).unsqueeze(-1).repeat(1, 2)

                # Node Feat
                feat_node_velocity = torch.sqrt(torch.sum(vel_0 ** 2, dim=1)).unsqueeze(1)
                feat_node_charge = node_type
                node_feat = torch.cat([feat_node_velocity, feat_node_charge / feat_node_charge.max()], dim=1)

                # Virtual node loc = mean
                loc_mean = torch.mean(loc_0, dim=0).unsqueeze(0)  # [1, 3]

                data.append(Data(x=node_feat, pos=loc_0, vel=vel_0, attr=node_type, target=loc_t, loc_mean=loc_mean, 
                            edge_index=edge_index, edge_attr=edge_attr, special_nodes=torch.ones((loc_0.size(0), ), dtype=torch.bool, device=loc_0.device)))
        
        data = []

        file_path = os.path.join(data_dir, dataset_name, f'{partition}.h5')
        with h5py.File(file_path, 'r') as file:
            keys = list(file.keys())
            
            with ThreadPoolExecutor(max_workers=25) as executor:
                if partition == 'train':
                    future_to_key = {executor.submit(process_key, file, key, data, 15): key for key in keys}
                else:
                    future_to_key = {executor.submit(process_key, file, key, data, 15): key for key in keys}

                for future in tqdm(as_completed(future_to_key), total=len(keys), desc='Processing keys'):
                    future.result()

        torch.save(data, processed_file_name)
        print(f'{processed_file_name} processed!')

    return processed_file_names


def cutoff_edge(edge_index, pos, cutoff_rate):
    edge_dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=1)
    _, id_chosen = torch.sort(edge_dist)
    id_chosen = id_chosen[:int(id_chosen.size(0) * (1 - cutoff_rate))]
    edge_index = edge_index[:, id_chosen]
    return edge_index


def process_water_3d_dist(rank, world_size, data_dir, dataset_name, outer_radius, inner_radius, max_samples, split_mode, delta_t):
    processed_dir = os.path.join(data_dir, dataset_name, 'processed')

    card_processed_file_names = []
    for partition in ['train', 'valid', 'test']:
        card_processed_file_name = f'{dataset_name}_{split_mode}_{partition}_{world_size}_{outer_radius:.3f}_{inner_radius:.3f}_{max_samples}_{delta_t}__{rank}-{world_size}.pt'
        card_processed_file_name = os.path.join(processed_dir, card_processed_file_name)
        card_processed_file_names.append(card_processed_file_name)

    if rank != 0:
        return card_processed_file_names
    
    # Process dataset for each GPU card
    os.makedirs(processed_dir, exist_ok=True)
    for partition in ['train', 'valid', 'test']:
        processed_file_template = f'{dataset_name}_{split_mode}_{partition}_{world_size}_{outer_radius:.3f}_{inner_radius:.3f}_{max_samples}_{delta_t}' + '__{rank}' + f'-{world_size}.pt'
        
        for i in range(world_size):
            processed_file_name = os.path.join(processed_dir, processed_file_template.format(rank=i))
            if os.path.exists(processed_file_name):
                print(f'{processed_file_name} exists!')
            else:
                break
        else:
                continue
        
        def process_key(file, key, data, num_frames):
            particle_type = file[key]['particle_type']
            particle_type = torch.tensor(np.array(particle_type)).float().unsqueeze(-1)
            position = file[key]['position']
            position = torch.tensor(np.array(position)).float()

            frames = [random.randint(0, 250) for _ in range(min(num_frames, max_samples - len(data)))]
            if len(frames) == 0:
                return

            for frame in frames:
                vel_frame = position[frame + 1, :, :] - position[frame, :, :]
                node_feat = torch.cat([torch.sqrt(torch.sum(vel_frame ** 2, dim=-1)).unsqueeze(-1), particle_type / particle_type.max()], dim=-1)

                if split_mode == 'metis':
                    result = split_large_graph_metis(
                        pos=position[frame, :, :].cpu(),
                        x=node_feat.cpu(),
                        target=position[frame + delta_t, :, :].cpu(),
                        vel=vel_frame.cpu(),
                        attr=particle_type.cpu(),
                        inner_radius=inner_radius,
                        outer_radius=outer_radius,
                        world_size=world_size,
                        device='cpu',
                    )
                elif split_mode == 'random':
                    result = split_large_graph_random(
                        pos=position[frame, :, :].cpu(),
                        x=node_feat.cpu(),
                        target=position[frame + delta_t, :, :].cpu(),
                        vel=vel_frame.cpu(),
                        attr=particle_type.cpu(),
                        radius=inner_radius,
                        # inner_radius=inner_radius,
                        # outer_radius=outer_radius,
                        world_size=world_size,
                        device='cpu',
                    )
                else:
                    raise NotImplementedError(f'{split_mode} not Implemented')
                # data = result[0]
                # print(key, frame)
                # print(position[frame, ...])
                # print(data.pos)
                # print(data.target)
                # print(data.vel)
                # print(data.loc_mean)
                # print(data.x)
                # print(data.attr)
                # print(data.edge_attr)
                # print(data.edge_index)

                # assert False
                for i, r in enumerate(result):
                    data[i].append(r)

        data = [[] for i in range(world_size)]
        file_path = os.path.join(data_dir, dataset_name, f'{partition}.h5')
        with h5py.File(file_path, 'r') as file:
            keys = list(file.keys())
            # keys = keys[:1]
            
            with ThreadPoolExecutor(max_workers=25) as executor:
                if partition == 'train':
                    future_to_key = {executor.submit(process_key, file, key, data, 15): key for key in keys}
                else:
                    future_to_key = {executor.submit(process_key, file, key, data, 15): key for key in keys}

                for future in tqdm(as_completed(future_to_key), total=len(keys), desc='Processing keys'):
                    future.result()

        for i in range(world_size - 1):
            assert len(data[i]) == len(data[i + 1])
        
        for i in range(world_size):
            processed_file_name = os.path.join(processed_dir, processed_file_template.format(rank=i))
            torch.save(data[i], processed_file_name)
            print(f'{processed_file_name} processed!')

    return card_processed_file_names


def process_large_fluid_dist(rank, world_size, data_dir, dataset_name, outer_radius, inner_radius, max_samples, split_mode, delta_t):
    train_valid_test_split = {
        'train': (1, 101),
        'valid': (101, 121),
        'test': (121, 141),
    }

    train_valid_test_ratio = {
        'train': (0, 40),
        'valid': (40, 45),
        'test': (45, 50),
    }

    processed_dir = os.path.join(data_dir, dataset_name, 'processed')

    card_processed_file_names = []
    for partition in ['train', 'valid', 'test']:
        card_processed_file_name = f'{dataset_name}_{split_mode}_{partition}_{world_size}_{outer_radius:.3f}_{inner_radius:.3f}_{max_samples}_{delta_t}__{rank}-{world_size}.pt'
        card_processed_file_name = os.path.join(processed_dir, card_processed_file_name)
        card_processed_file_names.append(card_processed_file_name)

    if rank != 0:
        return card_processed_file_names
    
    # Process dataset for each GPU card
    os.makedirs(processed_dir, exist_ok=True)    
    for partition in ['train', 'valid', 'test']:
        processed_file_template = f'{dataset_name}_{split_mode}_{partition}_{world_size}_{outer_radius:.3f}_{inner_radius:.3f}_{max_samples}_{delta_t}' + '__{rank}' + f'-{world_size}.pt'
        
        for i in range(world_size):
            processed_file_name = os.path.join(processed_dir, processed_file_template.format(rank=i))
            if os.path.exists(processed_file_name):
                print(f'{processed_file_name} exists!')
            else:
                break
        else:
                continue
        

        def process_key(idx, data, num_frames):
            position, vel = [], []
            viscosity, mass = [], []

            decompressor = zstd.ZstdDecompressor()
            for i in range(16):
                file_name = f'sim_{idx :04d}_{i :02d}.msgpack.zst'
                with open(os.path.join(data_dir, dataset_name, file_name), 'rb') as f:
                    decompressed_data = decompressor.decompress(f.read())
                    raw_data = msgpack.unpackb(decompressed_data, raw=False)
                    for frame in raw_data:
                        position.append(frame['pos'])
                        vel.append(frame['vel'])
                    viscosity = raw_data[0]['viscosity']
                    mass = raw_data[0]['m']

            position, vel = torch.tensor(np.array(position)).float(), torch.tensor(np.array(vel)).float()
            viscosity, mass = torch.tensor(np.array(viscosity)).float(), torch.tensor(np.array(mass)).float()

            frames = [random.randint(0, 50) for _ in range(min(num_frames, max_samples - len(data)))]
            if len(frames) == 0:
                return

            for frame in frames:
                node_attr = torch.stack([viscosity, mass], dim=-1)
                node_feat = torch.cat([node_attr, torch.sqrt(torch.sum(vel[frame] ** 2, dim=-1)).unsqueeze(-1)], dim=-1)

                if split_mode == 'metis':
                    result = split_large_graph_metis(
                        pos=position[frame, :, :].cpu(),
                        x=node_feat.cpu(),
                        target=position[frame + delta_t, :, :].cpu(),
                        vel=vel[frame, ...].cpu(),
                        attr=node_attr.cpu(),
                        inner_radius=inner_radius,
                        outer_radius=outer_radius,
                        world_size=world_size,
                        device='cpu',
                    )
                elif split_mode == 'random':
                    result = split_large_graph_random(
                        pos=position[frame, :, :].cpu(),
                        x=node_feat.cpu(),
                        target=position[frame + delta_t, :, :].cpu(),
                        vel=vel[frame, ...].cpu(),
                        attr=node_attr.cpu(),
                        radius=inner_radius,
                        world_size=world_size,
                        device='cpu',
                    )
                else:
                    raise NotImplementedError(f'{split_mode} not Implemented')
                for i, r in enumerate(result):
                    data[i].append(r)

        data = [[] for i in range(world_size)]

        with ThreadPoolExecutor(max_workers=20) as executor:
            if partition == 'train':
                future_to_key = {executor.submit(process_key, idx, data, 16): idx for idx in range(train_valid_test_split[partition][0], train_valid_test_split[partition][1])}
            else:
                future_to_key = {executor.submit(process_key, idx, data, 16): idx for idx in range(train_valid_test_split[partition][0], train_valid_test_split[partition][1])}

            for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc='Processing keys'):
                future.result()

        for i in range(world_size - 1):
            assert len(data[i]) == len(data[i + 1])
        
        for i in range(world_size):
            processed_file_name = os.path.join(processed_dir, processed_file_template.format(rank=i))
            torch.save(data[i], processed_file_name)
            print(f'{processed_file_name} processed!')

    return card_processed_file_names



class DatasetWrapper(Dataset):
    def __init__(self, processed_file_name):
        super(DatasetWrapper, self).__init__()
        self.data = torch.load(processed_file_name)

        print(f'Dataset total len: {len(self.data)}')
        print(self.data[0])


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, i):
        return self.data[i]
