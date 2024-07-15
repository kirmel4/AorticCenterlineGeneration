import numpy as np
import torch
import os
from collections import OrderedDict
from aaa.utils import torch_float, io
from aaa.datasets.datasetGz import aaaIGzDataset
from aaa.models.my_unet import my_unet
from torch import nn
from tqdm import tqdm
from aaa.datasets.MyDataset import SegAttDataset
from aaa.utils import ( init_determenistic, init_logging,
                            torch_float)
from aaa.losses.threedim import dice_with_logits_loss, focal_with_logits_loss
from aaa.process.pre import (VoxelRandomSampler, VoxelSequentialSampler )
from torch.utils.data import DataLoader
from aaa.utils import config as cfg
import copy
import torch.optim.swa_utils as tsu
import time
import click
from collections import OrderedDict
from aaa.datasets.datasetGz import aaaIGzDataset
import inference
import metrics
import yaml
import json

config = dict()

def init_global_config(**kwargs):
    cfg.init_timestamp(config)
    cfg.init_run_command(config)
    cfg.init_kwargs(config, kwargs)
    init_logging(config, __name__, config['LOGGER_TYPE'], filename=config['PREFIX']+'logger_name.txt')
    cfg.init_device(config)
    cfg.init_verboser(config, logger=config['LOGGER'])
    cfg.init_options(config)

class SegAttDataset(aaaIGzDataset):
    #data in z-x-y format
    def __init__(self, imgs, masks, attraction_fields, channels=None):
        self.channels = channels

        self.imgs = imgs
        self.masks = masks
        self.attraction_fields = attraction_fields

        self.keys = list()
        self.shapes = list()

        for key in self.imgs:
            mask = self.masks[key]
            print(mask.shape[:-1])
            self.keys.append(key)
            self.shapes.append(mask.shape[:-1]) ###
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        key, selector = idx
        voxel = self.imgs[key][selector]
        mask = self.masks[key][selector]
        attraction_field = self.attraction_fields[key][selector]
        voxel = io.split_images(voxel[:, :, None], config['LOADER_OPTIONS']) # :

        return voxel, mask, attraction_field, key, selector

def collate_fn(batch):
        voxels, masks, attraction_fields, keys, selectors = zip(*batch)
        voxels = np.moveaxis(voxels[0], 2, 0)

        selections = [*zip(keys, selectors)]

        voxels = torch_float(voxels, torch.device('cpu'))
        masks = torch_float(masks, torch.device('cpu'))
        attraction_fields = torch_float(attraction_fields, torch.device('cpu'))

        return voxels, masks, attraction_fields, selections

def load_data_with_labels(data):
    
    datapath = str(config['DATAPATH'] )

    imgs = OrderedDict()
    masks = { }
    attraction_fields = {}

    for name, keys in config['SPLIT_OPTIONS'].items():
        for key in keys:
            image = np.load(datapath + '/' + key + '/' + key + '_img.npz')['data']
            
            imgs[key] = image #####
 

            mask = np.load(datapath + '/' + key + '/' + key + '_segmentation.npz')['data']
            masks[key] = mask

            attraction_field = np.load(datapath + '/' + key + '/' + key + '_attraction.npz')['data']
            print('nfiiieeld', attraction_field.shape)

            attraction_fields[key] = attraction_field
        data[name] = SegAttDataset(
            { key: imgs[key] for key in keys },
            { key: masks[key] for key in keys },
            { key: attraction_fields[key] for key in keys}, channels = config['LOADER_OPTIONS']['channels']
        )

def load_data():
    data = { }
    load_data_with_labels(data)

    return data

def inner_supervised(model, voxels_batch, masks_batch, attraction_field_batch):
    voxels_batch = voxels_batch.to(config['DEVICE'])
    masks_batch = masks_batch.to(config['DEVICE'])
    attraction_field_batch = attraction_field_batch.to(config['DEVICE'])
    pred_mask_batch, pred_attraction_batch = model(voxels_batch)

    mse_loss = nn.MSELoss()

    masks_batch = masks_batch[:,:,:,:,1]

    loss_mask = dice_with_logits_loss(masks_batch, pred_mask_batch, average='binary', activation='softmax') +\
                focal_with_logits_loss(masks_batch.long(), pred_mask_batch)
    
    masks_batch_stacked = torch.moveaxis(torch.stack((masks_batch, masks_batch, masks_batch)),0,1)
    pred_mask_batch_stacked = torch.moveaxis(torch.stack((pred_mask_batch.argmax(axis = 1), pred_mask_batch.argmax(axis = 1), pred_mask_batch.argmax(axis = 1))),0,1)

    mse = mse_loss(torch.moveaxis(attraction_field_batch,-1,1)*masks_batch_stacked, pred_attraction_batch*pred_mask_batch_stacked)*25 

    att_magnitudes = torch.linalg.norm(attraction_field_batch, dim = -1)
    pred_att_magnitudes = torch.linalg.norm(pred_attraction_batch, dim = 1)

    mag_lumen = att_magnitudes*masks_batch
    pred_mag_lumen = pred_att_magnitudes*pred_mask_batch.argmax(axis = 1)

    vector_length_regularization = torch.mean(torch.abs(mag_lumen - pred_mag_lumen))

    regularization_coefficient = 125
    vector_length_regularization *= regularization_coefficient

    loss_attraction = (mse+vector_length_regularization)*1
    print(loss_mask, loss_attraction)

    return loss_mask, loss_attraction, mse, vector_length_regularization

def inner_train_loop(model, averaged_model, opt, dataset):
    model.train()
    averaged_model.train()

    batch_mask_losses = list()
    batch_att_losses = list()
    batch_mse_losses = list()
    batch_reg_losses = list()

    datasampler = VoxelRandomSampler( config['VOXELING_OPTIONS']['voxel_shape'],
                                      dataset.keys,
                                      dataset.shapes,
                                      config['N_ITERATIONS'] * config['BATCH_SIZE'])

    dataloader = DataLoader( dataset,
                             batch_size= config['BATCH_SIZE'],
                             sampler=datasampler,
                             collate_fn=collate_fn,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=1 )

    opt.zero_grad()

    for step_idx, (voxels_batch, masks_batch, attraction_field_batch, _) in config['VERBOSER'](enumerate(dataloader), total=len(dataloader)):
        print(step_idx)
  
        loss_mask, loss_attraction, mse, reg = inner_supervised(model, voxels_batch, masks_batch, attraction_field_batch)
        loss_attraction_normalized = loss_attraction
        loss_mask = 100*loss_mask
        loss = loss_attraction_normalized+loss_mask

        loss.backward()
        opt.step()
        opt.zero_grad()
        averaged_model.update_parameters(model)
    
        batch_mask_losses.append(loss_mask.item())
        batch_att_losses.append(loss_attraction_normalized.item())
        batch_mse_losses.append(mse.item())
        batch_reg_losses.append(reg.item())

    print('updating')

    datasampler = VoxelRandomSampler( config['VOXELING_OPTIONS']['voxel_shape'],
                                      dataset.keys,
                                      dataset.shapes,
                                      config['N_ITERATIONS'] * config['BATCH_SIZE']//4)

    dataloader = DataLoader( dataset,
                             batch_size= config['BATCH_SIZE'],
                             sampler=datasampler,
                             collate_fn=collate_fn,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=1 )
    torch.optim.swa_utils.update_bn(dataloader, averaged_model, device=torch.device('cuda'))

    return np.mean(batch_mask_losses), np.mean(batch_att_losses), np.mean(batch_mse_losses), np.mean(batch_reg_losses)

def inner_val_loop(model, dataset, epoch, phase):
    model.eval()
    save_pred_mask=[]
    save_mask = []
    save_image = []
    save_pred_attraction = []
    sd1mm_arr = []
    sd3mm_arr = []
    assd_arr = []
    hd_arr = []

    for idx in config['VERBOSER'](np.arange(len(dataset)), total=len(dataset)):

        key = dataset.keys[idx]
        shape = dataset.shapes[idx]

        datasampler = VoxelSequentialSampler( config['VOXELING_OPTIONS']['voxel_shape'],
                                              [key],
                                               [shape], 
                                              config['VOXELING_OPTIONS']['steps'] )

        dataloader = DataLoader( dataset,
                                 batch_size=config['BATCH_SIZE'], 
                                 sampler=datasampler,
                                 collate_fn=collate_fn,
                                 num_workers=config['NJOBS'],
                                 pin_memory=False,
                                 prefetch_factor=1 )
        imgs = dataset.imgs[key]
        masks = dataset.masks[key].astype(int)
        save_mask.append(masks[:config['Z_OUTPUT'],:,:,:])
        save_image.append(imgs[:config['Z_OUTPUT'],:,:])
        attraction_field = dataset.attraction_fields[key]
        prob_masks = np.zeros((2, *shape))
        prob_attraction = np.zeros((3, *shape))
        for voxels_batch, _, __, selections in tqdm(dataloader):
            with torch.no_grad():
                voxels_batch = voxels_batch.to(config['DEVICE'])
                pred_mask_batch, pred_attraction_batch = model(voxels_batch)
                prob_attraction[(0,*selections[-1][-1])] += pred_attraction_batch[0][0].cpu().data.numpy()
                prob_attraction[(1,*selections[-1][-1])] += pred_attraction_batch[0][1].cpu().data.numpy()
                prob_attraction[(2,*selections[-1][-1])] += pred_attraction_batch[0][2].cpu().data.numpy()
                prob_masks[(0,*selections[-1][-1])] += pred_mask_batch[0][0].cpu().data.numpy()
                prob_masks[(1,*selections[-1][-1])] += pred_mask_batch[0][1].cpu().data.numpy()

        if phase =='test':
            if not os.path.exists(config['TEST_OUTPUT'] + '/output'):
                os.mkdir(config['TEST_OUTPUT'] + '/output')
                os.mkdir(config['TEST_OUTPUT'] + '/output' + '/pred_attractions')
                os.mkdir(config['TEST_OUTPUT'] + '/output' + '/pred_masks')

            np.savez_compressed(config['TEST_OUTPUT'] + '/output' + '/pred_attractions' + f'/{key}_pred_attraction', data = prob_attraction)
            np.savez_compressed(config['TEST_OUTPUT'] + '/output' + '/pred_masks' + f'/{key}_pred_masks', data = prob_masks)
            if not os.path.exists(config['TEST_OUTPUT'] + '/images'):
                os.mkdir(config['TEST_OUTPUT'] + '/images')
            np.savez_compressed(config['TEST_OUTPUT'] + '/images' + f'/{key}_image', data = imgs)
            if not os.path.exists(config['TEST_OUTPUT'] + '/masks'):
                os.mkdir(config['TEST_OUTPUT'] + '/masks')
            np.savez_compressed(config['TEST_OUTPUT'] + '/masks' + f'/{key}_mask', data = masks)
            
        if epoch == 0:
            inference.true_centerlines(attraction_field, key, idx, phase, config)

        inference.greedy_k_centers(prob_attraction, prob_masks, epoch, key, idx, phase, config)
        inference.non_maximum_suppression(prob_attraction, prob_masks, epoch, key, idx, phase, config)
        # inference.mag_and_field_alg(prob_attraction, prob_masks, epoch,key,  idx, phase, config)
        # inference.mag(prob_attraction, prob_masks, epoch, idx, phase, config)
        # inference.field(prob_attraction, prob_masks, epoch, idx, phase, config)


        sd1mm , sd3mm, assd, hd = metrics.metrics(epoch, key, idx, 'nms', phase, config)
        _, _, _, _ = metrics.metrics(epoch, key, idx, 'k_centers', phase, config)
        # sd1mm , sd3mm, assd, hd  = metrics.metrics(epoch, idx, 'mag', phase, config)
        # sd1mm , sd3mm, assd, hd = metrics.metrics(epoch, idx, 'mag', phase, config)
        # sd1mm , sd3mm, assd, hd = metrics.metrics(epoch, idx, 'field', phase, config)

        sd1mm_arr.append(sd1mm)
        sd3mm_arr.append(sd3mm)
        assd_arr.append(assd)
        hd_arr.append(hd)

    if phase =='test':
        return np.mean(sd1mm_arr), np.mean(sd3mm_arr),np.mean(assd_arr), np.mean(hd_arr), save_image, save_mask, save_pred_mask, save_pred_attraction
    else:
        return np.mean(sd1mm_arr), np.mean(sd3mm_arr),np.mean(assd_arr), np.mean(hd_arr), save_image, save_mask

def fit(model, averaged_model, data):

    model.to(config['DEVICE'])
    averaged_model.to(config['DEVICE'])

    opt = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], eps=1e-8)

    epochs_without_going_up = 0
    best_score = 0
    best_state = copy.deepcopy(averaged_model.state_dict())
    loss_mask_arr= []
    loss_att_arr = []
    loss_mse_arr=[]
    loss_reg_arr=[]

    val_sd1mm = []
    val_sd3mm = []
    val_assd = []
    val_hd = []


    for epoch in range(config['EPOCHS']):
        start_time = time.perf_counter()

        loss_mask, loss_attraction, mse, reg = inner_train_loop( model, averaged_model,
                                 opt,
                                 data['train'])

        loss_mask_arr.append(loss_mask)
        loss_att_arr.append(loss_attraction)
        loss_mse_arr.append(mse)
        loss_reg_arr.append(reg)

        np.save(config['TRAIN_OUTPUT'] + '/loss_mask', loss_mask_arr)
        np.save(config['TRAIN_OUTPUT']+ '/loss_attraction', loss_att_arr)
        np.save(config['TRAIN_OUTPUT']+ '/loss_mse', loss_mse_arr)
        np.save(config['TRAIN_OUTPUT']+ '/loss_reg', loss_reg_arr)

        phase = 'val'
        sd1mm, sd3mm, assd, hd, save_image, save_mask = inner_val_loop(averaged_model, data[phase], epoch, phase)

        val_sd1mm.append(sd1mm)
        val_sd3mm.append(sd3mm)
        val_assd.append(assd)
        val_hd.append(hd)


        
        if epoch == 0:
            if not os.path.exists(config['VAL_OUTPUT'] + '/images'):
                os.mkdir(config['VAL_OUTPUT'] + '/images')
            np.savez_compressed(config['VAL_OUTPUT'] + f'/images/save_images', data = np.stack(save_image[:]))
            np.savez_compressed(config['VAL_OUTPUT'] + f'/images/save_masks', data =np.stack(save_mask[:]))
        if not os.path.exists(config['VAL_OUTPUT']+ '/metrics'):
            os.mkdir(config['VAL_OUTPUT'] + '/metrics')
        np.save(config['VAL_OUTPUT'] + '/metrics' + '/sd1mm', val_sd1mm)
        np.save(config['VAL_OUTPUT'] + '/metrics' +'/sd3mm', val_sd3mm)
        np.save(config['VAL_OUTPUT'] + '/metrics' +'/assd', val_assd)
        np.save(config['VAL_OUTPUT'] + '/metrics'+ '/hd', val_hd)

        if best_score < sd3mm:
            best_score = sd3mm
            best_state = copy.deepcopy(averaged_model.state_dict())
            epochs_without_going_up = 0

            store(averaged_model)
        else:
            epochs_without_going_up += 1

        if epochs_without_going_up == config['STOP_EPOCHS']:
            print(epochs_without_going_up)
            print('overfitting')
            break

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        config['LOGGER'].info(f'elapsed time {elapsed_time:.2f} s')
        config['LOGGER'].info(f'epoch without improve {epochs_without_going_up}')

    averaged_model.load_state_dict(best_state)

    phase = 'test'

    sd1mm, sd3mm, assd, hd, save_image, save_mask, save_pred_mask, save_pred_attraction = inner_val_loop(averaged_model, data[phase], 0, phase)

    test_results = {'sd1mm': sd1mm, 'sd3mm': sd3mm, 'assd': assd,\
                                        'hd': hd}
    
    with open(config['ROOTDIR'] +'/test_results.json', 'w') as file:
        json.dump(test_results, file)

def load(model):
    state = torch.load(config['MODELNAMEINPUT'], map_location=config['DEVICE'])
    model.load_state_dict(state)
def store(model):
    state = model.state_dict()
    path = config['MODELNAME']
    torch.save(state, path)

def get_ema_avg_fn(decay=0.99):
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1 - decay) * current_param

    return ema_update

@click.command()
@click.option('--rootdir', '-rd', type=str, default = './rootdir')
@click.option('--modelnameinput', '-mni', type=str, default='./rootdir/averaged_model.pth')
@click.option('--batch_size', '-bs', type=int, default=1)
@click.option('--n_iterations', '-ni', type=int, default=1000, help='The number of iteration per epoch')
@click.option('--epochs', '-e', type=int, default=100, help='The number of epoch per train loop')
@click.option('--stop_epochs', '-se', type=int, default=5)
@click.option('--learning_rate', '-lr', type=float, default=1e-3)
@click.option('--logger_type', '-lt', type=click.Choice(['stream', 'file'], case_sensitive=False), default='stream')
@click.option('--njobs', type=int, default=25, help='The number of jobs to run in parallel.')
@click.option('--verbose', is_flag=True, default = True, help='Whether progress bars are showed')

def main(**kwargs):
    init_determenistic()
    init_global_config(**kwargs)
    # config['DATAPATH'] = config['ROOTDIR'] + '/preprocessed_data'
    config['DATAPATH'] ='./preprocessed_data'
    config['TRAIN_OUTPUT'] = config['ROOTDIR'] + '/train_output'
    config['VAL_OUTPUT'] = config['ROOTDIR'] + '/val_output'
    config['TEST_OUTPUT'] = config['ROOTDIR'] + '/test_output'
    config['LOADER_OPTIONS'] = yaml.load(open(config['ROOTDIR'] + '/loader_options.yaml',).read(), yaml.SafeLoader)
    config['VOXELING_OPTIONS'] = yaml.load(open(config['ROOTDIR'] + '/voxeling_options.yaml',).read(), yaml.SafeLoader)
    config['SPLIT_OPTIONS'] = yaml.load(open(config['ROOTDIR'] + '/split_options.yaml',).read(), yaml.SafeLoader)
    config['MODELNAME'] = config['ROOTDIR'] + '/averaged_model_nms.pth'
    config['Z_OUTPUT'] = 124 #max image depth to save results
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    config['DEVICE'] = device
    
    if not os.path.exists(config['TRAIN_OUTPUT']):
        os.mkdir(config['TRAIN_OUTPUT'])
    if not os.path.exists(config['VAL_OUTPUT']):
        os.mkdir(config['VAL_OUTPUT'])
    if not os.path.exists(config['TEST_OUTPUT']):
        os.mkdir(config['TEST_OUTPUT'])

    for key in config:
        if key != 'LOGGER':
            config['LOGGER'].info(f'{key} {config[key]}')

    data = load_data()

    config['LOGGER'].info(f'create model')
    model = my_unet()
    averaged_model = tsu.AveragedModel(model, avg_fn=get_ema_avg_fn())

    # load(averaged_model) #if you have pretrained model

    config['LOGGER'].info(f'fit model')
    fit(model, averaged_model, data)

    fit(model, averaged_model, data)

    config['LOGGER'].info(f'store model')
    store(averaged_model)

if __name__ == '__main__':
    main()
