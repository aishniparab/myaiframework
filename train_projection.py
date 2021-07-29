import yaml
import os
import argparse
import random
import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from samplers.batch_sampler import BatchSampler
from datasets.bongard_dataset import BongardDataset
from models.linear import Linear
from representation.graph.init import Graph
from utils.debug import test_optimal_decision_rule, sample_x_from_gaussian
from utils.embed_input import embed
from utils.file_processing import save_list_to_file, get_dirname
from utils.train import train, val, reset_weights
import numpy as np

def main(config):
    # declare directories
    save_dir = args.save_dir
    
    if (config['train_args']['train_mode']):
        dir_name = get_dirname(args.experiment_tag, args.version)
        if not os.path.exists(os.path.join(save_dir, dir_name, 'trained_models')): 
            os.makedirs(os.path.join(save_dir, dir_name, 'trained_models'))

    best_model_path = os.path.join(save_dir, dir_name, 'trained_models/best_model.pth')
    last_model_path = os.path.join(save_dir, dir_name, 'trained_models/last_model.pth')
    flip_model_path = lambda i: os.path.join(save_dir, dir_name, 'trained_models/flip_{}_model.pth'.format(str(i)))
    
    # save config
    yaml.dump(config, open(os.path.join(save_dir, dir_name, 'config.yaml'), 'w'))
    
    # for visualization
    writer = SummaryWriter(os.path.join(save_dir, dir_name, 'tensorboard/scalar'))
    
    # for gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Dataset ###
    num_classes, num_context_per_class = config['num_classes'], config['num_context_per_class']
    num_probe_per_class = config['num_probe_per_class']
    
    dataset_name = config['dataset']
    dataset_dir = config['dataset_args']['path']
    
    batch_size = config['batch_size']
    one_hot_size = config['dataset_args']['one_hot_size']

    img_h, img_w = config['dataset_args']['img_h'], config['dataset_args']['img_w']
    img_dim = (img_h, img_w)
    
    seed = config['seed']

    tr_dataset = BongardDataset(random_seed=seed, batch_type='train', img_dim=img_dim, batch_size=batch_size, one_hot_size=one_hot_size, root=dataset_dir)
    val_dataset = BongardDataset(random_seed=seed, batch_type='val', img_dim=img_dim, batch_size=batch_size, one_hot_size=one_hot_size, root=dataset_dir)
    test_dataset = BongardDataset(random_seed=seed, batch_type='test', img_dim=img_dim, batch_size=batch_size, one_hot_size=one_hot_size, root=dataset_dir)

    tr_sampler = BatchSampler(random_seed=seed, labels=tr_dataset.y, batch_size=batch_size)
    val_sampler = BatchSampler(random_seed=seed, labels=val_dataset.y, batch_size=batch_size)
    test_sampler = BatchSampler(random_seed=seed, labels=test_dataset.y, batch_size=batch_size)

    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, sampler=tr_sampler, drop_last=config['dataset_args']['drop_last'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, drop_last=config['dataset_args']['drop_last'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, drop_last=config['dataset_args']['drop_last'])
    
    num_tr_iterations = tr_dataset.__len__()//batch_size
    num_val_iterations = val_dataset.__len__()//batch_size
    
    ### Graph ###
    graph = Graph(dataset_name, img_dim, batch_size, num_classes, num_context_per_class, num_probe_per_class)
    graph.print_info(show_plot=False)
    
    ### Model and Optimizer ###
    print('===================================================')
    # could add config item to load model from file
    # could refactor how you run tests with different model
    if config['model'] == 'linear':
        model = Linear(random_seed=seed, in_dim=config['model_args']['in_dim'], out_dim=config['model_args']['out_dim'])
        print("Model: ", model, "\n")
    if config['model'] == 'project':
        model = MatMul(random_seed=seed, in_dim=config['model_args']['in_dim'], out_dim=config['model_args']['out_dim'])

    if config['optimizer'] == 'adam':
        print("Optimizer: ", config['optimizer'], "\n")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer_args']['lr'], weight_decay=config['optimizer_args']['weight_decay'])

    if config['loss_fn'] == 'cross_entropy':
        print("Loss: ", config['loss_fn'])
        loss_fn = nn.CrossEntropyLoss()
    print('===================================================')
    print("\n")
    
    ### Training & Validation ###
    if config['debug_mode']:
        debug_step = config['debug_args']['debug_step']
        num_samples_per_class = num_context_per_class + num_probe_per_class
        
        ## Run single tests ## 
        # pass labels into model and flip them, compute loss on all (you should see a U shape)
        #test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples_per_class, graph.edge_index, device, img_h, img_w, "labels_only", None, None, None)
        
         #test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples_per_class, graph.edge_index, device, img_h, img_w, "resnet_labels", 1, None, None)
        
        #sample = sample_x_from_gaussian(config['debug_args']['gaussian_args']['mean_left'], 
        #                                                config['debug_args']['gaussian_args']['mean_right'], 
        #                                                config['debug_args']['gaussian_args']['std'], 
        #                                                batch_size, num_samples_per_class, num_classes, 
        #                                                config['debug_args']['gaussian_args']['vector_dim'])
        
        #test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples_per_class, graph.edge_index, device, img_h, img_w, "gaussian", None, sample, config['debug_args']['gaussian_args']['vector_dim'])
        #doesnt work with resnet 
        #test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples_per_class, graph.edge_index, device, img_h, img_w, "resnet_gaussian_labels", 1, sample, config['debug_args']['gaussian_args']['vector_dim'])
        
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_acc = 0
        best_state = None

        batch = next(iter(tr_dataloader))
        data, paths = embed(batch, i, config['loss_args']['mask_context'], graph.edge_index, device, img_h, img_w, "labels_only", None, None, None)
        loss, acc, preds, h = train(data, model, loss_fn, optimizer)
        print("loss: ", loss, "acc: ", acc, "preds: ", preds)

        """
        print("Number of training iterations: ", num_tr_iterations)
        print("Number of validation iterations: ", num_val_iterations)
        # flip labels
        num_flips = num_samples_per_class + num_probe_per_class # total num labels = context + probe
        for i in range(num_flips):
            print('=== Flip: {} ==='.format(i))
            # train each flip 
            for epoch in range(1, config['train_args']['num_epochs'] + 1):
                print('=== Epoch: {} ==='.format(epoch))
                logs = {}

                for phase in ['train', 'val']:
                    # set model state
                    if phase == 'train':
                        iter_obj = iter(tr_dataloader)
                        model.train()
                        model = model.to(device)
                    else:
                        iter_obj = iter(val_dataloader)
                        model.eval()

                    for batch in tqdm(iter_obj):
                        data, paths = embed(batch, i, config['loss_args']['mask_context'], graph.edge_index, device, img_h, img_w, "labels_only", None, None, None)
                        if phase == 'train':
                            loss, acc, preds, h = train(data, model, loss_fn, optimizer)
                            train_loss.append(loss.item())
                            train_acc.append(acc)
                        else: # val
                            loss, acc, preds, h = val(data, model, loss_fn)
                            val_loss.append(loss.item())
                            val_acc.append(acc)
                    # end for loop over batches
                    # compute avg loss and acc over epoch
                    if phase == 'train':
                        print('Avg Batch Train Loss: {}, Avg Batch Train Acc: {}'.format(np.mean(train_loss[-num_tr_iterations:]), np.mean(train_acc[-num_tr_iterations:]))) 
                        logs['loss'] = np.mean(train_loss[-num_tr_iterations:])
                        logs['acc'] = np.mean(train_acc[-num_tr_iterations:])
                        #writer.add_scalar('flip_{}/train_loss'.format(str(i)), logs['loss'], epoch)
                        #writer.add_scalar('flip_{}/train_acc'.format(str(i)), logs['acc'], epoch)
                        writer.add_scalars('flip_{}/train'.format(str(i)), {'loss': logs['loss'], 'acc': logs['acc']}, epoch)
                    else:
                        print('Avg Batch Val Loss: {}, Avg Batch Val Acc: {}'.format(np.mean(val_loss[-num_val_iterations:]), np.mean(val_acc[-num_val_iterations:]))) 
                        logs['val_loss'] = np.mean(val_loss[-num_val_iterations:])
                        logs['val_acc'] = np.mean(val_acc[-num_val_iterations:])
                        #writer.add_scalar('flip_{}/val_loss'.format(str(i)), logs['val_loss'], epoch)
                        #writer.add_scalar('flip_{}/val_acc'.format(str(i)), logs['val_acc'], epoch)
                        writer.add_scalars('flip_{}/val'.format(str(i)), {'loss': logs['loss'], 'acc': logs['acc']}, epoch)
                        # save model every two epochs
                        if epoch % 2 == 0:
                            #torch.save(model.state_dict(), last_model_path) # this will get rewritten as model accumulates trains over flip
                            torch.save(model.state_dict(), flip_model_path(i)) # this will remain unique to the flip
                        # save model if mean val acc is best so far
                        if np.mean(val_acc) >= best_acc:
                            best_state = model.state_dict()
                            torch.save(best_state, best_model_path)
                            best_acc = np.mean(val_acc)
                # end for loop over train/val phases
                # save loss log at the end of epoch
                for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
                  save_list_to_file(os.path.join(save_dir, dir_name, name + '.txt'), locals()[name])

                torch.cuda.empty_cache() 
                torch.autograd.set_detect_anomaly(True)  
            # end for loop over epochs
            writer.flush()
            # reset model parameters for next flip 
            reset_weights(model)    
        # end for loop over num_flips
        
        ### ADD CODE TO RESUME LAST STATE ###
        """
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--experiment_tag')
    parser.add_argument('--version')
    parser.add_argument('--save_dir', default='./save')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    main(config)