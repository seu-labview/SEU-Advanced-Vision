# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import shutil
from pose_6D_neuralnet import pose_6d_neuralnet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # why not google or baidu " sys.path.append"? 
#if you are intersted, you can uncommented the following sentences
# c = os.path.abspath(__file__)
# b = os.path.dirname(os.path.abspath(__file__))
# a = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(c)
# print(b)
# print(a)

import utils as utils
import dataset
def train(epoch):
    global processed_batches
    #initialize timer
    t0 = time.time()
    #get the dataloader for training dataset
    train_loader = torch.utils.data.DataLoader()


if __name__ == "__main__":
    utils.logging('QJ')
    print(os.path.join('/aa','bb/'))
    # Training settings
    # argv_len helps judge the number of argv given in the python3 script, when it is running in the terminal
    argv_len = len(sys.argv)
    if argv_len == 1:
        datacfg = 'datacfg.data'
        netcfg = 'netcfg.data'
        weightcfg = 'weightcfg.data'
    if argv_len == 4:
        datacfg = sys.argv[1]
        netcfg = sys.argv[2]
        weightcfg = sys.argv[3]

    # Parse configuration files
    print(torch.cuda.is_available())
    print(torch.__version__ )
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    datacfg_options = utils.read_data_cfg(datacfg)
    datacfg_options['gpus'] = '0'
    datacfg_options['num_workers'] = '10'
    
    netcfg_options = utils.read_net_cfg(netcfg)[0]

    trainlist      = datacfg_options['train']
    nsamples      = utils.read_train_list(trainlist)
    gpus          = datacfg_options['gpus']  
    num_workers   = int(datacfg_options['num_workers'])
    backupdir     = datacfg_options['backup']
    if not os.path.exists( backupdir ):
        os.makedirs( backupdir )
    batch_size    = int(netcfg_options['batch'])
    max_batches   = int(netcfg_options['max_batches'])
    learning_rate = float(netcfg_options['learning_rate'])
    momentum      = float(netcfg_options['momentum'])
    decay         = float(netcfg_options['decay'])
    steps         = [float(step) for step in netcfg_options['steps'].split(',')]
    scales        = [float(scale) for scale in netcfg_options['scales'].split(',')]
    bg_file_names = utils.get_all_files('./VOCdevkit/VOC2012/JPEGImages')

    # Train parameters
    max_epochs    = 700 # max_batches*batch_size/nsamples+1
    use_cuda      = True
    seed          = int(time.time())
    eps           = 1e-5
    save_interval = 10 # epoches
    dot_interval  = 70 # batches
    best_acc       = -1 

    # Test parameters
    conf_thresh   = 0.05
    nms_thresh    = 0.4
    match_thresh  = 0.5
    iou_thresh    = 0.5
    im_width      = 640
    im_height     = 480 
    
    # Specify which gpus to use
    # here we set the gpu which is running the net
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    # Specifiy the model and the loss
    model = pose_6d_neuralnet(netcfg)
    region_loss = model.loss

    # Model settings
    # model.load_weights(weightfile)
    # model.load_weights_until_last(weightcfg) # when train the net remember to uncomment
    model.print_network()
    model.seen        = 0
    region_loss.iter  = model.iter
    region_loss.seen  = model.seen
    processed_batches = model.seen//batch_size
    init_width        = model.width
    init_height       = model.height
    init_epoch        = model.seen//nsamples 

    # Variable to save
    training_iters          = []
    training_losses         = []
    testing_iters           = []
    testing_errors_pixel    = []
    testing_accuracies      = []

    # Specify the number of workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

     # Pass the model to GPU
    if use_cuda:
        # model = model.cuda() 
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda() # Multiple GPU parallelism
    #hower, we have only one gpu in the mobile computer
    
    # Get the optimizer
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
    evaluate = False
    if evaluate:
        utils.logging('evaluating ...')
        test(0, 0)
    else:
        for epoch in range(init_epoch, max_epochs): 
            # TRAIN
            niter = train(epoch)
            # TEST and SAVE
            if (epoch % 20 == 0) and (epoch is not 0): 
                test(niter)
                utils.logging('save training stats to %s/costs.npz' % (backupdir))
                np.savez(os.path.join(backupdir, "costs.npz"),
                    training_iters=training_iters,
                    training_losses=training_losses,
                    testing_iters=testing_iters,
                    testing_accuracies=testing_accuracies,
                    testing_errors_pixel=testing_errors_pixel) 
                if (np.mean(testing_accuracies[-5:]) > best_acc ):
                    best_acc = np.mean(testing_accuracies[-5:])
                    utils.logging('best model so far!')
                    utils.logging('save weights to %s/model.weights' % (backupdir))
                    model.save_weights('%s/model.weights' % (backupdir))
        shutil.copy2('%s/model.weights' % (backupdir), '%s/model_backup.weights' % (backupdir))