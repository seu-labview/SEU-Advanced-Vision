# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import time

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)

def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel() # return thr number of parameter in the element 
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))     
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b])) 
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]))  
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]))   
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight.data)) 
    start = start + num_w 
    return start

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))   
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight.data)) 
    start = start + num_w
    return start

def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))     
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(fc_model.weight.data))   
    start = start + num_w 
    return start

def read_data_cfg(datacfg):
    '''
        all commented sentences are printing tests which help you learn about reading files of python3 
    '''
    options = dict()
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()
    #print(lines)
    for line in lines:
        line = line.strip()
        # print(line)
        if line == '':
            #print("blank")
            continue
        key,value = line.split('=')
        
        key = key.strip()
        value = value.strip()
        options[key] = value
        # if key:
        #     print(key)
        # else :
        #     print("blank_key")
        # if value:    
        #     print(value)
        # else :
        #     print("blank_value")
    return options

def read_net_cfg(netcfg):
    '''
        it likes the function read_data_cfg
    '''
    blocks = []
    block =  None
    with open(netcfg, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        if line == '' or line[0] == '#':
            continue 
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
    blocks.append(block)
    return blocks

def read_train_list(trainpath):
    '''
        read the train data list and return how many data in the list
    '''
    count = 0
    with open(trainpath, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        else:
            count += 1
    return count
    
def get_all_files(directory):
    '''
        read all files in a directory and return the absolute path
    '''
    files = []
    files_count = 0
    for f in os.listdir(directory):

        if os.path.isfile(os.path.join(directory, f)):
            files.append(os.path.join(directory, f))
            files_count += 1     
    return files

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def corner_confidences9(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a torch.FloatTensor of shape (nA,) with 9 confidence values 
    '''
    shape = gt_corners.size()
    nA = shape[1]  
    dist = gt_corners - pr_corners
    dist = dist.t().contiguous().view(nA, 9, 2)
    dist[:, :, 0] = dist[:, :, 0] * im_width
    dist[:, :, 1] = dist[:, :, 1] * im_height

    eps = 1e-5
    distthresh = torch.FloatTensor([th]).repeat(nA, 9) 
    dist = torch.sqrt(torch.sum((dist)**2, dim=2)).squeeze() # nA x 9
    mask = (dist < distthresh).type(torch.FloatTensor)
    conf = torch.exp(sharpness*(1 - dist/distthresh))-1  # mask * (torch.exp(math.log(2) * (1.0 - dist/rrt)) - 1)
    conf0 = torch.exp(sharpness*(1 - torch.zeros(conf.size(0),1))) - 1
    conf = conf / conf0.repeat(1, 9)
    # conf = 1 - dist/distthresh
    conf = mask * conf  # nA x 9
    mean_conf = torch.mean(conf, dim=1)
    return mean_conf

def print_configure_of_net(blocks):
    print('layer     filters    size              input                output')
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters =[]
    out_widths =[]
    out_heights =[]
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)//2 if is_pad else 0
            width = (prev_width + 2*pad - kernel_size)//stride + 1
            height = (prev_height + 2*pad - kernel_size)//stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width//stride
            height = prev_height//stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width//stride
            height = prev_height//stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            print(layers)
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            print(layers)
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                print(len(out_widths))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert(prev_width == out_widths[layers[1]])
                assert(prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'region':
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters,  filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))