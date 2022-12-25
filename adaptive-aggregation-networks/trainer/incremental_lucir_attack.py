##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Training code for LUCIR """
import os
import random
import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


from utils.misc import *
from utils.process_fp import process_inputs_fp


from utils.attacks.attack_BSS import AttackBSS
from utils.loss_utils.pearson_loss import pearson_loss


import wandb


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor




cur_features = []
ref_features = []
old_scores = []
new_scores = []

def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y

def incremental_train_and_eval(the_args, epochs, fusion_vars, ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, \
    tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, trainloader, testloader, iteration, \
    start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list, the_lambda, dist, \
    K, lw_mr, balancedloader, fix_bn=False, weight_per_class=None, device=None,
                               attackpool=None, memoryset=None, X_train=None, map_Y_train=None, trainset=None, anchorset=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set the 1st branch reference model to the evaluation mode
    ref_model.eval()

    # Get the number of old classes
    num_old_classes = ref_model.fc.out_features

    # Get the features from the current and the reference model
    handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
    handle_cur_features = b1_model.fc.register_forward_hook(get_cur_features)
    handle_old_scores_bs = b1_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
    handle_new_scores_bs = b1_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

    # If the 2nd branch reference is not None, set it to the evaluation mode
    if iteration > start_iteration+1:
        ref_b2_model.eval()



    """
    ADD Attack images
    只在memory中做，将image attack到得分第二高的label
    """
    unorm = UnNormalize(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023))
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023)),
    ]
    trsf = transforms.Compose([*common_trsf])

    attack = AttackBSS(targeted=True, num_steps=10, max_epsilon=16, step_alpha=0.3, device=device, norm=2)


    map_Y_protoset_cumuls_this_step = np.concatenate(Y_protoset_cumuls, axis=0)
    X_protoset_cumuls_this_step = np.concatenate(X_protoset_cumuls, axis=0)
    map_Y_protoset_cumuls_this_step = np.array([order_list.index(i) for i in map_Y_protoset_cumuls_this_step])
    memoryset.data = X_protoset_cumuls_this_step.astype('uint8')
    memoryset.targets = map_Y_protoset_cumuls_this_step
    memoryloader = torch.utils.data.DataLoader(memoryset, batch_size=the_args.test_batch_size, shuffle=False, num_workers=the_args.num_workers)
    attackimage, attacktarget = [], []

    # os.mkdir(os.path.join('./attacked', str(iteration)))

    for i, (oldimages, oldtargets) in enumerate(memoryloader):
        oldimages = oldimages.to(device)
        # select class to attack
        if iteration == start_iteration + 1:
            ref_outputs = ref_model(oldimages)
        else:
            ref_outputs, _ = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, oldimages)

        class_idx = ref_outputs
        class_idx = class_idx.sort(dim=1, descending=True)[1][:, 1]
        attacked_images = attack.run(ref_model, oldimages, class_idx, the_args, ref_fusion_vars, ref_b2_model)

        if iteration == start_iteration + 1:
            predict = ref_model(attacked_images)
        else:
            predict, _ = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, attacked_images)

        pred_score, pred_idx = predict.sort(dim=1, descending=True)
        # pred_score = pred_score[:, 0]
        pred_idx = pred_idx[:, 0].cpu()
        # selectedidx = (pred_idx != oldtargets).nonzero().squeeze()  # 这个是620之前的做法，因为没有保证attack成功，就是pred_idx有可能！=class_idx
        selectedidx = (pred_idx == class_idx.cpu()).nonzero().squeeze()
        # import pdb;pdb.set_trace()

        for ii in selectedidx:
            inversed_attack_image = (unorm(attacked_images[ii]).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            origin_image = (unorm(oldimages[ii]).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if iteration == start_iteration + 1:
                at_pred = ref_model(trsf(inversed_attack_image).unsqueeze(0).to(device))
            else:
                at_pred, _ = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, trsf(inversed_attack_image).unsqueeze(0).to(device))

            if at_pred.max(1)[1] == pred_idx[ii]:
                attackimage.append(inversed_attack_image)
                attacktarget.append(pred_idx[ii])
                # attackimage.append(origin_image)
                # attacktarget.append(oldtargets[ii])
                # Code for saving attacked images
                # save image plt
                # im = Image.fromarray(inversed_attack_image)
                # im.save(os.path.join(os.path.join('./attacked', str(iteration)), '{}.{}_ak_{}.jpg'.format(i,ii,pred_idx[ii])))
                # im = Image.fromarray(origin_image)
                # im.save(os.path.join(os.path.join('./attacked', str(iteration)), '{}.{}_or_{}.jpg'.format(i,ii,oldtargets[ii])))

    allattacked = np.stack(attacktarget, axis=0)
    for cls_id in range(map_Y_protoset_cumuls_this_step.max()):
        print("Class {} has {} images".format(cls_id, sum(allattacked == cls_id)))


    """
    ADD Attack images
    只在memory中做，将新类数据attack到memory中类别
    """
    # unorm = UnNormalize(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023))
    # common_trsf = [
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023)),
    # ]
    # trsf = transforms.Compose([*common_trsf])
    # attack = AttackBSS(targeted=True, num_steps=50, max_epsilon=16, step_alpha=0.3, device=device, norm=2)
    #
    # map_Y_protoset_cumuls_this_step = np.concatenate(Y_protoset_cumuls, axis=0)
    # X_protoset_cumuls_this_step = np.concatenate(X_protoset_cumuls, axis=0)
    # map_Y_protoset_cumuls_this_step = np.array([order_list.index(i) for i in map_Y_protoset_cumuls_this_step])
    #
    # memoryset.data = X_train.astype('uint8')
    # memoryset.targets = map_Y_train
    # memoryloader = torch.utils.data.DataLoader(memoryset, batch_size=the_args.train_batch_size, shuffle=True, num_workers=the_args.num_workers)
    #
    # attackimage, attacktarget = [], []
    # for i, (oldimages, oldtargets) in enumerate(memoryloader):
    #     oldimages = oldimages.to(device)
    #     # select class to attack
    #     if iteration == start_iteration + 1:
    #         ref_outputs = ref_model(oldimages)
    #     else:
    #         ref_outputs, _ = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, oldimages)
    #
    #     class_idx = ref_outputs
    #     class_idx = class_idx.sort(dim=1, descending=True)[1][:, 1]
    #     attacked_images = attack.run(ref_model, oldimages, class_idx, the_args, ref_fusion_vars, ref_b2_model)
    #
    #     if iteration == start_iteration + 1:
    #         predict = ref_model(attacked_images)
    #     else:
    #         predict, _ = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, attacked_images)
    #
    #     pred_score, pred_idx = predict.sort(dim=1, descending=True)
    #     pred_idx = pred_idx[:, 0].cpu()
    #     selectedidx = (pred_idx == class_idx.cpu()).nonzero().squeeze()
    #     # print(selectedidx)
    #     # import pdb;pdb.set_trace()
    #     for ii in selectedidx:
    #         inversed_attack_image = (unorm(attacked_images[ii]).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #         origin_image = (unorm(oldimages[ii]).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #         if iteration == start_iteration + 1:
    #             at_pred = ref_model(trsf(inversed_attack_image).unsqueeze(0).to(device))
    #         else:
    #             at_pred, _ = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, trsf(inversed_attack_image).unsqueeze(0).to(device))
    #
    #         if at_pred.max(1)[1] == pred_idx[ii]:
    #             attackimage.append(inversed_attack_image)
    #             attacktarget.append(pred_idx[ii])

    # import pdb;pdb.set_trace()



    # trainset.data = np.concatenate((X_train, np.stack(attackimage, axis=0)), axis=0).astype('uint8')
    # trainset.targets = np.concatenate((map_Y_train, np.stack(attacktarget, axis=0)), axis=0)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=the_args.train_batch_size, shuffle=True, num_workers=the_args.num_workers)

    anchorset.data = np.concatenate((X_protoset_cumuls_this_step, np.stack(attackimage, axis=0)), axis=0).astype('uint8')
    anchorset.targets = np.concatenate((map_Y_protoset_cumuls_this_step, np.stack(attacktarget, axis=0)), axis=0)

    # import pdb;pdb.set_trace()
    # only use memory images
    # anchorset.data = X_protoset_cumuls_this_step.astype('uint8')
    # anchorset.targets = map_Y_protoset_cumuls_this_step

    anchorloader = torch.utils.data.DataLoader(anchorset, batch_size=the_args.train_batch_size, shuffle=True,
                                num_workers=the_args.num_workers)





    """
    Start to Incremental learning
    """
    # anchorloader = balancedloader
    anchoriter = iter(anchorloader)

    for epoch in range(epochs):
        # Start training for the current phase, set the two branch models to the training mode
        b1_model.train()
        b2_model.train()

        # Fix the batch norm parameters according to the config
        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Set all the losses to zeros
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        # Set the counters to zeros
        correct = 0
        total = 0
    
        # Learning rate decay
        tg_lr_scheduler.step()
        fusion_lr_scheduler.step()

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])
        wandb.log({"loss": tg_lr_scheduler.get_lr()[0], "epoch": epoch, "iteration": iteration,})

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            """
            TPCIL
            """
            try:
                inputsANC, targetsANC = anchoriter.next()
            except:
                anchoriter = iter(anchorloader)
                inputsANC, targetsANC = anchoriter.next()
            anchor_inputs = inputsANC.to(device)
            anchor_targets = targetsANC.to(device)
            _, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, anchor_inputs)

            if iteration == start_iteration+1:
                ref_outputs = ref_model(anchor_inputs)
                loss_kd = pearson_loss(ref_features.detach(), cur_features)*10

            else:
                _, ref_features_new = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, anchor_inputs)
                loss_kd = pearson_loss(ref_features_new.detach(), cur_features)*10
            # loss_kd = torch.zeros(1).to(device)


            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()

            # Forward the samples in the deep networks UAP
            # attacker = random.choice(attackpool)
            # attacker = attacker.to(device)

            # outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs+attacker)
            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)

            # Loss 1: feature-level distillation loss
            # if iteration == start_iteration+1:
            #     ref_outputs = ref_model(inputs)
            #     # ref_outputs = ref_model(inputs+attacker)
            #     loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), torch.ones(inputs.shape[0]).to(device)) * the_lambda
            # else:
            #     ref_outputs, ref_features_new = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, inputs)
            #     # ref_outputs, ref_features_new = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, inputs+attacker)
            #     loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features_new.detach(), torch.ones(inputs.shape[0]).to(device)) * the_lambda
            loss1 = torch.zeros(1).to(device)


            # Loss 2: classification loss
            loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

            # Loss 3: margin ranking loss
            # outputs_bs = torch.cat((old_scores, new_scores), dim=1)
            # assert(outputs_bs.size()==outputs.size())
            # gt_index = torch.zeros(outputs_bs.size()).to(device)
            # gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
            # gt_scores = outputs_bs.masked_select(gt_index)
            # max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
            # hard_index = targets.lt(num_old_classes)
            # hard_num = torch.nonzero(hard_index).size(0)
            # if hard_num > 0:
            #     gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
            #     max_novel_scores = max_novel_scores[hard_index]
            #     assert(gt_scores.size() == max_novel_scores.size())
            #     assert(gt_scores.size(0) == hard_num)
            #     loss3 = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1), torch.ones(hard_num*K).view(-1,1).to(device)) * lw_mr
            # else:
            #     loss3 = torch.zeros(1).to(device)
            loss3 = torch.zeros(1).to(device)

            # Sum up all looses
            loss = loss1 + loss2 + loss3 + loss_kd

            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the training losses and accuracies
        print('Train set: {}, train loss1: {:.4f}, train loss2: {:.4f}, train loss3: {:.4f}, train loss: {:.4f} accuracy: {:.4f}'
              .format(len(trainloader), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss3/(batch_idx+1), train_loss/(batch_idx+1), 100.*correct/total))
        wandb.log({"epoch": epoch, "iteration": iteration,
                   "train_loss1":train_loss1/(batch_idx+1),
                   "train_loss2":train_loss2/(batch_idx+1),
                   "train_loss3":train_loss3/(batch_idx+1),
                   "train_loss":train_loss/(batch_idx+1),
                   "train_acc":100.*correct/total})

        # Update the aggregation weights
        b1_model.eval()
        b2_model.eval()
     
        for batch_idx, (inputs, targets) in enumerate(balancedloader):
            if batch_idx <= 500:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss.backward()
                fusion_optimizer.step()

        # Running the test for this epoch
        b1_model.eval()
        b2_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

        wandb.log({"epoch": epoch, "iteration": iteration,
                   "test_loss":test_loss/(batch_idx+1),
                   "test_acc":100.*correct/total})

    print("Removing register forward hook")
    handle_ref_features.remove()
    handle_cur_features.remove()
    handle_old_scores_bs.remove()
    handle_new_scores_bs.remove()
    return b1_model, b2_model
