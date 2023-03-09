import os
import time
import math
import utils
from tqdm import tqdm
import logging
from torch.autograd import Variable
from evaluate import evaluate, evaluate_kd
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import random


def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    loss function for Knowledge Distillation (KD)
    """
    alpha = 0.95#params.alpha
    T = 6#params.temperature

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
    KD_loss =  (1. - alpha)*loss_CE + alpha*D_KL

    return KD_loss

def tp(arr,s,l):
  n = len(arr)
  # find the 20% index number (round down to the nearest integer)
  index = int(n * .02)
  index2 = int(n * .08)
  # find the first "index" number of least values
  least_values = heapq.nsmallest(index, arr)
  g_values = heapq.nlargest(index2, arr)
  # find the indices of the least values in the original array
  least_value_indices = [i for i in range(n) if arr[i] in least_values]
  g_value_indices = [i for i in range(n) if arr[i] in g_values]
  return g_value_indices + least_value_indices

# KD train and evaluate
def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                       loss_fn_kd, warmup_scheduler, params, args, restore_file=None):
    """
    KD Train the model and evaluate every epoch.
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)


    # tensorboard setting
    log_dir = args.model_dir + '/tensorboard/'
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    teacher_model.eval()
    teacher_acc = evaluate_kd(teacher_model, val_dataloader, params)
    print(">>>>>>>>>The teacher accuracy: {}>>>>>>>>>".format(teacher_acc['accuracy']))

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    for epoch in range(params.num_epochs):

        if epoch > 0:   # 0 is the warm up epoch
            scheduler.step()
        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1, params.num_epochs, optimizer.param_groups[0]['lr']))

        # KD Train
        train_acc, train_loss = train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader, warmup_scheduler, params, args, epoch)
        # Evaluate
        val_metrics = evaluate_kd(model, val_dataloader, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=args.model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            file_name = "eval_best_result.json"
            best_json_path = os.path.join(args.model_dir, file_name)
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(args.model_dir, "eval_last_result.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # WandB 
        wandb.log({"Training_accuracy": train_acc, "Train_loss": train_loss})
        wandb.log({"Test_accuracy": val_metrics['accuracy'], "Test_loss": val_metrics['loss']})

        # Tensorboard
        writer.add_scalar('Train_accuracy', train_acc, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Test_accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Test_loss', val_metrics['loss'], epoch)
        # export scalar data to JSON for external processing
    writer.close()


# Defining train_kd functions
def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, warmup_scheduler, params, args, epoch, flag=None):
    """
    KD Train the model on `num_steps` batches
    """
    # set model to training mode
    model.train()
    teacher_model.eval()
    loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    total = 0
    correct = 0
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if epoch<=0:
                warmup_scheduler.step()

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output, fetch teacher output, and compute KD loss
            output_batch = model(train_batch)

            # get one batch output from teacher model
            output_teacher_batch = teacher_model(train_batch).cuda()
            output_teacher_batch = Variable(output_teacher_batch, requires_grad=False)

            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            _, predicted = output_batch.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()
            # update the average loss
            loss_avg.update(loss.data)
            losses.update(loss.item(), train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    acc = 100.*correct/total
    logging.info("- Train accuracy: {acc:.4f}, training loss: {loss:.4f}".format(acc = acc, loss = losses.avg))
    return acc, losses.avg

def subtract_lists(list1, list2):
    return [x1 - x2 for (x1, x2) in zip(list1, list2)]
# normal training
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, params, model_dir, warmup_scheduler, args, restore_file=None):
    """
    Train the model and evaluate every epoch.
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # dir setting, tensorboard events will save in the dirctory
    log_dir = args.model_dir + '/base_train/'
    if args.regularization:
        log_dir = args.model_dir + '/Tf-KD_regularization/'
        model_dir = log_dir
    elif args.label_smoothing:
        log_dir = args.model_dir + '/label_smoothing/'
        model_dir = log_dir
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_acc = 0.0
    import copy
    ind = []
    teacher_model = model#copy.deepcopy(model)
    #restore_path = os.path.join(args.model_dir,  'teacher' + '.pth.tar')
    #utils.load_checkpoint(restore_path, teacher_model, optimizer)
    # learning rate schedulers
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    val = []
    s = 0
    l = 0
    #val2 = infer(teacher_model, optimizer, loss_fn, train_dataloader,ind, params, 0, warmup_scheduler, args)
    for epoch in range(params.num_epochs):
        if epoch > 0:   # 1 is the warm up epoch
            scheduler.step(epoch)

        if epoch == 0:
            ind = []

        # Run one epoch
        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1, params.num_epochs, optimizer.param_groups[0]['lr']))
        

        # compute number of batches in one epoch (one full pass over the training set)
        
        val2 = val
        #val2 = infer(teacher_model, optimizer, loss_fn, train_dataloader,ind, params, epoch, warmup_scheduler, args)
        train_acc, train_loss = train(model, optimizer, loss_fn, train_dataloader,ind, params,teacher_model, epoch, warmup_scheduler, args)
        val =  infer(model, optimizer, loss_fn, train_dataloader,ind, params,teacher_model, epoch, warmup_scheduler, args)

        res = subtract_lists(val,val2)
        

            
        ind = tp(res,s,l)
        print(ind)


        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, params, args)

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=model_dir)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "eval_best_results.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "eval_last_results.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # WandB 
        wandb.log({"Training_accuracy": train_acc, "Train_loss": train_loss})
        wandb.log({"Test_accuracy": val_metrics['accuracy'], "Test_loss": val_metrics['loss']})

        # Tensorboard
        writer.add_scalar('Train_accuracy', train_acc, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Test_accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Test_loss', val_metrics['loss'], epoch)
    writer.close()


# normal training function
def train(model, optimizer, loss_fn, dataloader,ind, params,teacher_model, epoch, warmup_scheduler, args):
    """
    Noraml training, without KD
    """

    # set model to training mode
    model.train()
    loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    total = 0
    correct = 0
    val = []
    kl = []
   

    # Use tqdm for progress bar
    
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            if epoch<=0:
                warmup_scheduler.step()
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            #optimizer.zero_grad()
            output_batch = model(train_batch)
            output_teacher_batch = teacher_model(train_batch)
            if args.regularization:
                loss = loss_fn(output_batch, labels_batch, params)
            else:
                loss = loss_fn(output_batch, labels_batch)
                #loss = nn.KLDivLoss()(F.log_softmax(outputs/6, dim=1), F.softmax(teacher_outputs/6, dim=1)) * 36
                #loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch)
            #val.append(loss.cpu().detach().numpy())
            #loss.backward()
           
            if epoch==0:
               

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            


            else:
                if i not in ind:
                   
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
               

            _, predicted = output_batch.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()
           
            # update the average loss
            loss_avg.update(loss.data)
            losses.update(loss.data, train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()
    
    
    

    acc = 100. * correct / total
    logging.info("- Train accuracy: {acc: .4f}, training loss: {loss: .4f}".format(acc=acc, loss=losses.avg))
    return acc, losses.avg 


def infer(model, optimizer, loss_fn, dataloader,ind, params,teacher_model, epoch, warmup_scheduler, args):
    """
    Noraml training, without KD
    """

    # set model to training mode
    model.eval()
    loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    total = 0
    correct = 0
    val = []
    kl = []

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
           
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            #optimizer.zero_grad()
            output_batch = model(train_batch)
            output_teacher_batch = teacher_model(train_batch)
            if args.regularization:
                loss = loss_fn(output_batch, labels_batch, params)
            else:
                loss = loss_fn(output_batch, labels_batch)
                #loss = nn.KLDivLoss()(F.log_softmax(output_batch, dim=1), F.softmax(output_teacher_batch, dim=1))
            val.append(loss.cpu().detach().numpy())
            #loss.backward()
           
            
    
    
    

    return val









