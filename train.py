import torch
import os
import numpy as np
from core.config import config
from core.build_dataset import build_dataset
from core.build_network import build_network, build_loss_network
from core.build_optimizer import build_optimizer
from core.build_summary_op import build_summary_op
from core.disp_loss import l1_loss, l2_loss, huber_loss, log_loss
from core.disp_loss import compute_metrics
from core.adjust_lr import adjust_lr
from core.sobel import Sobel
from torch.backends import cudnn
import random
cudnn.benchmark = True

np.random.seed(128)
random.seed(128)
torch.cuda.manual_seed_all(128)
torch.manual_seed(128)


def _display_process(img, rgb=False, gt=None):
    img = img.detach().cpu().numpy()
    img = img[0]
    if rgb:
        img += config.network.pixel_mean.reshape((-1, 1, 1))
        img = img[-1::-1].astype(np.uint8)
    else:
        if gt is None:
            gt = img
        else:
            gt = gt[0].detach().cpu().numpy()
        valid_mask = np.float32(gt>=0)
        img = img * valid_mask
        img_max = (img * valid_mask).max()
        img_min = (img + (1-valid_mask)*255).min()


        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img

        

def train():
    train_loader = build_dataset('train') 
    val_loader = build_dataset('val')
    net = build_network()
    loss_net = build_loss_network(config.train.perc_snapshot)
    optimizer = build_optimizer(net)
    train_summary_op, val_summary_op = build_summary_op()

    begin_epoch = 0
    global_step = 0
    best_p1 = None
    best_epoch = 0
    # resmue model
    if config.train.resume:
        ckpt = torch.load(config.train.snapshot, map_location = config.gpu[0])
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        begin_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        if ckpt.has_key('p1'):
            best_p1 = ckpt['p1']
        best_epoch = ckpt['epoch']
    elif config.train.pretrained:
        ckpt = torch.load(config.train.snapshot, map_location = config.gpu[0])
        net.load_state_dict(ckpt['model_state_dict'])
    elif config.train.pretrained_backbone is not None:
        state_dict = net.module.res_backbone.modify_state_dict_keys(torch.load(config.train.pretrained_backbone, map_location = config.gpu[0]))
        net.module.res_backbone.load_state_dict(state_dict, strict=False)
    get_gradient = Sobel().to(config.gpu[0]) 
    for epoch in range(begin_epoch, config.train.max_epoch):
        np.random.seed()
        net.module.set_stage('train')
        for batch_id, batch in enumerate(train_loader):
            if global_step in config.train.lr_decay_iterations:
                ckpt = torch.load(os.path.join(config.train.output_path, 'epoch-%d.pth'%(best_epoch)), map_location = config.gpu[0])
                net.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            lr = adjust_lr(config.train.lr, global_step, config.train.lr_decay_iterations, optimizer)
            optimizer.zero_grad()
            # train loop
            image = batch['data'].pin_memory().to(config.gpu[0])
            gt = batch['gt'].pin_memory().to(config.gpu[0])
            
            prediction = net(image)
            loss_l1 = l1_loss(gt, prediction)
            feat3, feat = loss_net(torch.cat([gt, prediction], 0))
            feat1, feat2 = feat
            batch_size = gt.shape[0]
            gt_feat3 = feat3[:batch_size]
            gt_feat2 = feat2[:batch_size]
            gt_feat1 = feat1[:batch_size]
            pre_feat3 = feat3[batch_size:]
            pre_feat2 = feat2[batch_size:]
            pre_feat1 = feat1[batch_size:]
            loss1 = l1_loss(gt_feat1, pre_feat1, False) * 10
            loss2 = l1_loss(gt_feat2, pre_feat2, False) * 15
            loss3 = l1_loss(gt_feat3, pre_feat3, False)
            if global_step < config.train.perc_loss_warmup:
                perc_weight = (1.0 * global_step / config.train.perc_loss_warmup)
            else: 
                perc_weight = 1.0
            loss = loss_l1 + perc_weight * (loss1 + loss2)

            prediction_g = get_gradient(prediction)
            gt_g = get_gradient(gt)
            grad_loss = 2*l1_loss(gt_g, prediction_g, mask=torch.cat([gt, gt], 1))
            loss = loss + grad_loss
            loss.backward()
            optimizer.step(lr)
            
            if global_step % config.train.display_iter == 0 and global_step != 0:
                print('Epoch: %d/%d, Iteration: %d, Batch ID: %d/%d, Loss: %f'%
                      (epoch, config.train.max_epoch, global_step, batch_id, len(train_loader), loss.item()))
            if global_step % config.train.summary_iter == 0 and global_step != 0:
                train_summary_op.add_image('image', _display_process(image,rgb=True), global_step=global_step)
                train_summary_op.add_image('gt', _display_process(gt), global_step=global_step)
                train_summary_op.add_image('pre', _display_process(prediction, gt=gt), global_step=global_step)
                train_summary_op.add_scalar('l1_loss', loss_l1.item(), global_step=global_step)
                train_summary_op.add_scalar('grad_loss', grad_loss.item(), global_step=global_step)
                train_summary_op.add_scalar('perc_loss1', loss1.item(), global_step=global_step)
                train_summary_op.add_scalar('perc_loss2', loss2.item(), global_step=global_step)
                train_summary_op.add_scalar('perc_loss3', loss3.item(), global_step=global_step)
                train_summary_op.add_scalar('lr', lr, global_step=global_step)

            global_step += 1


        rms, rel, rms_log10, p1, p2, p3 = 0, 0, 0, 0, 0, 0

        net.module.set_stage('eval')
        for batch in val_loader:
            image = batch['data'].pin_memory().to(config.gpu[0])
            gt = batch['gt'].pin_memory().to(config.gpu[0])
            with torch.no_grad():
                prediction = net(image)
                #prediction = torch.nn.functional.interpolate(prediction, gt.shape[2:], mode='bilinear')
            metrics = compute_metrics(gt, prediction, [1.25, 1.25**2, 1.25**3])
            rms += metrics[0]
            rel += metrics[1]
            rms_log10 += metrics[2]
            p1 += metrics[3][0]
            p2 += metrics[3][1]
            p3 += metrics[3][2]
        rms = (rms / len(val_loader))
        rel /= len(val_loader)
        rms_log10 /= len(val_loader)
        p1 /= len(val_loader)
        p2 /= len(val_loader)
        p3 /= len(val_loader)
        val_summary_op.add_image('image', _display_process(image,rgb=True), global_step=global_step)
        val_summary_op.add_image('gt', _display_process(gt), global_step=global_step)
        val_summary_op.add_image('pre', _display_process(prediction, gt=gt), global_step=global_step)
        val_summary_op.add_scalar('rms', rms, global_step=global_step)
        val_summary_op.add_scalar('rel', rel, global_step=global_step)
        val_summary_op.add_scalar('rms_log10', rms_log10, global_step=global_step)
        val_summary_op.add_scalar('p1', p1, global_step=global_step)
        val_summary_op.add_scalar('p2', p2, global_step=global_step)
        val_summary_op.add_scalar('p3', p3, global_step=global_step)


        print("Epoch: %d, Val rms: %f, rel: %f, rms_log10: %f, \
              p1: %f, p2: %f, p3: %f"%(epoch, rms, rel, rms_log10, p1, p2, p3))

        if best_p1 is None or best_p1 <= p1:
            best_p1 = p1
            best_epoch = epoch

        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'p1': p1,
            'config': config
        }, os.path.join(config.train.output_path, 'epoch-%d.pth'%epoch))
    

if __name__ == '__main__':
    train()
