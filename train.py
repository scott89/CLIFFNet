import torch
import os
import numpy as np
from core.config import config
from core.build_dataset import build_dataset
from core.build_network import build_network
from core.build_optimizer import build_optimizer
from core.build_summary_op import build_summary_op
from core.disp_loss import l1_loss
from core.adjust_lr import adjust_lr

def _display_process(img, rgb=False, gt=None):
    img = img.detach().cpu().numpy()
    img = img[0]
    if rgb:
        img += config.train.augment.pixel_mean.reshape((-1, 1, 1))
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
    optimizer = build_optimizer(net)
    train_summary_op, val_summary_op = build_summary_op()

    begin_epoch = 0
    global_step = 0
    best_loss = None
    best_epoch = 0
    # resmue model
    if config.train.resume:
        ckpt = torch.load(config.train.snapshot, map_location = config.gpu[0])
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        begin_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step'] + 1
        best_loss = ckpt['loss']
        best_epoch = ckpt['epoch']
    elif config.train.pretrained:
        ckpt = torch.load(config.train.snapshot, map_location = config.gpu[0])
        net.load_state_dict(ckpt['model_state_dict'])
    elif config.train.pretrained_backbone is not None:
        state_dict = net.module.res_backbone.modify_state_dict_keys(torch.load(config.train.pretrained_backbone, map_location = config.gpu[0]))
        net.module.res_backbone.load_state_dict(state_dict, strict=False)
    best_epoch = begin_epoch
    
    net.eval()
    for epoch in range(begin_epoch, config.train.max_epoch):
        net, optimizer = adjust_lr(epoch, net, optimizer, best_epoch)
        for batch_id, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # train loop
            image = batch['image'].pin_memory().to(config.gpu[0])
            gt = batch['gt'].pin_memory().to(config.gpu[0])
            prediction = net(image)
            loss = l1_loss(gt, prediction)
            loss.backward()
            optimizer.step()
            
            if global_step % config.train.display_iter == 0 and batch_id != 0:
                print('Epoch: %d/%d, Batch ID: %d/%d, Loss: %f'%
                      (epoch, config.train.max_epoch, batch_id, len(train_loader), loss.item()))
            if global_step % config.train.summary_iter == 0 and batch_id != 0:
                train_summary_op.add_image('image', _display_process(image,rgb=True), global_step=global_step)
                train_summary_op.add_image('gt', _display_process(gt), global_step=global_step)
                train_summary_op.add_image('pre', _display_process(prediction, gt=gt), global_step=global_step)
                train_summary_op.add_scalar('l1_loss', loss.item(), global_step=global_step)

            global_step += 1


        loss = 0
        for batch in val_loader:
            image = batch['image'].pin_memory().to(config.gpu[0])
            gt = batch['gt'].pin_memory().to(config.gpu[0])
            with torch.no_grad():
                prediction = net(image)
            cur_loss = l1_loss(gt, prediction)
            loss += cur_loss.item()
        loss /= len(val_loader)
        val_summary_op.add_image('image', _display_process(image,rgb=True), global_step=global_step)
        val_summary_op.add_image('gt', _display_process(gt), global_step=global_step)
        val_summary_op.add_image('pre', _display_process(prediction, gt=gt), global_step=global_step)
        val_summary_op.add_scalar('l1_loss', loss, global_step=global_step)
        print("Epoch: %d, Val Loss: %f Best Loss: %s"%(epoch, loss, str(best_loss)))

        if best_loss is None or loss <= best_loss:
            best_loss = loss
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, os.path.join(config.train.output_path, 'epoch-%d.pth'%epoch))
    

if __name__ == '__main__':
    train()
