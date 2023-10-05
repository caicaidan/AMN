# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys, time, os
import numpy as np
import torch
import copy
import utils

from copy import deepcopy
from tqdm import tqdm

sys.path.append('../')

from networks.discriminator import Discriminator

class ACL(object):

    def __init__(self, model, args, network):
        self.args=args
        self.nepochs=args.nepochs
        self.sbatch=args.batch_size

        # optimizer & adaptive lr
        self.e_lr=args.e_lr
        self.d_lr=args.d_lr

        if not args.experiment == 'multidatasets':
            self.e_lr=[args.e_lr] * args.ntasks
            self.d_lr=[args.d_lr] * args.ntasks
        else:
            self.e_lr = [self.args.lrs[i][1] for i in range(len(args.lrs))]
            self.d_lr = [self.args.lrs[i][1]/10. for i in range(len(args.lrs))]
            print ("d_lrs : ", self.d_lr)

        self.lr_min=args.lr_min
        self.lr_factor=args.lr_factor
        self.lr_patience=args.lr_patience

        self.samples=args.samples

        self.device=args.device
        self.checkpoint=args.checkpoint

        self.adv_loss_reg=args.adv
        self.diff_loss_reg=args.orth
        self.s_steps=args.s_step
        self.d_steps=args.d_step

        self.diff=args.diff

        self.network=network
        self.inputsize=args.inputsize
        self.taskcla=args.taskcla
        self.num_tasks=args.ntasks

        # Initialize generator and discriminator
        self.model=model
        self.discriminator=self.get_discriminator(0)
        self.discriminator.get_size() # print the size of the discriminator

        self.latent_dim=args.latent_dim    # Stores the latent dimension value, commonly used in models like autoencoders or GANs

        self.task_loss=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_d=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_s=torch.nn.CrossEntropyLoss().to(self.device)
        self.diff_loss=DiffLoss().to(self.device) # Initialize the diff loss

        self.optimizer_S=self.get_S_optimizer(0)
        self.optimizer_D=self.get_D_optimizer(0)

        self.task_encoded={} # Initializes an empty dictionary to store encoded tasks

        self.mu=0.0 # mean
        self.sigma=1.0 # standard deviation

        print()

# The method aims to load the checkpoint of the model and initialize and return a Discriminator object
    def get_discriminator(self, task_id): # task_id is the current task id
        discriminator=Discriminator(self.args, task_id).to(self.args.device)   # Initialize the discriminator, pass the args and task_id to the discriminator, then the discriminator is moved to the device
        return discriminator # return the discriminator
#
    def get_S_optimizer(self, task_id, e_lr=None):
        if e_lr is None: e_lr=self.e_lr[task_id]    # If e_lr isn't provided, it sets e_lr to the learning rate associated with the given task_id
        optimizer_S=torch.optim.SGD(self.model.parameters(), momentum=self.args.mom,
                                    weight_decay=self.args.e_wd, lr=e_lr) # Initializes a SGD optimizer for the main model's parameters with the specified momentum, weight decay, and learning rate from self.args.
        return optimizer_S

# This method initializes and returns an optimizer for the discriminator.
    def get_D_optimizer(self, task_id, d_lr=None):
        if d_lr is None: d_lr=self.d_lr[task_id]   # If d_lr isn't provided, it sets d_lr to the learning rate associated with the given task_id
        optimizer_D=torch.optim.SGD(self.discriminator.parameters(), weight_decay=self.args.d_wd, lr=d_lr) # Initializes an SGD optimizer for the discriminator's parameters with the specified weight decay and learning rate
        return optimizer_D


# Trains the model and the discriminator on a given dataset for a specific task.
    def train(self, task_id, dataset):
        self.discriminator=self.get_discriminator(task_id)

        best_loss=np.inf # Initializes the best loss to infinity
        best_model=utils.get_model(self.model)


        best_loss_d=np.inf # Initializes the best loss for the discriminator to infinity
        best_model_d=utils.get_model(self.discriminator)

        dis_lr_update=True # A boolean variable to indicate whether the discriminator's learning rate should be updated or not
        d_lr=self.d_lr[task_id] # Sets the discriminator's learning rate to the learning rate associated with the given task_id
        patience_d=self.lr_patience  #耐心值,不是指标停止改变后立马就调整学习率,而是如果在patience个epoch中看不到模型性能提升，则减少学习率
        self.optimizer_D=self.get_D_optimizer(task_id, d_lr)

        e_lr=self.e_lr[task_id] # Sets the model's learning rate to the learning rate associated with the given task_id
        patience=self.lr_patience
        self.optimizer_S=self.get_S_optimizer(task_id, e_lr) # Initializes the optimizer for the model


        for e in range(self.nepochs):

            # Train
            clock0=time.time() # Returns the current time in seconds since the Epoch
            self.train_epoch(dataset['train'], task_id)  # train_epoch method is called to perform one training iteration on the dataset.
            clock1=time.time() # Returns the current time in seconds since the Epoch

            train_res=self.eval_(dataset['train'], task_id) # Evaluates the model and the discriminator on the training dataset for the given task_id

            utils.report_tr(train_res, e, self.sbatch, clock0, clock1) #  print( '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, ' 'Diff loss:{:.3f} |'.format(...


#Adjust Learning Rate Based on Initial Performance
            # If the experiment is either 'cifar100' or 'miniimagenet' and 5 epochs have passed,
            # there's a check to see if the model performs very poorly (almost random chance).
            # If it does, the learning rates for both the discriminator and the main model are decreased. e : epoch
            if (self.args.experiment == 'cifar100' or self.args.experiment == 'miniimagenet') and e == 4:
                random_chance=20.
                threshold=random_chance + 2

                if train_res['acc_t'] < threshold:
                    # Restore best validation model
                    d_lr=self.d_lr[task_id] / 10.
                    self.optimizer_D=self.get_D_optimizer(task_id, d_lr)
                    print("Performance on task {} is {} so Dis's lr is decreased to {}".format(task_id, train_res[
                        'acc_t'], d_lr), end=" ")

                    e_lr=self.e_lr[task_id] / 10.
                    self.optimizer_S=self.get_S_optimizer(task_id, e_lr)

                    self.discriminator=self.get_discriminator(task_id)

                    if task_id > 0:
                        self.model=self.load_checkpoint(task_id - 1)
                    else:
                        self.model=self.network.Net(self.args).to(self.args.device)


            # Valid
            # The performance of the model is evaluated on the validation set using eval_ method
            valid_res=self.eval_(dataset['valid'], task_id)
            utils.report_val(valid_res)


            # Adapt lr for S and D with ""early stopping" with patience
                #After the training loop ends, the model's weights are restored to those from the epoch with the lowest validation loss.

            # Adaptation of Learning Rate for S (the main model)
            if valid_res['loss_tot'] < best_loss:        # If the current validation loss (valid_res['loss_tot']) is lower than the best validation loss seen so far (best_loss), then:
                best_loss=valid_res['loss_tot']          # Update the best validation loss to the current validation loss
                best_model=utils.get_model(self.model)   # Update the best model to the current model
                patience=self.lr_patience               # Reset the patience to the initial value
                print(' *', end='')
            else:
                patience-=1                             # If the current validation loss is not lower than the best validation loss seen so far, then decrease the patience by 1
                if patience <= 0:                       # If the patience is less than or equal to 0, then:
                    e_lr/=self.lr_factor                # Decrease the learning rate by a factor of 10
                    print(' lr={:.1e}'.format(e_lr), end='') # Print the new learning rate
                    if e_lr < self.lr_min:              # If the learning rate is less than the minimum learning rate, then:
                        print()                         # Print a new line
                        break                           # Break out of the loop
                    patience=self.lr_patience           # Reset the patience to the initial value
                    self.optimizer_S=self.get_S_optimizer(task_id, e_lr) # Initialize the optimizer for the model with the new learning rate

            # Adaptation of Learning Rate for D (the discriminator)
            if train_res['loss_a'] < best_loss_d:
                best_loss_d=train_res['loss_a']
                best_model_d=utils.get_model(self.discriminator)
                patience_d=self.lr_patience
            else:
                patience_d-=1
                if patience_d <= 0 and dis_lr_update:
                    d_lr/=self.lr_factor
                    print(' Dis lr={:.1e}'.format(d_lr))
                    if d_lr < self.lr_min:
                        dis_lr_update=False
                        print("Dis lr reached minimum value")
                        print()
                    patience_d=self.lr_patience
                    self.optimizer_D=self.get_D_optimizer(task_id, d_lr)    # Initialize the optimizer for the discriminator with the new learning rate
            print()

        # All models are saved using self.save_all_models(task_id) method
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.discriminator.load_state_dict(copy.deepcopy(best_model_d))

        self.save_all_models(task_id)                                       # print("Saving all models for task {} ...".format(task_id+1))

# 此方法 train_epoch 似乎是对具有共享模块（在代码中称为 S）和鉴别器 (D) 的模型进行单次训练。 训练例程让人想起对抗训练中使用的技术，例如 GAN（生成对抗网络），但适应了基于任务的设置。
    def train_epoch(self, train_loader, task_id):

        self.model.train()
        self.discriminator.train()

        for data, target, tt, td in train_loader:

        # Preprocessing
            x=data.to(device=self.device)                        # x is the input data
            y=target.to(device=self.device, dtype=torch.long)    # y is the target
            tt=tt.to(device=self.device)                        # tt is the task label

            # Detaching samples in the batch which do not belong to the current task before feeding them to P ((它通过分离与当前任务无关的样本来准备输入数据))
            t_current=task_id * torch.ones_like(tt)             # t_current就是一个相同形状的张量，其中每个元素都是当前任务的ID, 对于不属于当前任务的样本， 将它们分离（即将它们排除在梯度计算之外）以确保它们不会对梯度更新做出贡献。
            body_mask=torch.eq(t_current, tt).cpu().numpy()     # 计算当前任务样本的掩码: eq : equal,  计算了一个布尔掩码，其中每个元素表示对应的样本是否属于当前任务。然后将此掩码转换为numpy数组 masked_greater_equal(x,value) 将数组x中大于等于value值的元素设为掩码
                                                                # 如果tt在特定索引上与t_current具有相同的任务ID，那么该索引处的body_mask将为True，否则将为False。

            # x_task_module=data.to(device=self.device)

        # 这段代码首先将输入数据克隆到x_task_module。然后遍历样本并分离那些不属于当前任务的样本（即，body_mask为False或0的地方）。最后，它将生成的张量移动到指定的设备。
        # 分离的目的是确保不为这些样本计算梯度，有效地在反向传播中忽略它们。这意味着即使这些样本通过模型，它们也不影响模型的参数更新。
            x_task_module=data.clone()
            for index in range(x.size(0)):
                if body_mask[index] == 0:
                    x_task_module[index]=x_task_module[index].detach()
            x_task_module=x_task_module.to(device=self.device)

            # Discriminator's real and fake task labels
            # t_real_D只是移动到设备的任务特定标签。使用零创建t_fake_D张量，并且与t_real_D的形状相同。这些标签用于训练鉴别器。
            t_real_D=td.to(self.device)
            t_fake_D=torch.zeros_like(t_real_D).to(self.device)

            # ================================================================== #
            #                        Train Shared Module                          #
            # ================================================================== #
            # training S for s_steps 迭代 s_steps 次： 为了训练共享模块，我们首先将梯度设置为零，然后计算输出并计算任务损失。然后，我们计算鉴别器的输出并计算鉴别器损失。最后，我们计算总损失并执行反向传播。
            for s_step in range(self.s_steps):
                self.optimizer_S.zero_grad()                        # 将梯度设置为零 (在执行反向传播和更新权重之前，重要的是将当前的梯度设置为零。这是PyTorch中的常见做法，以避免从之前的步骤累积梯度。)
                self.model.zero_grad()                              # 将梯度设置为零

                output=self.model(x, x_task_module, tt, task_id)    # 模型前向传播： 将输入 x 和 x_task_module 以及任务信息 tt 和 task_id 提供给模型并获得输出
                task_loss=self.task_loss(output, y)                 # 计算任务损失： 使用交叉熵损失计算任务损失

            # 模型提取编码特征（shared_encoded 和 task_encoded）。然后将共享编码特征传递给判别器，判别器尝试区分真实和生成的（或在这种情况下，特定于任务的）特征。然后使用判别器的输出计算共享模块的对抗性损失。
                shared_encoded, task_encoded=self.model.get_encoded_ftrs(x, x_task_module, task_id)  # 获取编码特征： 使用 get_encoded_ftrs 方法获取编码特征
                dis_out_gen_training=self.discriminator.forward(shared_encoded, t_real_D, task_id)   # 计算鉴别器输出： 使用鉴别器的 forward 方法计算鉴别器输出
                adv_loss=self.adversarial_loss_s(dis_out_gen_training, t_real_D)                    # 计算鉴别器损失： 使用交叉熵损失计算鉴别器损失

            # 计算总损失： 计算总损失并执行反向传播
                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_encoded, task_encoded)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)   # 如果没有使用正交损失，那么将 diff_loss 设置为 0
                    self.diff_loss_reg=0                                                   # 将 diff_loss_reg 设置为 0

                total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss # 计算总损失： 计算总损失并执行反向传播
                total_loss.backward(retain_graph=True)  # 执行反向传播

                self.optimizer_S.step() # 更新参数

            # ================================================================== #
            #                          Train Discriminator                       #
            # ================================================================== #
            # 迭代 d_steps 次来训练判别器。我们首先将梯度设置为零，然后计算输出并计算判别器损失。接着，我们计算总损失并执行反向传播
            for d_step in range(self.d_steps):
                self.optimizer_D.zero_grad()
                self.discriminator.zero_grad()

                # training discriminator on real data
                output=self.model(x, x_task_module, tt, task_id)
                shared_encoded, task_out=self.model.get_encoded_ftrs(x, x_task_module, task_id)
                dis_real_out=self.discriminator.forward(shared_encoded.detach(), t_real_D, task_id)
                dis_real_loss=self.adversarial_loss_d(dis_real_out, t_real_D)
                if self.args.experiment == 'miniimagenet':
                    dis_real_loss*=self.adv_loss_reg
                dis_real_loss.backward(retain_graph=True) # 执行反向传播

                # training discriminator on fake data
                z_fake=torch.as_tensor(np.random.normal(self.mu, self.sigma, (x.size(0), self.latent_dim)),dtype=torch.float32, device=self.device)
                dis_fake_out=self.discriminator.forward(z_fake, t_real_D, task_id)
                dis_fake_loss=self.adversarial_loss_d(dis_fake_out, t_fake_D)
                if self.args.experiment == 'miniimagenet':
                    dis_fake_loss*=self.adv_loss_reg
                dis_fake_loss.backward(retain_graph=True)   # 执行反向传播

                self.optimizer_D.step() # 更新参数

        return


    def eval_(self, data_loader, task_id):
        # 初始化各种损失和正确分类的数量
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t = 0, 0
        num=0
        batch=0

        # 将模型和鉴别器设置为评估模式
        self.model.eval()
        self.discriminator.eval()

        res={}
        # 确保在没有梯度的情况下，对数据集进行迭代
        with torch.no_grad():
            for batch, (data, target, tt, td) in enumerate(data_loader):
                # Preprocessing （将数据迁移到设备）
                x=data.to(device=self.device)
                y=target.to(device=self.device, dtype=torch.long)
                tt=tt.to(device=self.device)
                t_real_D=td.to(self.device)

                # Forward
                output=self.model(x, x, tt, task_id)
                shared_out, task_out=self.model.get_encoded_ftrs(x, x, task_id)
                _, pred=output.max(1)
                # 计算正确分类的数量
                correct_t+=pred.eq(y.view_as(pred)).sum().item()

                # Discriminator's performance:
                output_d=self.discriminator.forward(shared_out, t_real_D, task_id)
                _, pred_d=output_d.max(1)
                # 计算正确分类的数量
                correct_d+=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                # Loss values
                task_loss=self.task_loss(output, y)
                adv_loss=self.adversarial_loss_d(output_d, t_real_D)

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_out, task_out)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0

                # Loss values
                total_loss = task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss

                loss_t+=task_loss
                loss_a+=adv_loss
                loss_d+=diff_loss
                loss_total+=total_loss

                # 更新已处理的样本数
                num+=x.size(0)

        # 计算最终的平均损失和准确率并返回
        res['loss_t'], res['acc_t']=loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (batch + 1)
        res['loss_tot']=loss_total.item() / (batch + 1)
        res['size']=self.loader_size(data_loader)

        return res

    #


# 这个函数是用来测试给定模型在某个数据集(data_loader)上的性能。与评估函数相似，但重点是使用外部传入的模型进行测试，而不是内部已训练的模型。

    def test(self, data_loader, task_id, model):
        # 初始化各种损失和正确分类的数量
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t=0, 0
        num=0
        batch=0

        # 将模型和鉴别器设置为评估模式
        model.eval()
        self.discriminator.eval()

        res={}

        with torch.no_grad():
            # 确保在没有梯度的情况下，对数据集进行迭代
            for batch, (data, target, tt, td) in enumerate(data_loader):
                x=data.to(device=self.device)
                y=target.to(device=self.device, dtype=torch.long)
                tt=tt.to(device=self.device)
                t_real_D=td.to(self.device)

                # Forward
                output=model.forward(x, x, tt, task_id)
                shared_out, task_out=model.get_encoded_ftrs(x, x, task_id)

                _, pred=output.max(1)
                correct_t+=pred.eq(y.view_as(pred)).sum().item()

                # Discriminator's performance:
                output_d=self.discriminator.forward(shared_out, tt, task_id)
                _, pred_d=output_d.max(1)
                correct_d+=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_out, task_out)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0 # 将 diff_loss_reg 设置为 0

                # Loss values
                adv_loss=self.adversarial_loss_d(output_d, t_real_D)
                task_loss=self.task_loss(output, y)

                total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss # 计算总损失

                loss_t+=task_loss # 计算任务损失
                loss_a+=adv_loss # 计算鉴别器损失
                loss_d+=diff_loss # 计算正交损失
                loss_total+=total_loss # 计算总损失

                num+=x.size(0) # 更新已处理的样本数

        res['loss_t'], res['acc_t']=loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (batch + 1)
        res['loss_tot']=loss_total.item() / (batch + 1)
        res['size']=self.loader_size(data_loader)

        return res



    def save_all_models(self, task_id):
        print("Saving all models for task {} ...".format(task_id+1))
        dis=utils.get_model(self.discriminator)
        torch.save({'model_state_dict': dis,
                    }, os.path.join(self.checkpoint, 'discriminator_{}.pth.tar'.format(task_id)))

        model=utils.get_model(self.model)
        torch.save({'model_state_dict': model,
                    }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))



    def load_model(self, task_id):

        # Load a previous model
        net=self.network.Net(self.args)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])

        # # Change the previous shared module with the current one
        current_shared_module=deepcopy(self.model.shared.state_dict())
        net.shared.load_state_dict(current_shared_module)

        net=net.to(self.args.device)
        return net


    def load_checkpoint(self, task_id):
        print("Loading checkpoint for task {} ...".format(task_id))

        # Load a previous model
        net=self.network.Net(self.args)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])
        net=net.to(self.args.device)
        return net


    def loader_size(self, data_loader):
        return data_loader.dataset.__len__()



# 主要目的是从模型中提取前10个任务的特征嵌入（使用t-SNE降维），并将它们存储在tensorboardX的日志中以进行可视化
    def get_tsne_embeddings_first_ten_tasks(self, dataset, model):
        from tensorboardX import SummaryWriter

        model.eval() # 将模型设置为评估模式

        tag_ = '_diff_{}'.format(self.args.diff)
        all_images, all_shared, all_private = [], [], [] # 初始化列表以存储特征嵌入和图像

        # Test final model on first 10 tasks:
        writer = SummaryWriter() # 初始化一个tensorboardX的SummaryWriter对象
        for t in range(10): # 遍历前10个任务
            for itr, (data, _, tt, td) in enumerate(dataset[t]['tsne']): # 遍历数据集
                x = data.to(device=self.device) # 将数据迁移到设备
                tt = tt.to(device=self.device) # 将任务标签迁移到设备
                output = model.forward(x, x, tt, t) # 模型前向传播
                shared_out, private_out = model.get_encoded_ftrs(x, x, t) # 获取编码特征
                all_shared.append(shared_out) # 将共享编码特征添加到列表中
                all_private.append(private_out) # 将私有编码特征添加到列表中
                all_images.append(x) # 将图像添加到列表中

        print (torch.stack(all_shared).size()) # # 打印共享特征嵌入的形状 torch.Size([10, 10, 64])

        # 将特征嵌入和图像添加到tensorboardX的日志中
        tag = ['Shared10_{}_{}'.format(tag_,i) for i in range(1,11)]
        writer.add_embedding(mat=torch.stack(all_shared,dim=1).data, label_img=torch.stack(all_images,dim=1).data, metadata=list(range(1,11)),
                             tag=tag)#, metadata_header=list(range(1,11)))
        # 将特征嵌入和图像添加到tensorboardX的日志中
        tag = ['Private10_{}_{}'.format(tag_, i) for i in range(1, 11)]
        writer.add_embedding(mat=torch.stack(all_private,dim=1).data, label_img=torch.stack(all_images,dim=1).data, metadata=list(range(1,11)),
                         tag=tag)#,metadata_header=list(range(1,11)))
        writer.close() # 关闭SummaryWriter对象


    def get_tsne_embeddings_last_three_tasks(self, dataset, model):
        from tensorboardX import SummaryWriter

        # Test final model on last 3 tasks:
        model.eval()
        tag = '_diff_{}'.format(self.args.diff)

        for t in [17,18,19]: # 遍历最后3个任务
            all_images, all_labels, all_shared, all_private = [], [], [], []
            writer = SummaryWriter()
            for itr, (data, target, tt, td) in enumerate(dataset[t]['tsne']):
                x = data.to(device=self.device)
                y = target.to(device=self.device, dtype=torch.long)
                tt = tt.to(device=self.device)
                output = model.forward(x, x, tt, t)
                shared_out, private_out = model.get_encoded_ftrs(x, x, t) # 获取编码特征
                # print (shared_out.size())

                all_shared.append(shared_out) # 将共享编码特征添加到列表中
                all_private.append(private_out) # 将私有编码特征添加到列表中
                all_images.append(x) # 将图像添加到列表中
                all_labels.append(y)    # 将标签添加到列表中

            writer.add_embedding(mat=torch.stack(all_shared,dim=1).data, label_img=torch.stack(all_images,dim=1).data,
                                 metadata=list(range(1,6)), tag='Shared_{}_{}'.format(t, tag))
                                 # ,metadata_header=list(range(1,6))) # 将特征嵌入和图像添加到tensorboardX的日志中
            writer.add_embedding(mat=torch.stack(all_private,dim=1).data, label_img=torch.stack(all_images,dim=1).data,
                                 metadata=list(range(1,6)), tag='Private_{}_{}'.format(t, tag))
                                 # ,metadata_header=list(range(1,6))) # 将特征嵌入和图像添加到tensorboardX的日志中

        writer.close()



        #
class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        # 将D1重塑为二维张量 (batch_size, -1)
        D1=D1.view(D1.size(0), -1)
        # 计算D1的L2范数
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        # 使用L2范数对D1进行归一化
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)
        # 将D2重塑为二维张量 (batch_size, -1)
        D2=D2.view(D2.size(0), -1)
        # 计算D2的L2范数
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        # 使用L2范数对D2进行归一化
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        # 这里计算的是两个归一化的矩阵之间的内积，并平方，然后求均值，这个值越小越好，因为这个值越小，说明两个矩阵越正交，也就是说两个矩阵之间的差异越大，这也就是为什么这个损失函数叫做正交损失的原因。
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2))) # 计算正交损失
