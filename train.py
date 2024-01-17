
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
from transform import transforms as datasforms
from model import MCDC_Net

def text_create(name):
    desktop_path = "./output/"
    # 新创建的txt文件的存放路径
    full_path = desktop_path + name+'.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse.parse_args()
    train_path = args.train_path
    val_path = args.val_path
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_path = args.model_path

    # creat train and val dataloader
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=datasforms['train'])
    val_dataset = torchvision.datasets.ImageFolder(val_path, transform=datasforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                               num_workers=0)  # num_workers载入数据的线程数
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                             num_workers=0)
    train_dataset_size = len(train_dataset)
    print("train-dataset-size:",train_dataset_size)
    val_dataset_size = len(val_dataset)
    print("val-dataset-size",val_dataset_size)

    # Creat the networks
    model = MCDC_Net()
    if continue_train:
        model.load_state_dict(torch.load(model_path))  # 载入权重
    model = model.cuda()  # 使用gpu
    criterion = nn.CrossEntropyLoss()  # 损失函数
    # optimizer: 神经⽹络训练中使⽤的优化器，如optimizer=torch.optim.SGD(…)
    # gamma(float): 学习率调整倍数，默认为0.1
    # last_epoch(int): 上⼀个epoch数，这个变量⽤来指⽰学习率是否需要调整。当last_epoch符合设定的间隔时，就会对学习率进⾏调整；当
    # 为-1时，学习率设置为初始值
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) #设定优化器更新时刻表

    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    for epoch in range(epoches):  # 将训练集迭代的轮数：epoches
        scheduler.step()  #修改学习率，在每轮epoch之前更新学习率

        print('Epoch {}/{}'.format(epoch + 1, epoches))
        print('-' * 10)
        model = model.train()  # 在训练过程启用drop方法，预测时并不使用
        train_loss = 0.0  # 累加在训练过程中的损失
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        for (image, labels) in train_loader:  # 遍历训练集样本
            image = image.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()  # 将历史损失梯度清零
            outputs = model(image)  # 将图片传入网络模型进行正向传播得到输出
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)  # 计算损失：output：预测值，labels：标签值
            loss.backward()  # 将loss进行反向传播
            optimizer.step()  # 参数更新
            iter_loss = loss.data.item()
            train_loss += iter_loss  # 累加损失
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % 100):
                print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size,
                                                                           iter_corrects / batch_size))
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model.eval()  # 在验证或者预测过程关掉drop方法

        with torch.no_grad():  # 在接下来的计算不要区计算每个节点的误差损失梯度
            for (image, labels) in val_loader:
                image = image.cuda()
                labels = labels.cuda()
                outputs = model(image)  # 进行正向传播
                aa, preds = torch.max(outputs.data, 1)  # 输出预测值
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)  # 预测正确的数目
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            if epoch_acc > best_acc:  # 如果当前准确率大于历史最优准确率
                best_acc = epoch_acc
                best_model_wts = model.state_dict()  # 保存当前的权重
        scheduler.step()
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join("best.pkl"))  # 保存训练参
    # outputfile.close()




if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='xception_pdc_Net')
    parse.add_argument('--train_path', '-tp', type=str, default=r'/media/ling/软件安装/1229/hdf/c23/deepfakes/train')
    parse.add_argument('--val_path', '-vp', type=str, default=r'/media/ling/软件安装/1229/hdf/c23/deepfakes/val')
    parse.add_argument('--batch_size', '-bz', type=int, default=6)
    parse.add_argument('--epoches', '-e', type=int, default='60')
    parse.add_argument('--continue_train', type=bool, default=False)
    main()





