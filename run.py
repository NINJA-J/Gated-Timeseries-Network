# @Time    : 2021/01/22 25:16
# @Author  : SY.M
# @FileName: run.py
import itertools
import os
import re

import torch
import torch.optim as optim
import visdom
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset_process.dataset_process import MyDataset
from module.encoder import EncoderList
from module.feedForward import FeedForward
from module.loss import MyLoss
from module.multiHeadAttention import MultiHeadAttention
from module.transformer import Transformer
from utils.visualization import root_static, forward_timer

# 超参数设置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(f'use device: {DEVICE}', torch.cuda.get_device_properties(DEVICE))
utils.device = DEVICE


# 测试函数
def test(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()

        return round((100 * correct / total), 2)


# 训练函数
def train(path, BATCH_SIZE, d_model, EPOCH=10, LR=1e-4):
    global net, optimizer  # , correct_on_test, correct_on_train
    utils.batch = BATCH_SIZE
    print(f"Train on Path={path}, Batch Size={BATCH_SIZE}, d_model={d_model}")

    test_interval = 2  # 测试间隔 单位：epoch

    d_hidden = d_model * 2
    q = 8
    v = 8
    h = 8
    N = 8
    dropout = 0.2
    pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
    mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
    # 优化器选择
    optimizer_name = 'Adagrad'

    file_name, _ = os.path.splitext(os.path.basename(path))  # 获得文件名字
    utils.env_name = f"{file_name} dModel={d_model} batch={BATCH_SIZE} epoch={EPOCH}"
    utils.vis = visdom.Visdom(env=utils.env_name, port=6543)

    train_dataset = MyDataset(path, 'train')
    test_dataset = MyDataset(path, 'test')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    DATA_LEN = train_dataset.train_len  # 训练集样本数量
    d_input = train_dataset.input_len  # 时间部数量
    d_channel = train_dataset.channel_len  # 时间序列维度
    d_output = train_dataset.output_len  # 分类类别

    # 维度展示
    print(f'data structure: [lines, timesteps, features], '
          f'train: [{DATA_LEN}, {d_input}, {d_channel}], '
          f'test: [{train_dataset.test_len}, {d_input}, {d_channel}]')
    print(f'Number of classes: {d_output}')

    # 创建Transformer模型
    net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                      q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask).to(DEVICE)
    utils.vis.text(
        f"<span style='color: red; font-size: 48px;'>Running</span><br><pre>{net}</pre>".replace('\n', '<br>'),
        win="net-info", opts=dict(title="Network Inforation"))
    # print(net)
    for name, module in net.named_modules():
        if isinstance(module, (MultiHeadAttention, FeedForward)):
            module.forward = forward_timer(module.forward, re.sub("list\\.\\d+", "list", name), name)
        if isinstance(module, EncoderList):
            module.forward = forward_timer(module.forward, name)
        if isinstance(module, Transformer):
            module.forward = forward_timer(module.forward, "transformer")

    # 创建loss函数 此处使用 交叉熵损失
    loss_function = MyLoss()
    if optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=LR)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=LR)

    # 用于记录准确率变化
    max_train_acc = 0
    max_test_acc = 0
    # 用于记录损失变化

    detail_loss = CrossEntropyLoss(reduction='none')

    net.train()
    pbar = tqdm(total=EPOCH)
    for index in range(EPOCH):
        utils.current_stage, utils.current_epoch = "train", index
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            utils.current_i = i
            y_pre = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))
            root_static.update_values("train.loss",
                                      detail_loss(y_pre, y.to(DEVICE).long()).tolist())

            if i % 50 == 0:
                print(f'Epoch:{index}/{i}:\t\tloss:{loss.item()}')

            loss.backward()

            optimizer.step()
            if i % 50 == 0:
                root_static.update_iter()

        root_static.update_iter()

        if ((index + 1) % test_interval) == 0:
            utils.current_stage = "test"
            train_acc, test_acc = test(net, train_dataloader), test(net, test_dataloader)
            max_train_acc, max_test_acc = max(max_train_acc, train_acc), max(max_test_acc, test_acc)

            print(f"当前准确率/最大准确率:\t训练集:{train_acc}/{max_train_acc}%\t测试集:{test_acc}/{max_test_acc}%")

        pbar.update()
        root_static.update_epoch()

    root_static.save(utils.env_name)
    root_static.clear()

    utils.vis.text(
        f"<span style='color: green; font-size: 48px;'>Finished</span><br><pre>{net}</pre>".replace('\n', '<br>'),
        win="net-info", opts=dict(title="Network Information"))

    del net
    torch.cuda.empty_cache()


if __name__ == '__main__':
    tasks = [
        # {
        #     'path': ['D:\\Data\\UWave\\UWave.mat'],
        #     'batch': [16, 32],
        #     'd_model': [128],
        # },
        {
            'path': ['D:\\Data\\UWave\\UWave.mat'],
            'batch': [16, 32, 64],
            'd_model': [128, 256],
        },
        {
            'path': ['D:\\Data\\ArabicDigits\\ArabicDigits.mat'],
            'batch': [16, 32, 64],
            'd_model': [128, 256],
        },
        {
            'path': ['D:\\Data\\CharacterTrajectories\\CharacterTrajectories.mat'],
            'batch': [16, 32, 64],
            'd_model': [128, 256],
        },
    ]
    for t in tasks:
        for param in itertools.product(t.get('path'), t.get('batch', [16]), t.get('d_model', [128])):
            try:
                train(*param)
            except Exception as e:
                print(e)
