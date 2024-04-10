import json
import math
import re
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties

import utils


class Statistic:
    def __init__(self):
        self.map = {}
        self.first = True

    def update(self, name, value):
        ctx = self.map.get(name)
        if ctx is None:
            ctx = [[]]
            self.map[name] = ctx
        ctx[-1].append(value)

    def update_values(self, name, value):
        ctx = self.map.get(name)
        if ctx is None:
            ctx = [[]]
            self.map[name] = ctx
        ctx[-1] += value

    def update_iter(self):
        for (k, v) in self.map.items():
            if len(v[-1]) > 1 and not re.search("list\\.\\d+", k):
                utils.vis.histogram(X=v[-1], win=f"{k}.hist", opts=dict(
                    title=f"{k}", xlabel="Iter Time/us", ylabel="Histogram"))

    def update_epoch(self):
        if self.first:
            self.first = False
            ctx = self.map.get("train.transformer")
            if ctx is not None:
                max = np.max(ctx[-1])
                mean = np.mean(ctx[-1])
                ctx[-1] = [i if i != max else mean for i in ctx[-1]]
        for (k, v) in self.map.items():
            if len(v[-1]) > 0:
                v.append([])
            if not re.search("list\\.\\d+", k):
                v = v[:-1]
                xticksval = [i for i in range(len(v))]
                legends = [f"Epoch {i}" for i in range(len(v))]
                utils.vis.boxplot(np.transpose(v, (1, 0)), win=f"{k}.box", opts=dict(showfliers=False,
                                                                                     legend=legends,
                                                                                     title=f"{k}", xlabel="Epoch",
                                                                                     ylabel="Time/us per Test",
                                                                                     xtickvals=xticksval,
                                                                                     xticklabels=xticksval))

    def save(self, file_name):
        with open(f'statistics/{file_name}.json', 'w') as f:
            json.dump(self.map, f)

    def clear(self):
        self.map.clear()
        self.first = True


root_static = Statistic()


def forward_timer(func, *arg_names):
    # global vis, env_name
    @wraps(func)  # 保持原函数名不变
    def wrapper(self, *args, **kwargs):
        start = time.time_ns()
        res = func(self, *args, **kwargs)
        gap = (time.time_ns() - start) / 1000  # to microseconds

        for name in arg_names:
            func_name = f"{utils.current_stage}.{name}"
            root_static.update(func_name, gap / utils.batch)
        return res

    return wrapper


def result_visualization(loss_list: list,
                         correct_on_test: list,
                         correct_on_train: list,
                         test_interval: int,
                         d_model: int,
                         q: int,
                         v: int,
                         h: int,
                         N: int,
                         dropout: float,
                         DATA_LEN: int,
                         BATCH_SIZE: int,
                         time_cost: float,
                         EPOCH: int,
                         draw_key: int,
                         reslut_figure_path: str,
                         optimizer_name: str,
                         file_name: str,
                         LR: float,
                         pe: bool,
                         mask: bool):
    my_font = fp(fname=r"font/simsun.ttc")  # 2、设置字体路径

    # 设置风格
    plt.style.use('seaborn')

    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(311)  # 创建两个子图
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # 添加折线
    ax2.plot(correct_on_test, color='red', label='on Test Dataset')
    ax2.plot(correct_on_train, color='blue', label='on Train Dataset')

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')

    # 设置文本
    fig.text(x=0.13, y=0.4, s=f'最小loss：{min(loss_list)}' '    '
                              f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}' '    '
                              f'最后一轮loss:{loss_list[-1]}' '\n'
                              f'最大correct：测试集:{max(correct_on_test)}% 训练集:{max(correct_on_train)}%' '    '
                              f'最大correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}' '    '
                              f'最后一轮correct：{correct_on_test[-1]}%' '\n'
                              f'd_model={d_model}   q={q}   v={v}   h={h}   N={N}  drop_out={dropout}'  '\n'
                              f'共耗时{round(time_cost, 2)}分钟', FontProperties=my_font)

    # 保存结果图   测试不保存图（epoch少于draw_key）
    if EPOCH >= draw_key:
        plt.savefig(
            f'{reslut_figure_path}/{file_name} {max(correct_on_test)}% {optimizer_name} epoch={EPOCH} batch={BATCH_SIZE} lr={LR} pe={pe} mask={mask} [{d_model},{q},{v},{h},{N},{dropout}].png')

    # 展示图
    plt.show()

    print('正确率列表', correct_on_test)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：测试集:{max(correct_on_test)}\t 训练集:{max(correct_on_train)}\r\n'
          f'最correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_on_test[-1]}')

    print(f'共耗时{round(time_cost, 2)}分钟')
