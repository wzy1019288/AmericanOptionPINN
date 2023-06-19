
import os
import sys
import time
import torch
import visdom
import shutil
import subprocess
import numpy as np
import pandas as pd




def mkdir_p(path):
    '''make dir if not exist'''
    os.makedirs(path, exist_ok=True)

def run_cmd(cmd_string, shell=True):
    """
    执行 cmd 命令，并得到执行后的返回值，Python 调试界面不输出返回值
    :param cmd_string: cmd 命令，如：'adb devices"'
    :return:
    """
    print('运行 cmd 指令: {}'.format(cmd_string))
    process = subprocess.Popen(cmd_string, shell=shell)
    return process

##############################################################################################################
## SAVE
def save_csv(df, csv_file_path):
    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, index=False)
    else: # else it exists so append without writing the header
        df.to_csv(csv_file_path, mode='a', header=False, index=False)

'''Save best model to checkpoint'''
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

'''Save learning curves during training and testing'''
def save_learncurve(state, curve='curve', filename='curve.pt'):
    filepath = os.path.join(curve, filename)
    torch.save(state, filepath)  
         
class Logger(object):
    '''Save training process to log file'''
    def __init__(self, fpath, title=None): 
        self.file = None
        self.title = '' if title == None else title
        if fpath is not None:
        	self.file = open(fpath, 'w')

    def set_names(self, names):
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.8f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()  

class VisdomLogger(object):

    def __init__(self, env='main', env_path='/') -> None:
        self.env = env
        self.env_path = env_path
        os.makedirs(self.env_path, exist_ok=True)
        self.cmd_process = run_cmd('start cmd /k visdom -env_path {}'.format(env_path), shell=True)
        self.vis = visdom.Visdom(env=env)
    
    def record_lines(self,
                     Y: list=[0.],
                     X: list=[0.],
                     legend: list=['line1'],
                     panel_name: str='panel',
                     title: str='title',
                     append: bool=False
                     ):
        self.vis.line([Y], X, 
                      win=panel_name,
                      env=self.env, 
                      opts=dict(title=title, legend=legend),
                      update='append' if append else None
                      )
        time.sleep(0.1)
    
    def record_texts(self, 
                     msg,
                     window_text = None,
                     append: bool=False
                     ):
        window_text = self.vis.text(msg, win=window_text, append=append)
        return window_text

    def save(self):
        ''''''
        self.vis.save([self.env])
    
    def close(self):
        
        def find_cmd_process():
            """
            查找与 cmd 窗口相关的进程并返回其 PID
            """
            output = subprocess.check_output('tasklist', shell=True, encoding='gbk')
            lines = output.strip().split('\n')
            for line in lines:
                if 'visdom' in line:
                    process_info = line.split()
                    pid = int(process_info[1])
                    return pid
            return None
        cmd_pid = find_cmd_process()

        if cmd_pid:
            subprocess.Popen('taskkill /F /T /PID {}'.format(cmd_pid), shell=True)

##############################################################################################################




##############################################################################################################
## MODEL
def print_grad(sol_model, fb_model, epoch=0, if_print=False):
    _sol_tmp_mean = {'epoch': epoch}
    _sol_tmp_grad_mean = {'epoch': epoch}
    for name, param in sol_model.named_parameters():
        if param.requires_grad:
            _sol_tmp_mean.update({name: torch.mean(param).item()})
            _sol_tmp_grad_mean.update({name: torch.mean(param.grad).item()})
    sol_mean = np.mean([v for k, v in _sol_tmp_mean.items() if k != 'epoch'])
    _sol_tmp_mean.update({'sol_mean': sol_mean})
    sol_grad_mean = np.mean([v for k, v in _sol_tmp_grad_mean.items() if k != 'epoch'])
    _sol_tmp_grad_mean.update({'sol_grad_mean': sol_grad_mean})

    _sol_tmp_grad_max = {'epoch': epoch}
    for name, param in sol_model.named_parameters():
        if param.requires_grad:
            _sol_tmp_grad_max.update({name: torch.norm(param.grad.data, p=float('inf')).item()})
    sol_grad_max = np.mean([v for k, v in _sol_tmp_grad_max.items() if k != 'epoch'])
    _sol_tmp_grad_max.update({'sol_grad_mean': sol_grad_max})
    
    _fb_tmp_mean = {'epoch': epoch}
    _fb_tmp_grad_mean = {'epoch': epoch}
    for name, param in fb_model.named_parameters():
        if param.requires_grad:
            _fb_tmp_mean.update({name: torch.mean(param).item()})
            _fb_tmp_grad_mean.update({name: torch.mean(param.grad).item()})
    fb_mean = np.mean([v for k, v in _fb_tmp_mean.items() if k != 'epoch'])
    _fb_tmp_mean.update({'fb_mean': fb_mean})
    fb_grad_mean = np.mean([v for k, v in _fb_tmp_grad_mean.items() if k != 'epoch'])
    _fb_tmp_grad_mean.update({'fb_grad_mean': fb_grad_mean})
    
    _fb_tmp_grad_max = {'epoch': epoch}
    for name, param in fb_model.named_parameters():
        if param.requires_grad:
            _fb_tmp_grad_max.update({name: torch.norm(param.grad.data, p=float('inf')).item()})
    fb_grad_max = np.mean([v for k, v in _fb_tmp_grad_max.items() if k != 'epoch'])
    _fb_tmp_grad_max.update({'fb_grad_mean': fb_grad_max})
    
    if if_print:
        print('sol_grad_max: {}, fb_grad_max: {}'.format(sol_grad_max, fb_grad_max))
        print('sol_mean    : {}, fb_mean    : {}'.format(sol_mean, fb_mean))

    return (_sol_tmp_grad_mean, _fb_tmp_grad_mean, _sol_tmp_grad_max, _fb_tmp_grad_max,
            _sol_tmp_mean, _fb_tmp_mean)

def record_grad(sol_model, fb_model, config, epoch):
    os.makedirs('DEBUG', exist_ok=True)
    _sol_tmp_grad_mean, _fb_tmp_grad_mean, _sol_tmp_grad_max, _fb_tmp_grad_max, _sol_tmp_mean, _fb_tmp_mean\
          = print_grad(sol_model, fb_model, epoch, if_print= (epoch % config.verbose == 0) )
    save_csv(pd.DataFrame([_sol_tmp_grad_mean]), 'DEBUG/sol_grad_mean.csv')
    save_csv(pd.DataFrame([_fb_tmp_grad_mean]), 'DEBUG/fb_grad_mean.csv')
    save_csv(pd.DataFrame([_sol_tmp_grad_max]), 'DEBUG/sol_grad_max.csv')
    save_csv(pd.DataFrame([_fb_tmp_grad_max]), 'DEBUG/fb_grad_max.csv')
    save_csv(pd.DataFrame([_sol_tmp_mean]), 'DEBUG/sol_mean.csv')
    save_csv(pd.DataFrame([_fb_tmp_mean]), 'DEBUG/fb_mean.csv')

def all_model_zero_grad(sol_model, fb_model):
    # zero parameter gradients and then compute NN prediction of gradient
    sol_model.zero_grad()
    fb_model.zero_grad()
    return sol_model, fb_model

def all_optimizer_zero_grad(sol_optimizer, fb_optimizer):
    # zero parameter gradients
    sol_optimizer.zero_grad()
    fb_optimizer.zero_grad()
    return sol_optimizer, fb_optimizer
##############################################################################################################
