##############################################################################################
import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import argparse 
import random
# import scipy.io as io

from torch import optim, autograd
from matplotlib import pyplot as plt

# create training and testing datasets
from torch.utils.data import Dataset, DataLoader
from DataSets.Square2D import Sample_Points, Exact_Solution
from Utils import helper

# create neural network surrogate model
from Models.FcNet import FcNet

# load data from two datasets within the same loop
from itertools import cycle

print("pytorch version", torch.__version__, "\n")

## parser arguments
parser = argparse.ArgumentParser(description='Deep Residual Method for Poisson Equation with Dirichlet Boundary Condition')
# checkpoints
parser.add_argument('-c', '--checkpoint', default='Checkpoints/Square2D/DeepResidual_DirichletBVP/simulation_0', type=str, metavar='PATH', help='path to save checkpoint')
# figures 
parser.add_argument('-i', '--image', default='Images/Square2D/DeepResidual_DirichletBVP/simulation_0', type=str, metavar='PATH', help='path to save figures')
args = parser.parse_args()

# fixed random seed
seed = 1029
def seed_torch():
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()
##############################################################################################



##############################################################################################
## hyperparameter configuration
## ------------------------- ##
# problem setting
dim_prob = 2
sub_prob = 1 # Dirichlet boundary on Gamma
alpha = 1
## ------------------------- ##
# dataset setting
num_intrr_pts = 10000
num_bndry_pts_D = 1000 # each line segment of Dirichlet boundary
num_bndry_pts_G = 1000 # line segment of Dirichlet boundary (Gamma)
num_test_pts = 100 # each dimension

batch_num = 10 # for the ease of dataloader, require interior and boundary points have the same number of mini-batches
batchsize_intrr_pts = num_intrr_pts // batch_num
batchsize_bndry_pts_D = 3*num_bndry_pts_D // batch_num
batchsize_bndry_pts_G = num_bndry_pts_G // batch_num
## ------------------------- ##
# network setting
width = 30
depth = 3
## ------------------------- ##
# optimization setting
beta = 400 # intial penalty coefficient
num_epochs = 900
milestones = 400, 600, 700
##############################################################################################



##############################################################################################
print('*', '-' * 45, '*')
print('===> preparing training and testing datasets ...')
print('*', '-' * 45, '*')

# training dataset for sample points inside the domain
class TraindataInterior(Dataset):    
    def __init__(self, num_intrr_pts, dim_prob): 
        
        self.SmpPts_Interior = Sample_Points.SmpPts_Interior_Square2D(num_intrr_pts, dim_prob)
        self.f_Exact_SmpPts = Exact_Solution.f_Exact_Square2D(self.SmpPts_Interior[:,0], self.SmpPts_Interior[:,1])        

    def __len__(self):
        return len(self.SmpPts_Interior)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Interior[idx]
        f_SmpPt = self.f_Exact_SmpPts[idx]

        return [SmpPt, f_SmpPt]

# training dataset for sample points at the Dirichlet boundary
class TraindataBoundaryDirichlet(Dataset):    
    def __init__(self, num_bndry_pts_D, dim_prob):         
        
        self.SmpPts_Bndry_D = Sample_Points.SmpPts_Boundary_Square2D(num_bndry_pts_D, dim_prob)
        self.g_SmpPts = Exact_Solution.g_Exact_Square2D(self.SmpPts_Bndry_D[:,0], self.SmpPts_Bndry_D[:,1])        
        
    def __len__(self):
        return len(self.SmpPts_Bndry_D)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Bndry_D[idx]
        g_SmpPt = self.g_SmpPts[idx]

        return [SmpPt, g_SmpPt]    
    
# training dataset for sample points at the Neumann boundary
class TraindataGamma(Dataset):    
    def __init__(self, sub_prob, num_bndry_pts_G, dim_prob):         
        
        self.SmpPts_Bndry_G = Sample_Points.SmpPts_Interface_Square2D(num_bndry_pts_G, dim_prob)
        self.h_SmpPts = Exact_Solution.h_Exact_Square2D(sub_prob, self.SmpPts_Bndry_G[:,0], self.SmpPts_Bndry_G[:,1], alpha)        
        
    def __len__(self):
        return len(self.SmpPts_Bndry_G)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Bndry_G[idx]
        h_SmpPt = self.h_SmpPts[idx]

        return [SmpPt, h_SmpPt]    
    
# testing dataset for equidistant sample points over the entire domain
class Testdata(Dataset):    
    def __init__(self, num_test_pts): 
        
        self.SmpPts_Test = Sample_Points.SmpPts_Test_Square2D(num_test_pts)
        self.u_Exact_SmpPts = Exact_Solution.u_Exact_Square2D(self.SmpPts_Test[:,0], self.SmpPts_Test[:,1])        
        self.Gradu_x_Exact_SmpPts = Exact_Solution.Gradu_x_Exact_Square2D(self.SmpPts_Test[:,0], self.SmpPts_Test[:,1])
        self.Gradu_y_Exact_SmpPts = Exact_Solution.Gradu_y_Exact_Square2D(self.SmpPts_Test[:,0], self.SmpPts_Test[:,1])         
                 
    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Test[idx]
        u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
        Gradu_x_Exact_SmpPt = self.Gradu_x_Exact_SmpPts[idx]
        Gradu_y_Exact_SmpPt = self.Gradu_y_Exact_SmpPts[idx]
 
        return [SmpPt, u_Exact_SmpPt, Gradu_x_Exact_SmpPt, Gradu_y_Exact_SmpPt] 

# create training and testing datasets         
traindata_intrr = TraindataInterior(num_intrr_pts, dim_prob)
traindata_bndry_D = TraindataBoundaryDirichlet(num_bndry_pts_D, dim_prob)
traindata_bndry_G = TraindataGamma(sub_prob, num_bndry_pts_G, dim_prob)
testdata = Testdata(num_test_pts)

# define dataloader 
dataloader_intrr = DataLoader(traindata_intrr, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
dataloader_bndry_D = DataLoader(traindata_bndry_D, batch_size=batchsize_bndry_pts_D, shuffle=True, num_workers=0)
dataloader_bndry_G = DataLoader(traindata_bndry_G, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)
dataloader_test = DataLoader(testdata, batch_size=num_test_pts*num_test_pts, shuffle=False, num_workers=0)
##############################################################################################

    

##############################################################################################
# plot sample points during training and testing
if not os.path.isdir(args.image):
    helper.mkdir_p(args.image)

fig = plt.figure()
plt.scatter(traindata_intrr.SmpPts_Interior[:,0], traindata_intrr.SmpPts_Interior[:,1], c = 'red', label = 'interior points' )
plt.scatter(traindata_bndry_D.SmpPts_Bndry_D[:,0], traindata_bndry_D.SmpPts_Bndry_D[:,1], c = 'blue', label = 'Dirichlet boundary points' )
plt.scatter(traindata_bndry_G.SmpPts_Bndry_G[:,0], traindata_bndry_G.SmpPts_Bndry_G[:,1], c = 'green', label = 'Dirichlet boundary points' )
plt.title('Sample Points during Training')
plt.legend(loc = 'lower right')
# plt.show()
plt.savefig(os.path.join(args.image,'TrainSmpPts.png'))
plt.close(fig)

fig = plt.figure()
plt.scatter(testdata.SmpPts_Test[:,0], testdata.SmpPts_Test[:,1], c = 'black')
plt.title('Sample Points during Testing')
# plt.show()
plt.savefig(os.path.join(args.image,'TestSmpPts.png'))
plt.close(fig)
##############################################################################################


##############################################################################################
print('*', '-' * 45, '*')
print('===> creating training model ...')
print('*', '-' * 45, '*', "\n", "\n")

def train_epoch(epoch, model, optimizer, device):
    
    # set model to training mode
    model.train()

    loss_epoch, loss_intrr_epoch, loss_bndry_D_epoch, loss_bndry_G_epoch = 0, 0, 0, 0

    # ideally, sample points within the interior domain and at its boundary have the same number of mini-batches
    # otherwise, it wont's shuffle the dataloader_boundary samples again when it starts again (see https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7)
    for i, (data_intrr, data_bndry_D, data_bndry_G) in enumerate(zip(dataloader_intrr, cycle(dataloader_bndry_D), cycle(dataloader_bndry_G))):
        
        # get mini-batch training data
        smppts_intrr, f_smppts = data_intrr
        smppts_bndry_D, g_smppts = data_bndry_D
        smppts_bndry_G, h_smppts = data_bndry_G

        # send training data to device
        smppts_intrr = smppts_intrr.to(device)
        f_smppts = f_smppts.to(device)
        smppts_bndry_D = smppts_bndry_D.to(device)
        g_smppts = g_smppts.to(device)
        smppts_bndry_G = smppts_bndry_G.to(device)
        h_smppts = h_smppts.to(device)
        
        smppts_intrr.requires_grad = True
        
        # forward pass to obtain NN prediction of u(x)
        u_NN_intrr = model(smppts_intrr)
        u_NN_bndry_D = model(smppts_bndry_D)
        u_NN_bndry_G = model(smppts_bndry_G)
        # zero parameter gradients and then compute NN prediction of gradient u(x)
        model.zero_grad()
        # grad_u_NN_intrr = [grad_u_x, grad_u_y]
        grad_u_NN_intrr = torch.autograd.grad(outputs=u_NN_intrr, inputs=smppts_intrr, grad_outputs = torch.ones_like(u_NN_intrr), retain_graph=True, create_graph=True, only_inputs=True)[0]        
        # grad_ux_NN_intrr = [grad_u_xx, grad_u_xy]
        grad_ux_NN_intrr = torch.autograd.grad(outputs=torch.squeeze(grad_u_NN_intrr[:,0]), inputs=smppts_intrr, grad_outputs = torch.ones_like(grad_u_NN_intrr[:,0]), retain_graph=True, create_graph=True, only_inputs=True)[0]        
        # grad_uy_NN_intrr = [grad_u_yx, grad_u_yy]
        grad_uy_NN_intrr = torch.autograd.grad(outputs=torch.squeeze(grad_u_NN_intrr[:,1]), inputs=smppts_intrr, grad_outputs = torch.ones_like(grad_u_NN_intrr[:,1]), retain_graph=True, create_graph=True, only_inputs=True)[0]        

        # construct mini-batch loss function and then perform backward pass
        # loss_intrr = torch.mean(0.5 * torch.sum(torch.pow(grad_u_NN_intrr, 2), dim=1) - f_smppts * torch.squeeze(u_NN_intrr))
        loss_intrr = torch.mean(torch.pow(grad_ux_NN_intrr[:,0] + grad_uy_NN_intrr[:,1] + f_smppts, 2))  # div(grad): grad_u_xx + grad_u_yy
        loss_bndry_D = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D) - g_smppts, 2))
        loss_bndry_G = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_G) - h_smppts, 2))

        loss_minibatch = loss_intrr + beta * (loss_bndry_D + loss_bndry_G)

        # zero parameter gradients
        optimizer.zero_grad()
        # backpropagation
        loss_minibatch.backward()
        # parameter update
        optimizer.step()     

        # integrate loss over the entire training datset
        loss_intrr_epoch += loss_intrr.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
        loss_bndry_D_epoch += loss_bndry_D.item() * smppts_bndry_D.size(0) / traindata_bndry_D.SmpPts_Bndry_D.shape[0]
        loss_bndry_G_epoch += loss_bndry_G.item() * smppts_bndry_G.size(0) / traindata_bndry_G.SmpPts_Bndry_G.shape[0]
        loss_epoch += loss_intrr_epoch + beta * (loss_bndry_D_epoch + loss_bndry_G_epoch)                          
        
    return loss_intrr_epoch, loss_bndry_D_epoch, loss_bndry_G_epoch, loss_epoch
##############################################################################################



##############################################################################################
print('*', '-' * 45, '*')
print('===> creating testing model ...')
print('*', '-' * 45, '*', "\n", "\n")

def test_epoch(epoch, model, optimizer, device):
    
    # set model to testing mode
    model.eval()

    epoch_loss_u, epoch_loss_gradu_x, epoch_loss_gradu_y = 0, 0, 0
    for smppts_test, u_exact_smppts, gradu_x_exact_smppts, gradu_y_exact_smppts in dataloader_test:
        
        # send inputs, outputs to device
        smppts_test = smppts_test.to(device)
        u_exact_smppts = u_exact_smppts.to(device)  
        gradu_x_exact_smppts = gradu_x_exact_smppts.to(device)
        gradu_y_exact_smppts = gradu_y_exact_smppts.to(device)
        
        smppts_test.requires_grad = True
        
        # forward pass and then compute loss function for approximating u by u_NN
        u_NN_smppts = model(smppts_test) 
        
        loss_u = torch.mean(torch.pow(torch.squeeze(u_NN_smppts) - u_exact_smppts, 2))         
        
        # backward pass to obtain gradient and then compute loss function for approximating grad_u by grad_u_NN
        model.zero_grad()
        gradu_NN_smppts = torch.autograd.grad(outputs=u_NN_smppts, inputs=smppts_test, grad_outputs=torch.ones_like(u_NN_smppts), retain_graph=True, create_graph=True, only_inputs=True)[0]
        
        loss_gradu_x = torch.mean(torch.pow(torch.squeeze(gradu_NN_smppts[:,0]) - gradu_x_exact_smppts, 2)) 
        loss_gradu_y = torch.mean(torch.pow(torch.squeeze(gradu_NN_smppts[:,1]) - gradu_y_exact_smppts, 2))
                
        # integrate loss      
        epoch_loss_u += loss_u.item()         
        epoch_loss_gradu_x += loss_gradu_x.item()  
        epoch_loss_gradu_y += loss_gradu_y.item()  
    
    return epoch_loss_u, epoch_loss_gradu_x, epoch_loss_gradu_y
##############################################################################################



##############################################################################################
print('*', '-' * 45, '*')
print('===> neural network training ...')

if not os.path.isdir(args.checkpoint):
    helper.mkdir_p(args.checkpoint)

# create model
model = FcNet.FcNet(dim_prob,width,1,depth)
# for i in model.modules():
#     print(i)
#     input()
model.Xavier_initi()
print('Network Architecture:', "\n", model)
print('Total number of trainable parameters = ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# create optimizer and learning rate schedular
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

# load model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: {}'.format(device), "\n")
model = model.to(device)

# create log file
logger = helper.Logger(os.path.join(args.checkpoint, 'log.txt'), title='Deep-Rtiz-Poisson-Square2D')
logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'TrainLoss Interior', 'TrainLoss Bndry_D', 'TrainLoss Bndry_G', 'TestLoss Gradu_x', 'TestLoss Gradu_y'])

# train and test 
train_loss, test_loss_u, test_loss_gradu_x, test_loss_gradu_y = [], [], [], []
trainloss_best = 1e10
since = time.time()
for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs-1), 'with LR = {:.1e}'.format(optimizer.param_groups[0]['lr']))  

    # execute training and testing
    trainloss_intrr_epoch, trainloss_bndry_D_epoch, trainloss_bndry_G_epoch, trainloss_epoch = train_epoch(epoch, model, optimizer, device)
    testloss_u_epoch, testloss_gradu_x_epoch, testloss_gradu_y_epoch = test_epoch(epoch, model, optimizer, device)

    # save current and best models to checkpoint
    is_best = trainloss_epoch < trainloss_best
    if is_best:
        print('==> Saving best model ...')
    trainloss_best = min(trainloss_epoch, trainloss_best)
    helper.save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'trainloss_intrr_epoch': trainloss_intrr_epoch,
                            'trainloss_bndry_D_epoch': trainloss_bndry_D_epoch,
                            'trainloss_bndry_G_epoch': trainloss_bndry_G_epoch,
                            'trainloss_epoch': trainloss_epoch,
                            'testloss_u_epoch': testloss_u_epoch,
                            'testloss_gradu_x_epoch': testloss_gradu_x_epoch,
                            'testloss_gradu_y_epoch': testloss_gradu_y_epoch,
                            'trainloss_best': trainloss_best,
                            'optimizer': optimizer.state_dict(),
                           }, is_best, checkpoint=args.checkpoint)   
    # save training process to log file
    logger.append([optimizer.param_groups[0]['lr'], trainloss_epoch, testloss_u_epoch, trainloss_intrr_epoch, trainloss_bndry_D_epoch, trainloss_bndry_G_epoch, testloss_gradu_x_epoch, testloss_gradu_y_epoch])
    
    # adjust learning rate according to predefined schedule
    schedular.step()

    # print results
    train_loss.append(trainloss_epoch)
    test_loss_u.append(testloss_u_epoch)
    test_loss_gradu_x.append(testloss_gradu_x_epoch)
    test_loss_gradu_y.append(testloss_gradu_y_epoch)
    print('==> Full-Batch Training Loss = {:.4e}'.format(trainloss_epoch))
    print('    Fubb-Batch Testing Loss : ', 'u-u_NN = {:.4e}'.format(testloss_u_epoch), '  Grad_x(u-u_NN) = {:.4e}'.format(testloss_gradu_x_epoch), '  Grad_y(u-u_NN) = {:.4e}'.format(testloss_gradu_y_epoch), "\n")

logger.close()
time_elapsed = time.time() - since

# # save learning curves
# helper.save_learncurve({'train_curve': train_loss, 'test_curve': test_loss}, curve=args.image)  

print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
##############################################################################################



##############################################################################################
# plot learning curves
fig = plt.figure()
plt.plot(torch.log10(torch.tensor(train_loss)), c = 'red', label = 'training loss' )
plt.title('Learning Curve during Training')
plt.legend(loc = 'upper right')
# plt.show()
fig.savefig(os.path.join(args.image,'TrainCurve.png'))

fig = plt.figure()
plt.plot(torch.log10(torch.tensor(test_loss_u)), c = 'red', label = 'testing loss (u)' )
plt.plot(torch.log10(torch.tensor(test_loss_gradu_x)), c = 'blue', label = 'testing loss (gradu_x)' )
plt.plot(torch.log10(torch.tensor(test_loss_gradu_y)), c = 'black', label = 'testing loss (gradu_y)' )
plt.title('Learning Curve during Testing')
plt.legend(loc = 'upper right')
# plt.show()
fig.savefig(os.path.join(args.image,'TestCurve.png'))
##############################################################################################



##############################################################################################
print('*', '-' * 45, '*')
print('===> loading trained model for inference ...')

# load trained model
checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])

# compute NN predicution of u and gradu
with torch.no_grad():    
    SmpPts_Test = testdata.SmpPts_Test
    SmpPts_Test = SmpPts_Test.to(device)
    u_NN = model(SmpPts_Test).cpu()

SmpPts_Test.requires_grad = True
u_NN_Test = model(SmpPts_Test).cpu()
model.zero_grad()
gradu_NN_test = torch.autograd.grad(
                    outputs=u_NN_Test, 
                    inputs=SmpPts_Test, 
                    grad_outputs=torch.ones_like(u_NN_Test), 
                    retain_graph=True, 
                    create_graph=True, 
                    only_inputs=True)[0].cpu()

x = testdata.SmpPts_Test[:,0].reshape(num_test_pts,num_test_pts)
y = testdata.SmpPts_Test[:,1].reshape(num_test_pts,num_test_pts)

with torch.no_grad(): 
    # plot u and its network prediction on testing dataset
    plt.figure()
    u_Exact = testdata.u_Exact_SmpPts.reshape(num_test_pts,num_test_pts).cpu()
    plt.contourf(x, y, u_Exact, 40, cmap = 'jet')
    plt.title('Exact Solution u on Test Dataset')
    plt.colorbar()
    # plt.show()  
    plt.savefig(os.path.join(args.image,'Exact_u_TestData.png'))
    plt.close(fig)

    plt.figure()
    u_NN = u_NN.reshape(num_test_pts, num_test_pts)    
    plt.contourf(x, y, u_NN, 40, cmap = 'jet')
    plt.title('Network Prediction u_NN on Test Dataset')
    plt.colorbar()
    # plt.show()  
    plt.savefig(os.path.join(args.image,'NN_u_TestData.png'))
    plt.close(fig)

    plt.figure()
    u_Exact = testdata.u_Exact_SmpPts.reshape(num_test_pts,num_test_pts).cpu()
    err = u_Exact - u_NN
    plt.contourf(x, y, err, 40, cmap = 'jet')
    plt.title('Pointwise AppErr u-u_NN on Test Dataset')
    plt.colorbar()
    # plt.show() 
    plt.savefig(os.path.join(args.image,'AppErr_u_TestData.png'))
    plt.close(fig)
        
    # plot gradu_x and its network prediction on testing dataset
    plt.figure()
    gradu_x_Exact = testdata.Gradu_x_Exact_SmpPts.reshape(num_test_pts,num_test_pts).cpu()
    plt.contourf(x, y, gradu_x_Exact, 40, cmap = 'jet')
    plt.title('Exact Solution Grad_x(u) on Test Dataset')
    plt.colorbar()
    # plt.show()
    plt.savefig(os.path.join(args.image,'Exact_gradu_x_TestData.png'))
    plt.close(fig)

    plt.figure()
    gradu_NN_x_test = gradu_NN_test[:,0].detach().numpy().reshape(num_test_pts,num_test_pts)
    plt.contourf(x, y, gradu_NN_x_test, 40, cmap = 'jet')
    plt.title('Network Prediction Grad_x(u_NN) on Test Dataset')
    plt.colorbar()
    # plt.show()
    plt.savefig(os.path.join(args.image,'NN_gradu_x_TestData.png'))
    plt.close(fig)

    plt.figure()
    plt.contourf(x, y, gradu_x_Exact - gradu_NN_x_test, 40, cmap = 'jet')
    plt.title('Poinwise AppErr Grad_x(u-u_NN) on Test Dataset')
    plt.colorbar()
    # plt.show()  
    plt.savefig(os.path.join(args.image,'AppErr_gradu_x_TestData.png')) 
    plt.close(fig)
        
    # plot gradu_y and its network prediction on testing dataset
    plt.figure()
    gradu_y_Exact = testdata.Gradu_y_Exact_SmpPts.reshape(num_test_pts,num_test_pts).cpu()
    plt.contourf(x, y, gradu_y_Exact, 40, cmap = 'jet')
    plt.title('Exact Solution Grad_y(u) on Test Dataset')
    plt.colorbar()
    # plt.show()
    plt.savefig(os.path.join(args.image,'Exact_gradu_y_TestData.png')) 
    plt.close(fig)

    plt.figure()
    gradu_NN_y_test = gradu_NN_test[:,1].detach().numpy().reshape(num_test_pts,num_test_pts)
    plt.contourf(x, y, gradu_NN_y_test, 40, cmap = 'jet')
    plt.title('Network Prediction Grad_y(u_NN) on Test Dataset')
    plt.colorbar()
    # plt.show()
    plt.savefig(os.path.join(args.image,'NN_gradu_y_TestData.png')) 
    plt.close(fig)

    plt.figure()
    plt.contourf(x, y, gradu_y_Exact - gradu_NN_y_test, 40, cmap = 'jet')
    plt.title('Poinwise AppErr Grad_y(u-u_NN) on Test Dataset')
    plt.colorbar()
    # plt.show()
    plt.savefig(os.path.join(args.image,'AppErr_gradu_y_TestData.png')) 
    plt.close(fig)
        
print('*', '-' * 45, '*', "\n", "\n")
##############################################################################################







  
