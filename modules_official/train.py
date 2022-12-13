
import torch
import torch.utils.tensorboard as tb
import numpy as np
from os import path
import torch
import inspect
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from modules_jt.global_variables import *
from .planner import *

class TrainArgs:
  def __init__(self):
    self.log_dir=PROJECT_ROOT+"/train_log"
    self.num_epoch=150
    self.num_workers=4
    self.learning_rate=1e-3
    self.continue_training=True
    self.transform='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])'

def train(args:TrainArgs):


  print("Initializing Model =================================")
  model = Planner()
  train_logger, valid_logger = None, None
  if args.log_dir is not None:
      train_logger = tb.SummaryWriter(args.log_dir)

  """
  Your code here, modify your HW4 code
  
  """
  print("installing torch ...")


  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  model = model.to(device)
  if args.continue_training:
      model.load_state_dict(torch.load(path.join(path.dirname(FILE_PATH), 'planner.th')))

  loss = torch.nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  
  print("installing inspect ...")
  
  transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
  print("loading data ...")
  train_data = load_data(args.trainset_dir, transform=transform, num_workers=args.num_workers)

  global_step = 0
  print("Begin Training =================================")
  for epoch in range(args.num_epoch):
      model.train()
      losses = []
      for img, label in train_data:
          img, label = img.to(device), label.to(device)

          pred = model(img)
          loss_val = loss(pred, label)

          if train_logger is not None:
              train_logger.add_scalar('loss', loss_val, global_step)
              if global_step % 100 == 0:
                  log(train_logger, img, label, pred, global_step)

          optimizer.zero_grad()
          loss_val.backward()
          optimizer.step()
          global_step += 1
          
          losses.append(loss_val.detach().cpu().numpy())
      
      avg_loss = np.mean(losses)
      # if train_logger is None:
      if True:
          print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
      save_model(model)

  save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """

    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--log_dir')
#     # Put custom arguments here
#     parser.add_argument('-n', '--num_epoch', type=int, default=150)
#     parser.add_argument('-w', '--num_workers', type=int, default=4)
#     parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
#     parser.add_argument('-c', '--continue_training', action='store_true')
#     parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

#     args = parser.parse_args()
#     train(args)

