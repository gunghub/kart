

from modules_official import dense_transforms
from modules_jt.global_variables import *
import torch.utils.tensorboard as tb
import torch
from modules_official.train import log
import os
from modules_official.util import *
from modules_jt.train_module_jt import *
from os import path
from modules_jt.dataset_module_jt import *
import torch.nn.functional as F
from modules_jt.train_module_jt2 import *


class TrainClassifierArgsJT:
  def __init__(self):
    self.log_dir=PROJECT_ROOT+"/train_log"
    self.num_epochs=150
    self.num_workers=4
    self.learning_rate=1e-3
    self.continue_training=True
    # self.transform='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])'
    self.transform='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2),RandomHorizontalFlip(), ToTensor()])'
    self.val_transform='Compose([ RandomHorizontalFlip(), ToTensor()])'


val_data_g=None
model_g=None

def train_classifier_jt(args:TrainClassifierArgsJT):


  print("Initializing Model =================================")
  model = args.classifier_type()
  print(model)
  train_logger, valid_logger = None, None
  if args.log_dir is not None:
      train_logger = tb.SummaryWriter(args.log_dir)

  """
  Your code here, modify your HW4 code
  
  """
  print("installing torch ...")

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  model = model.to(device)

  model_path=PROJECT_ROOT+"/model_base/"+args.classifier_name
  if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

  loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  
  print("installing inspect ...")

  import inspect
  transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
  print("loading data ...")
  train_data = load_data_jt2(args.trainset_dirs, tracks=args.tracks, transform=transform, num_workers=args.num_workers,proportion=args.proportion,random_seed=args.random_seed)
  
  
  val_transform = eval(args.val_transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
  
  global val_data_g
  
  val_data=load_data_jt2(args.valset_dirs, tracks=args.tracks, transform=val_transform, num_workers=args.num_workers,proportion=args.proportion,random_seed=args.random_seed)
  val_data_g=val_data
  
  
  global_step = 0
  
  ##### Initial val loss
  vc_args_jt=ValClassifierArgsJT()
  vc_args_jt.model=model
  vc_args_jt.val_data=val_data
  initial_val_loss,accuracy=val_classifier_jt(vc_args_jt)
  print("initial val loss =",initial_val_loss,"accuracy=",accuracy)
  
  
  print("Begin Training =================================")
  for epoch in range(args.num_epochs):
      model.train()
      losses = []
      for img, label, track_codes in train_data:
          img, label, track_codes = img.to(device), label.to(device), track_codes.to(device)

          pred = model(img)
          global model_g
          model_g=model
          loss_val = loss(pred, track_codes)

        #   if train_logger is not None:
        #       train_logger.add_scalar('loss', loss_val, global_step)
        #       if global_step % 100 == 0:
        #           log(train_logger, img, label, pred, global_step)

          optimizer.zero_grad()
          loss_val.backward()
          optimizer.step()
          global_step += 1
          
          losses.append(loss_val.detach().cpu().numpy())
      
      avg_loss = np.mean(losses)
      save_planner_jt(model,args.classifier_name,args.classifier_type)
      
      
      
      
      val_loss=None
      if epoch%args.val_interval==0:
      ################ Validation
        vc_args_jt=ValClassifierArgsJT()
        vc_args_jt.model=model
        vc_args_jt.val_data=val_data
        val_loss, accuracy=val_classifier_jt(vc_args_jt)
      
      
      ###### Print
      print('epoch %-3d \t loss = %0.6f' % (epoch, avg_loss),"val_loss=",val_loss, "accuracy=",accuracy)
      

  save_planner_jt(model,args.classifier_name,args.classifier_type)



class ValClassifierArgsJT:
  def __init__(self):
    self.num_workers=2


def val_classifier_jt(args:ValClassifierArgsJT):


  model = args.model
  # print(model)
  train_logger, valid_logger = None, None
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = model.to(device)
  loss = torch.nn.CrossEntropyLoss()
  val_data = args.val_data
  global_step = 0
  with torch.no_grad():
    
    
    for epoch in range(1):
      model.eval()
      losses = []
      correct=0
      for img, label, track_codes in val_data:
        
        img, label,track_codes = img.to(device), label.to(device), track_codes.to(device)

        pred = model(img)
        
        correct += (pred.argmax(1) == track_codes).type(torch.float).sum().item()
        # print(pred.argmax(1),track_codes)
        # print ((pred.argmax(1) == track_codes).type(torch.float).sum().item())
        loss_val = loss(pred, track_codes)
        
        

        # if train_logger is not None:
        #     train_logger.add_scalar('loss', loss_val, global_step)
        #     if global_step % 100 == 0:
        #         log(train_logger, img, label, pred, global_step)

        global_step += 1
        losses.append(loss_val.detach().cpu().numpy())
        
      avg_loss = np.mean(losses)
      accuracy=correct/len(val_data.dataset)
  
        # print('loss = %0.3f' % (avg_loss))
        
  return avg_loss, accuracy




# def test_classifier_jt(dataloader, model, loss_fn):
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#       for X, y in dataloader:
#           X, y = X.to(device), y.to(device)
#           pred = model(X)
#           test_loss += loss_fn(pred, y).item()
#           correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")