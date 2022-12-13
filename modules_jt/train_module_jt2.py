
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


def load_data_jt2(dataset_names, tracks,transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128, proportion=1.0,random_seed=200):
  dataset_list=[]
  for dataset_name in dataset_names:
      for track in tracks:
        dataset = SuperTuxDatasetJT(dataset_name,track_name=track, transform=transform,proportion=proportion,random_seed=random_seed)
        dataset_list.append(dataset)
  
  datasets = torch.utils.data.ConcatDataset(dataset_list)


  return DataLoader(datasets, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


class TrainArgsJT2:
  def __init__(self):
    self.log_dir=PROJECT_ROOT+"/train_log"
    self.num_epochs=150
    self.num_workers=4
    self.learning_rate=1e-3
    self.continue_training=True
    # self.transform='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])'
    self.transform='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2),RandomHorizontalFlip(), ToTensor()])'
    self.val_transform='Compose([ RandomHorizontalFlip(), ToTensor()])'
    self.loss=torch.nn.L1Loss()
    self.device="cuda:0"



def train_jt2(args:TrainArgsJT2):


  print("Initializing Model =================================")
  model = args.planner_type()
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

  model_path=PROJECT_ROOT+"/model_base/"+args.planner_name
  if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

  loss = args.loss
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  
  print("installing inspect ...")

  import inspect
  transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
  print("loading data ...")
  train_data = load_data_jt2(args.trainset_dirs, tracks=args.tracks, transform=transform, num_workers=args.num_workers,proportion=args.proportion,random_seed=args.random_seed)
  
  
  val_transform = eval(args.val_transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
  val_data=load_data_jt2(args.valset_dirs, tracks=args.tracks, transform=val_transform, num_workers=args.num_workers,proportion=args.proportion,random_seed=args.random_seed)

  global_step = 0
  
  ##### Initial val loss
  vdt_args_jt=ValDuringTrainArgsJT()
  vdt_args_jt.model=model
  vdt_args_jt.val_data=val_data
  initial_val_loss=val_planner_during_train_jt(vdt_args_jt)
  print("initial val loss =",initial_val_loss)
  
  
  print("Begin Training =================================")
  for epoch in range(args.num_epochs):
      model.train()
      losses = []
      for img, label,_ in train_data:
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
      save_planner_jt(model,args.planner_name,args.planner_type)
      
      
      val_loss=None
      if epoch%args.val_interval==0:
      ################ Validation
        vdt_args_jt=ValDuringTrainArgsJT()
        vdt_args_jt.model=model
        vdt_args_jt.val_data=val_data
        val_loss=val_planner_during_train_jt(vdt_args_jt)
      
      
      
      ###### Print
      print('epoch %-3d \t loss = %0.6f' % (epoch, avg_loss),"val_loss=",val_loss)
      

  save_planner_jt(model,args.planner_name,args.planner_type)



class ValDuringTrainArgsJT:
  def __init__(self):
    self.num_workers=2
    self.device="cuda:0"

    


def val_planner_during_train_jt(args:ValDuringTrainArgsJT):


  model = args.model
  # print(model)
  train_logger, valid_logger = None, None
  device = args.device
  model = model.to(device)
  loss = torch.nn.L1Loss()
  val_data = args.val_data
  global_step = 0
  with torch.no_grad():
    for epoch in range(1):
        model.eval()
        losses = []
        for img, label,_ in val_data:
            img, label = img.to(args.device), label.to(args.device)

            pred = model(img)
            loss_val = loss(pred, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 100 == 0:
                    log(train_logger, img, label, pred, global_step)

            global_step += 1
            losses.append(loss_val.detach().cpu().numpy())
        
        avg_loss = np.mean(losses)
  
        # print('loss = %0.3f' % (avg_loss))
        
  return avg_loss