
from modules_official import dense_transforms
from modules_jt.global_variables import *
import torch.utils.tensorboard as tb
import torch
from modules_official.train import log
import os
from modules_official.util import *
import torch.nn.functional as F



# def load_data_jt(dataset_paths, tracks,transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
#   dataset_list=[]
#   for dataset_path in dataset_paths:
#       for track in tracks:
#         dataset = SuperTuxDataset("dataset_base/"+dataset_path+"/"+track, transform=transform)
#         dataset_list.append(dataset)
  
#   datasets = torch.utils.data.ConcatDataset(dataset_list)

#   return DataLoader(datasets, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


# class TrainArgsJT:
#   def __init__(self):
#     self.log_dir=PROJECT_ROOT+"/train_log"
#     self.num_epochs=150
#     self.num_workers=4
#     self.learning_rate=1e-3
#     self.continue_training=True
#     self.transform='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])'
#     self.loss=torch.nn.L1Loss()
    
    


# def train_jt(args:TrainArgsJT):


#   print("Initializing Model =================================")
#   model = args.planner_type()
#   print(model)
#   train_logger, valid_logger = None, None
#   if args.log_dir is not None:
#       train_logger = tb.SummaryWriter(args.log_dir)

#   """
#   Your code here, modify your HW4 code
  
#   """
#   print("installing torch ...")

#   device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#   model = model.to(device)

#   model_path=PROJECT_ROOT+"/model_base/"+args.planner_name
#   if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path))

#   loss = args.loss
#   optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  
#   print("installing inspect ...")

#   import inspect
#   transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
#   print("loading data ...")
#   train_data = load_data_jt(args.trainset_dirs, tracks=args.tracks, transform=transform, num_workers=args.num_workers)

#   global_step = 0
#   print("Begin Training =================================")
#   for epoch in range(args.num_epochs):
#       model.train()
#       losses = []
#       for img, label in train_data:
#           img, label = img.to(device), label.to(device)

#           pred = model(img)
#           loss_val = loss(pred, label)

#           if train_logger is not None:
#               train_logger.add_scalar('loss', loss_val, global_step)
#               if global_step % 100 == 0:
#                   log(train_logger, img, label, pred, global_step)

#           optimizer.zero_grad()
#           loss_val.backward()
#           optimizer.step()
#           global_step += 1
          
#           losses.append(loss_val.detach().cpu().numpy())
      
#       avg_loss = np.mean(losses)

#       print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
#       save_planner_jt(model,args.planner_name,args.planner_type)

#   save_planner_jt(model,args.planner_name,args.planner_type)


def save_planner_jt(planner,planner_name,planner_type):
  from torch import save
  from os import path
  if isinstance(planner, planner_type):
    model_path=PROJECT_ROOT+"/model_base/"+planner_name
    
    save(planner.state_dict(), model_path)
    # print("saved ",planner_name)
  
  else:
    raise ValueError("planner type '%s' not supported!" % str(type(planner)))


def load_planner_jt(planner_name,planner_type):
    from torch import load
    from os import path
    planner = planner_type()
    model_path=PROJECT_ROOT+"/model_base/"+planner_name
    planner.load_state_dict(load(model_path, map_location='cpu'))
    return planner


