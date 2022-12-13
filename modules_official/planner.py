import torch
import torch.nn.functional as F
from modules_jt.global_variables import *

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
   
   
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    #ok, let's discuss what this just did
    # logit.view(logit.size(0), -1) reshapes the input, which is a 1xaxb tensor as 
    # 1 x ab
    #softmax then takes the softmax
    #finally, this is converted back to a 1xaxb tensor

    #the following code computes the center of mass
    #formula in more convenient form is given in slides
    firstcoord = (weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1)
    secondcoord = (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)
    a = torch.stack((firstcoord, secondcoord), 1)

    return a


class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 32, 32]):

        super().__init__()

        conv_block = lambda c, h: [torch.nn.Conv2d(h, c, 5, 2, 2), torch.nn.ReLU(True)]

        h, _conv = 3, []
        for c in channels:
            _conv += conv_block(c, h)
            h = c

        self._conv = torch.nn.Sequential(*_conv, torch.nn.Conv2d(h, 1, 1))
        # self.classifier = torch.nn.Linear(h, 2)
        # self.classifier = torch.nn.Conv2d(h, 1, 1)

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        #print(img.shape)
        #print(x.shape)
        return spatial_argmax(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(FILE_PATH), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(FILE_PATH), 'planner.th'), map_location='cpu'))
    return r







def test_planner(pytux, tracks, verbose):
    # Load model
    planner = load_model().eval()
    print(planner)
    for t in tracks:
        steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=verbose)
        print(steps, how_far)


# if __name__ == '__main__':
#     from controller import control
#     from utils import PyTux
#     from argparse import ArgumentParser
#     from torchvision import models

#     def test_planner(args):
#         # Load model
#         planner = load_model().eval()
#         print(planner)
#         pytux = PyTux()
#         for t in args.track:
#             steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
#             print(steps, how_far)
#         pytux.close()


#     parser = ArgumentParser("Test the planner")
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')
#     args = parser.parse_args()
#     test_planner(args)
