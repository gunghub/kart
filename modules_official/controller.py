import pystk




def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
  import numpy as np
  #this seems to initialize an object
  action = pystk.Action()

  

  #compute acceleration
  action.acceleration = np.clip(target_vel - current_vel ,0,1)
  
  if current_vel > target_vel:
    action.brake = True
    action.nitro = False
  else:
    action.brake = False	
    action.nitro = True
  
  
  # Compute steering
  action.steer = np.clip(steer_gain * aim_point[0], -1, 1)

  # Compute skidding
  if abs(aim_point[0]) > skid_thresh:
      action.drift = True

  else:
      action.drift = False
      

  

  return action
  

def test_controller(pytux, track, verbose=False):
    import numpy as np

    track = [track] if isinstance(track, str) else track

    for t in track:
        steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=verbose)
        print(steps, how_far)




# from .utils import PyTux
# if __name__ == '__main__':
#     from argparse import ArgumentParser

#     parser = ArgumentParser()
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')

#     pytux = PyTux()
#     test_controller(pytux, **vars(parser.parse_args()))
#     pytux.close()
