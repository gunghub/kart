import pystk
def control_func_jt(aim_point, current_vel):
  
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    
    """
    
    aim_direction=aim_point[0]
    

    # aim velocity
    HIGHEST_VEL=30
    LOWEST_VEL=8
    SLOPE=HIGHEST_VEL-LOWEST_VEL
    aim_vel=HIGHEST_VEL-SLOPE*aim_direction*aim_direction
    

    # accelerate and brake
    if current_vel<aim_vel:
        action.acceleration=True
        action.brake=False
    elif current_vel>aim_vel:
        action.acceleration=False
        action.brake=True
    elif current_vel==aim_vel:
        action.brake=False
        action.acceleration=False


    # steer
    STEER_COEF=3
    action.steer=aim_direction*STEER_COEF
    action.steer=min(1,action.steer)
    action.steer=max(-1,action.steer)
    

    # nitro
    NITRO_THRESHOLD=8
    if aim_vel-current_vel>NITRO_THRESHOLD:
      action.nitro=True
    else:
      action.nitro=False
  


    # drift
    DRIFT_THRESHOLD=0.5
    if abs(aim_direction)>DRIFT_THRESHOLD:
      action.drift=True
    else:
      action.drift=False



    return action

  
def test_control_func_jt(pytux, track, control_func, verbose=False):
    import numpy as np

    track = [track] if isinstance(track, str) else track

    for t in track:
        steps, how_far = pytux.rollout(t, control_func, max_frames=1000, verbose=verbose)
        print(steps, how_far)