
import numpy as np
def generate_dataset_jt(pytux, tracks, control_func, output, images_per_track=1000, steps_per_track=20000, aim_noise=0.1, vel_noise=5, verbose=False):
    from os import makedirs, remove

    try:
        makedirs("dataset_base/"+output)
    except OSError:
        pass



    for track in tracks:
        output_track="dataset_base/"+output+"/"+track
        try:
          makedirs(output_track)
        except OSError:
          pass
        n= 0

        def collect(_, im, pt):
            from PIL import Image
            from os import path
            nonlocal n
            id = n if n < images_per_track else np.random.randint(0, n + 1)
            if id < images_per_track:
                fn = path.join(output_track, track + '_%05d' % id)
                Image.fromarray(im).save(fn + '.png')
                with open(fn + '.csv', 'w') as f:
                    f.write('%0.1f,%0.1f' % tuple(pt))
            n += 1

        # Use 0 noise for the first round
        _aim_noise, _vel_noise = 0, 0

        while n < steps_per_track:
            def noisy_control(aim_pt, vel):
                return control_func(
                        aim_pt + np.random.randn(*aim_pt.shape) * _aim_noise,
                        vel + np.random.randn() * _vel_noise)

            steps, how_far = pytux.rollout(track, noisy_control, max_frames=1000, verbose=verbose, data_callback=collect)
            print(track,steps, how_far)

            # Add noise after the first round
            _aim_noise, _vel_noise = aim_noise, vel_noise