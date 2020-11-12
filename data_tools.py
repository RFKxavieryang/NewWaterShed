import SimpleITK as sitk
import numpy as np
import scipy
import scipy.signal



def load_nrrd(full_path_filename):
    "Load an nrrd file - provided by the challenge organizers"
    data = sitk.ReadImage(full_path_filename)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
    data = sitk.GetArrayFromImage(data)
    return data


def regularize(x, output=False):
    "Normalize the input volume"
    x = x.astype('float32')
    low = 0

    eigth = tuple(3 * i // 8 for i in x.shape)

    hist, bins = np.histogram(x[eigth[0] : -eigth[0], eigth[1] : -eigth[1], eigth[2] : -eigth[2]].flatten(), \
            bins=np.arange(256), range=(low, 255))

    #  cumsum = np.cumsum(hist)
    #  oldmax = x.max()

    #  argmaxs = scipy.signal.argrelextrema(hist, np.greater)[0]

    #  if len(argmaxs) < 1:
        #  return x
    #  lastarg = argmaxs[-1]
    #  lastmax = bins[lastarg]
    #  if output:
        #  print(lastmax)
    #  x = x * 120 / lastmax

    nmax = np.nonzero(hist)[0][-1] # FIXME: check len

    x = x * 255 / nmax

    x = np.clip(x, 0, 255)
    return x

