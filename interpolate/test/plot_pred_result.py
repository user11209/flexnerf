import os
import shutil
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import subprocess
import math

VANILLA = False

test_data_path = os.path.join( os.path.split(os.path.abspath(__file__))[0], "test_data", "test_data.pt" )
pred_dir = os.path.join( os.path.split(os.path.abspath(__file__))[0], "pred_data" )
if VANILLA:
    pred_data_path = os.path.join( pred_dir, "pred.pt" )
    png_dir = os.path.join( os.path.split(os.path.abspath(__file__))[0], "pred_data", "png" )
    video_dir = os.path.join(pred_dir, "pred_data.mp4")
else:
    pred_data_path = os.path.join( pred_dir, "pred_adaptive.pt" )
    png_dir = os.path.join( os.path.split(os.path.abspath(__file__))[0], "pred_data", "png_adaptive" )
    video_dir = os.path.join(pred_dir, "pred_data_adaptive.mp4")


if not os.path.exists(png_dir):
    os.mkdir(png_dir)

test_data = torch.load(test_data_path, map_location=torch.device("cpu"))
gaussian_sample = test_data["sample"]
gaussian_label = test_data["label"]
func_config = test_data["func_config"]
mean_offset = test_data["mean_offset"]
mean_variance = test_data["mean_variance"]
SAMPLE = gaussian_sample.shape[0]

pred_data = torch.load(pred_data_path, map_location=torch.device("cpu"))
est_end_value = pred_data["est"]
if "end" not in pred_data:
    end = torch.tensor([[-3],[3]], dtype=torch.float32)
else:
    end = pred_data["end"]
if "gaussian" not in pred_data:
    pred_gaussian = None
else:
    pred_gaussian = pred_data["gaussian"]
    pred_gaussian.requires_grad = False
est_end_value.requires_grad = False


def map_id_cmap(min, id):
    ret = torch.where(id < min, 0.2, 0.2+(id-min)/100*0.8)
    return ret

for i in range(0, SAMPLE, 10):
    plt.xlim((-3,3))
    plt.ylim((-50,50))
    colors = cm.Greys(map_id_cmap(i-100, torch.arange(i)))
    plt.scatter(gaussian_sample[:i], gaussian_label[:i], color=colors)
    plt.axline((0, func_config[i, -1]), slope=func_config[i, 0], color="red")
    gaussian_range = plt.Rectangle((mean_offset[i, 0]-mean_variance[i,0], -50), 2*mean_variance[i,0], 5, fc="red", alpha=0.5)
    plt.gca().add_patch(gaussian_range)

    est_i = est_end_value[i,:,0]
    end_i = end[:,0]
    plt.plot(end_i, est_i, color="blue")
    if pred_gaussian != None:
        pred_mean = pred_gaussian[i, 0, 1]
        pred_var_sqrt = math.sqrt( pred_gaussian[i, 0, 0] )
        pred_gaussian_range = plt.Rectangle((pred_mean-pred_var_sqrt, -45), 2*pred_var_sqrt, 5, fc="blue", alpha=0.5)
        plt.gca().add_patch(pred_gaussian_range)

    plt.savefig(os.path.join(png_dir, "example_"+"{}".format(i).zfill(6)) + ".png")
    plt.cla()

subprocess.call([
    'ffmpeg', '-i', '{}/example_%05d0.png'.format(png_dir), '-r', '4',
    video_dir
])