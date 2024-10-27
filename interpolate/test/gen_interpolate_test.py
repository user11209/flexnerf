import os
import shutil
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import subprocess

STORE_GAUSSIAN = True
SKIP_PNG = True
CLEAR_LAST = False
GEN_VIDEO = False

gen_dir = os.path.join( os.path.split(os.path.abspath(__file__))[0], "test_data" )
if CLEAR_LAST:
    shutil.rmtree(gen_dir)
if not os.path.exists(gen_dir):
    os.mkdir(gen_dir)
if not os.path.exists(os.path.join(gen_dir, "png")):
    os.mkdir(os.path.join(gen_dir, "png"))
DIM = 1
SAMPLE = 10000
torch.manual_seed(1567)

if STORE_GAUSSIAN or not SKIP_PNG:
    gaussian_base = torch.empty(SAMPLE, DIM).normal_(mean=0, std=1)
    mean_offset = torch.empty(SAMPLE, DIM)
    mean_offset_amplitude_config = torch.empty(DIM).uniform_(0,1)
    mean_offset_frequency_config = 1/SAMPLE * torch.pow(10, torch.empty(DIM).uniform_(1,2))
    for i in range(DIM):
        mean_offset[:, i] = torch.sin(mean_offset_frequency_config[i] * torch.arange(SAMPLE)) * \
                            mean_offset_amplitude_config[i]

    mean_variance = torch.empty(SAMPLE, DIM)
    mean_variance_amplitude_config = torch.empty(DIM).uniform_(0.1, 1)
    mean_variance_decay_config = torch.empty(DIM).uniform_(1,2)
    for i in range(DIM):
        mean_variance[:, i] = torch.pow(3, -mean_variance_decay_config[i] * torch.arange(SAMPLE)/SAMPLE) * \
                            mean_variance_amplitude_config[i]

    gaussian_sample = gaussian_base * mean_variance + mean_offset

    if DIM == 1:
        # no rotation is needed
        pass
    else:
        print("rotation for DIM {} not supported".format(DIM))

    func_config = torch.empty(SAMPLE, DIM+1)
    func_amplitude_config = torch.empty(DIM+1).uniform_(-20,20)
    func_frequency_config = 1/SAMPLE * torch.pow(10, torch.empty(DIM+1).uniform_(1,2))
    for i in range(DIM+1):
        func_config[:, i] = torch.cos(func_frequency_config[i] * torch.arange(SAMPLE)) * \
                            func_amplitude_config[i]

    gaussian_label = torch.sum(gaussian_sample * func_config[:, :DIM], 1) + func_config[:, DIM]

if STORE_GAUSSIAN:
    save_dict = {"sample":gaussian_sample, "label":gaussian_label, "mean_offset":mean_offset, "mean_variance":mean_variance, "func_config":func_config, }
    torch.save(save_dict, os.path.join(gen_dir, "test_data.pt"))

if not SKIP_PNG:
    def map_id_cmap(min, id):
        ret = torch.where(id < min, 0.2, 0.2+(id-min)/100*0.8)
        return ret

    for i in range(0, SAMPLE, 10):
        plt.xlim((-3,3))
        plt.ylim((-50,50))
        colors = cm.Greys(map_id_cmap(i-100, torch.arange(i)))
        plt.scatter(gaussian_sample[:i], gaussian_label[:i], color=colors)
        plt.axline((0, func_config[i, -1]), slope=func_config[i, 0], color="red")
        gaussian_range = plt.Rectangle((mean_offset[i, 0]-mean_variance[i,0], -50), 2*mean_variance[i,0], 5, fc="blue", alpha=0.5)
        plt.gca().add_patch(gaussian_range)
        plt.savefig(os.path.join(gen_dir, "png", "example_"+"{}".format(i).zfill(6)) + ".png")
        plt.cla()

if GEN_VIDEO:
    subprocess.call([
        'ffmpeg', '-i', '{}/png/example_%05d0.png'.format(gen_dir), '-r', '4',
        '{}/test_data.mp4'.format(gen_dir)
    ])

