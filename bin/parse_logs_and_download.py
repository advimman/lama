import os
import re
import pathlib

import yaml
import pandas as pd
from tqdm import tqdm

ROOT_PATH = "/mnt/storage/images"
CLUSTER_EXP_PATH = "/vol1/dbstore/orc_srr/multimodal/a.mashikhin/inpainting/experiments/"
folders = [
    # list of folders
]


def save_txt_metrics(folder, metrics_block, params):
    metrics_str = "Validation metrics after epoch #"+metrics_block[:-1] + f"\n\n Params: {round(params,1)}M"
    path = os.path.join(ROOT_PATH, folder, f"best_metrics.txt")
    text_file = open(path, "w")
    text_file.write(metrics_str)
    text_file.close()


def convert_log_to_metrics(file):
    df = []
    for i, metrics_block in enumerate(file.split("saicinpainting.training.trainers.base][INFO] - Validation metrics after epoch #")):
        if i == 0:
            continue
        ts = re.findall('\[(.*?)\]', metrics_block)[0]

        metrics_block_raw = metrics_block.split(ts)[0]
        epoch = metrics_block_raw.split(", total")[0]
        metrics_list = metrics_block_raw.split("\n")[-2].split()
        d = {"epoch": int(epoch),
             "fid_mean": float(metrics_list[1]),
             "fid_std": float(metrics_list[2]),
             "lpips_mean": float(metrics_list[3]),
             "lpips_std": float(metrics_list[4]),
             "ssim_mean": float(metrics_list[5]),
             "ssim_std": float(metrics_list[6]),
             "ssim_fid100_f1_mean": float(metrics_list[7]),
             "ssim_fid100_f1_std": float(metrics_list[8]),
            }
        df.append(d)
    df = pd.DataFrame(df).drop_duplicates("epoch")
    df = df.sort_values("ssim_fid100_f1_mean", ascending=False)
    return df, metrics_block_raw


def download_latest_logs(folder):
    output_path = os.path.join(ROOT_PATH, folder)
    pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)
    exp_folder = os.path.join(CLUSTER_EXP_PATH, folder)
    os.system(f"rsync -av --delete korea:{exp_folder}/train.log {output_path}")
    if not os.path.exists(os.path.join(output_path, "config.yaml")):
        os.system(f"rsync -av korea:{exp_folder}/config.yaml {output_path}")
        

def download_best_images(folder, epoch):
    print(f"Downloading sample for {folder}, epoch = {epoch}")
    output_path = os.path.join(ROOT_PATH, folder, "samples")
    pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)
    exp_folder = os.path.join(CLUSTER_EXP_PATH, folder, "samples", f"epoch{epoch:04d}_test")
    os.system(f"rsync -av korea:{exp_folder} {output_path}")


def main():
    result = {}
    for folder in tqdm(folders):
        exp_name = folder
        download_latest_logs(folder)
        path = os.path.join(ROOT_PATH, folder, "train.log")
        with open(path) as f:
            file = f.read()
        
        # metrics
        metrics_df, metrics_block = convert_log_to_metrics(file)
        best_metrics = metrics_df.iloc[0].to_dict()
        d = best_metrics
        
        # model
        params = file.split("generator      | ")[1].split("discriminator")[0].strip().replace("|", "").split()[1].strip()
        yaml_config_path = os.path.join(ROOT_PATH, folder, 'config.yaml')
        with open(yaml_config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        params = float(params)
        d.update({"ngf": config['generator']['ngf'],
                "n_blocks": config['generator']['n_blocks'],
                "kind": config['generator']['kind'],
                "dilation_num" : config['generator'].get("multidilation_kwargs", {}).get("dilation_num", 0),
                "params": params})
        result[exp_name] = d
        save_txt_metrics(folder, metrics_block, params)
        download_best_images(folder, int(best_metrics['epoch']))


def plot()
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.patches as mpatches
    %matplotlib inline
    plt.style.use(['default'])
    matplotlib.rcParams.update({'font.size': 22})

    x = []
    y = []
    plot_df = []
    for k,v in result.items():
        new_v = {"name": k}
        new_v.update(v)
        plot_df.append(new_v)
    plot_df = pd.DataFrame(plot_df)


    def get_label_and_color(kind, n_block, ngf, dilation_num):
        if  kind == "pix2pixhd_global":
            k=""
            color = "green"
        elif kind == "pix2pixhd_multidilated":
            k="_dil"
            if dilation_num == 2:
                post = "2"
                color = "red"
            elif dilation_num == 4:
                post = "4"
                color = "purple"
            else:
                raise NotImplementedError( f"dilation_num= {dilation_num}")
            k = f"{k}{post}"
        else:
            raise NotImplementedError(kind)
        label = f"b{n_block}_n{ngf}{k}"
        return label, color

    tmp_df = plot_df[plot_df['epoch'] > 0].sort_values("params")
    x = tmp_df['params'].tolist()
    y = tmp_df['ssim_fid100_f1_mean'].tolist()
    kind = tmp_df['kind'].tolist()
    n_block = tmp_df['n_blocks'].tolist()
    ngf = tmp_df['ngf'].tolist()
    dilation_num = tmp_df['dilation_num'].tolist()
    name = tmp_df['name'].tolist()

    f,ax = plt.subplots(1,1,figsize=(20,12))

    ax.scatter(x,y, color="black") 
    for i in range(len(tmp_df)):
        label, color = get_label_and_color(kind=kind[i], n_block=n_block[i], ngf=ngf[i], dilation_num=dilation_num[i])
        p = round(y[i]/0.9043 *100,1)
        ax.annotate(f"{label}({p}%)", (x[i], y[i]), fontsize=18, color=color)

    ax.set_xlabel("params (M)")
    ax.set_ylabel("ssim_fid100_f1")
    ax.set_title("Distillation of SOTA (ssim_fid100_f1 =0.9043)")
    patches = [mpatches.Patch(color='green', label='Normal convs'),
            mpatches.Patch(color='red', label='Dilated convs CATin (dil_num=2)'),
            mpatches.Patch(color='purple', label='Dilated convs CATin (dil_num=4). (all <= 6 epochs)'),]
    ax.legend(handles=patches, loc=4)
    ax.grid()
    plt.savefig("1.png", format='png', bbox_inches='tight')


if __name__ == "__main__":
    main()