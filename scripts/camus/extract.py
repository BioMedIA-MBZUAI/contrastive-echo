import os
import torch
import torchvision
from skimage.transform import resize
from PIL import Image

from camus_dataset import Camus, ResizeImagesAndLabels

root = ""  # Path to data
save_path = "./../../data" # Path to save the extracted images
split = "train"
global_transforms = [ ResizeImagesAndLabels(size=[224, 224]) ]
augment_transforms = []
extract_all_images_between = False

os.makedirs(os.path.join(save_path, "camus", "segmentation", split), exist_ok=True)

segmentation_save_path = os.path.join(save_path, "camus", "segmentation", split)

ds = Camus(
    root=root,
    split=split,
    global_transforms=global_transforms,
    augment_transforms=augment_transforms,
)

param_Loader = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 4
}

data = torch.utils.data.DataLoader(ds, **param_Loader)

for i,d in enumerate(data):
    ed_4ch = int(d["info_4CH"]["ED"][0])
    es_4ch = int(d["info_4CH"]["ES"][0])

    im = d["4CH_ED"].cpu().numpy().squeeze()
    im = Image.fromarray(im).convert("RGB").resize((224, 224))
    im.save(os.path.join(segmentation_save_path, d["patient"][0] + "_" + str(ed_4ch) + "_0" + ".png"))

    im = d["4CH_ES"].cpu().numpy().squeeze()
    im = Image.fromarray(im).convert("RGB").resize((224, 224))
    im.save(os.path.join(segmentation_save_path, d["patient"][0] + "_" + str(es_4ch) + "_1" + ".png"))

    im = d["4CH_ED_gt"].cpu().numpy().squeeze()
    im = Image.fromarray(im).convert("L").resize((224, 224))
    im.save(os.path.join(segmentation_save_path, d["patient"][0] + "_" + str(ed_4ch) + "_0" + ".png"))

    im = d["4CH_ES_gt"].cpu().numpy().squeeze()
    im = Image.fromarray(im).convert("L").resize((224, 224))
    im.save(os.path.join(segmentation_save_path, d["patient"][0] + "_" + str(es_4ch) + "_1" + ".png"))

    # Extract all images in between sequence for contrastive pretraining
    if  extract_all_images_between:
        os.makedirs(os.path.join(save_path, "camus", "pretraining", split), exist_ok=True)

        if ed_4ch < es_4ch:
            _from, _to = ed_4ch, es_4ch
        else:
            _from, _to = es_4ch, ed_4ch

        for j in range(_from-1, _to):
            for j in range(_from, _to-1):
                im = d["4CH_sequence"].squeeze()[j, :, :].numpy()
                im = Image.fromarray(im).convert("RGB").resize((224, 224))
                im.save(os.path.join(save_path, "camus", "pretraining", split, d["patient"][0] + "_" + str(j) + ".png"))
