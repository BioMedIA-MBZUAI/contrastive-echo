# Contrastive Pretraining for Echocardiography Segmentation with Limited Data

### Published in MIUA 2022 [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-12053-4_50)]

This repository contains a PyTorch implementation of our method for self-supervised pretraining using SimCLR and BYOL in to improve performance on the downstream task of ventricular segmentation in ultrasound images especially with limited data.

![image](https://github.com/rmuhtaseb/echo-miua/blob/main/figures/arch.png)

### Installation

Clone this repository and enter the directory:
```bash
git clone https://github.com/BioMedIA-MBZUAI/contrastive-echo.git
cd contrastive-echo
```

Make sure you have Python installed. The code is implemented for Python 3.8.10.

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

### Datasets

#### EchoNet-Dynamic
1. Download the dataset from [EchoNet-Dynamic website](https://echonet.github.io/dynamic/index.html#dataset)
2. Run the below script to extract the images from the videos:

```
cd scripts/echonet
python3 extract.py
```

#### CAMUS
1. Download the dataset from [CAMUS challenge website](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html)
2. Run the below script to extract the images as `PNGs`:

```
cd scripts/camus
python3 extract.py
```

### Usage

First, create a configuration file in `configs/experiment/`, following the samples in the same directory.

Then you can train a model by running the experiment as follows:

```
python3 run.py experiment=yourconfig.yaml
```

### Citation

```bash
@InProceedings{10.1007/978-3-031-12053-4_50,
  author="Saeed, Mohamed
  and Muhtaseb, Rand
  and Yaqub, Mohammad",
  editor="Yang, Guang
  and Aviles-Rivero, Angelica
  and Roberts, Michael
  and Sch{\"o}nlieb, Carola-Bibiane",
  title="Contrastive Pretraining for Echocardiography Segmentation with Limited Data",
  booktitle="Medical Image Understanding and Analysis",
  year="2022",
  publisher="Springer International Publishing",
  address="Cham",
  pages="680--691",
  isbn="978-3-031-12053-4"
}
```
