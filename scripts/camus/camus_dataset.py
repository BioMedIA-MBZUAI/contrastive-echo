import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

import SimpleITK as sitk
from skimage.transform import resize

class Camus(Dataset):
    def __init__(
        self,
        root = None,
        split = 'train',
        global_transforms = [],
        augment_transforms = []
    ):
        super(Camus, self).__init__()

        if root is None:
            raise Exception('No root directory!!')

        if split == 'train' or split == "val":
            data_dir = os.path.join(root, "training")
        elif split == 'test':
            data_dir = os.path.join(root, "testing")
        else:
            raise Exception('Wrong split for CamusIterator')

        self.split = split
        self.data_dir = data_dir
        self.global_transforms = global_transforms
        self.augment_transforms = augment_transforms
        self.patients = []

        for patient in os.listdir(self.data_dir):
            if len(os.listdir(os.path.join(self.data_dir, patient)) ) > 0:
                self.patients.append(patient)
            else:
                print("Empty patient folder: " + str(patient))

        if self.split == "train":
            self.patients = self.patients[0:400]
        elif self.split == "val":
            self.patients = self.patients[400:450]

        print("Dataset " + split + " :" + str(len(self.patients)))


    def __read_image(self, patient_file, suffix):
        image_file = '{}/{}/{}'.format(self.data_dir, patient_file, patient_file + suffix)
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_file, sitk.sitkFloat32))
        return image

    def __read_info(self, data_dir):
        info = {}
        with open(data_dir, 'r') as f:
            for line in f.readlines():
                info_type, info_details = line.strip( '\n' ).split(': ')
                info[info_type] = info_details
        return info

    def __len__( self ):
        return len(self.patients)

    def __getitem__( self, index ):
        patient_file = self.patients[index]

        image_2CH_ED = self.__read_image(patient_file, '_2CH_ED.mhd')
        image_2CH_ES = self.__read_image(patient_file, '_2CH_ES.mhd')
        image_4CH_ED = self.__read_image(patient_file, '_4CH_ED.mhd')
        image_4CH_ES = self.__read_image(patient_file, '_4CH_ES.mhd')
        image_2CH_sequence = self.__read_image(patient_file, '_2CH_sequence.mhd')
        image_4CH_sequence = self.__read_image(patient_file, '_4CH_sequence.mhd')

        if self.split== 'train' or self.split == "val":
            image_2CH_ED_gt = self.__read_image(patient_file, '_2CH_ED_gt.mhd')
            image_2CH_ES_gt = self.__read_image(patient_file, '_2CH_ES_gt.mhd')
            image_4CH_ED_gt = self.__read_image(patient_file, '_4CH_ED_gt.mhd')
            image_4CH_ES_gt = self.__read_image(patient_file, '_4CH_ES_gt.mhd')

        info_2CH = self.__read_info('{}/{}/{}'.format(self.data_dir, patient_file, 'Info_2CH.cfg'))
        info_4CH = self.__read_info('{}/{}/{}'.format(self.data_dir, patient_file, 'Info_4CH.cfg'))

        if self.split == 'train' or self.split == "val":
            data = {
                'patient': patient_file,
                '2CH_ED': image_2CH_ED.astype(np.uint8),
                '2CH_ES': image_2CH_ES.astype(np.uint8),
                '4CH_ED': image_4CH_ED.astype(np.uint8),
                '4CH_ES': image_4CH_ES.astype(np.uint8),
                '2CH_sequence': image_2CH_sequence,
                '4CH_sequence': image_4CH_sequence,
                '2CH_ED_gt': image_2CH_ED_gt.astype(np.uint8),
                '2CH_ES_gt': image_2CH_ES_gt.astype(np.uint8),
                '4CH_ED_gt': image_4CH_ED_gt.astype(np.uint8),
                '4CH_ES_gt': image_4CH_ES_gt.astype(np.uint8),
                'info_2CH': info_2CH,    # Dictionary of infos
                'info_4CH': info_4CH}    # Dictionary of infos
        elif self.split == 'test':
            data = {
                'patient': patient_file,
                '2CH_ED': image_2CH_ED,
                '2CH_ES': image_2CH_ES,
                '4CH_ED': image_4CH_ED,
                '4CH_ES': image_4CH_ES,
                '2CH_sequence': image_2CH_sequence,
                '4CH_sequence': image_4CH_sequence,
                'info_2CH': info_2CH,   # Dictionary of infos
                'info_4CH': info_4CH}   # Dictionary of infos

        fields=['2CH_ED_gt', '2CH_ES_gt', '4CH_ED_gt', '4CH_ES_gt']

        for f in fields:
            if f in data.keys():
                new = np.zeros(data[f].shape)
                new[data[f]==1] = 1
                data[f] = new

        return data

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class ResizeImagesAndLabels(object):
    '''
    Ripped out of Prof. Stough's code
    '''

    def __init__(self, size, fields=['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES',
                                     '2CH_ED_gt', '2CH_ES_gt', '4CH_ED_gt', '4CH_ES_gt']):
        self.size = size
        self.fields = fields

    def __call__(self, data):
        print(data)
        for field in self.fields:
            # transpose to go from chan x h x w to h x w x chan and back.
            data[field] = resize(data[field].transpose([1,2,0]),
                                 self.size, mode='constant',
                                 anti_aliasing=True)
            data[field] = data[field].transpose([2,0,1])

        return data
