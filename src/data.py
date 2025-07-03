import re
import os
import copy
# from functools import cache
import numpy as np # type: ignore
import cv2 # type: ignore
import pandas as pd #type: ignore

import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
# from torchvision.io import read_image # type: ignore
from torchvision.transforms import v2 # type: ignore
from torchvision.io import decode_jpeg, write_jpeg

class GlaucomaDataset(Dataset):
    """
    This is the class that loads and holds the data for training.
    
    This is a Pytorch dataset class that is used by the HuggingFace
    Trainer. It has the required methods of __init__, __len__ and 
    __get_item__ which initializes the class, fetches the length of
    the dataset and gets a specific item.
    
    It also includes the split, oversample, save, and get_pos_weight
    methods to perform specific operations on the dataset such as
    oversampling."""
    def __init__(self, csv_path, img_dir, hyperparams):
        """
        This method initializes the dataset and creates
        a pandas Dataframe containing the paths to each image
        and its label.
        """
        self.resize = hyperparams["RESIZE"]
        self.yolo_size = hyperparams["YOLO-SIZE"]
        self.yolo_path = hyperparams["YOLO-PATH"]
        self.target_size = hyperparams["TARGET-SIZE"]
        self.save_path = hyperparams.get("SAVE-PATH") # None if key is not in dict
        seed = hyperparams["SEED"]

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.is_oversample = False
        self.type = "" # not sure what default value should be

        columns = ["Eye ID", "Final Label"]
        df = pd.read_csv(csv_path, sep=';')[columns] # drop all unspecified columns
        df["path"] = df["Eye ID"] \
            .map(lambda id: os.path.join(img_dir, str(self._get_folder(id)), f"{id}.JPG"))
        df = df[df["path"].map(os.path.exists)] # filter if image exists
        
        df["label"] = (df["Final Label"] != "NRG").astype(dtype = np.int32)
        df = df.drop(columns, axis=1)
        
        self.df = df
    
    def __getitem__(self, idx):
        """
        This loads an image from the disk, performs preprocessing
        and data augmentation if necessary. Two rounds of data augmentaion
        are performed, one on the positive class for oversampling, and
        one on every image.
        """
        # img = self._preprocess(self.df["path"][idx])
        # img = self._cache(img, idx)
        orig_path = self.df["path"][idx]
        if self.save_path and os.path.exists(self.save_path):
            filename = os.path.basename(orig_path)
            path = os.path.join(self.save_path, self.type, filename)
            img = decode_jpeg(path)
        else:
            # TODO verify that images being on the gpu doesn't
            # mess up DataLoader
            img = self._preprocess(orig_path)

        if self.is_oversample and self.df["oversample"][idx]:
            img = self._oversample(img)

        # hyperparams from https://arxiv.org/pdf/2106.10270 med2
        img = v2.RandAugment(num_ops = 2, magnitude = 15)(img)

        img = v2.functional.to_dtype(img, torch.float32)
        # mean and std are from ImageNet
        img = v2.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        label = self.df["label"][idx]
        output =  {
            "pixel_values": img,
            "label": label
            # "interpolate_pos_encoding": True # Allows 224x224 encoding layer to be converted to 512x512
        }

        return output

    def __len__(self):
        """
        This method return the length when using
        the len function.
        """
        return len(self.df)

    # def load(self):
    #     for index in self.df.index:
    #         print(index)
    #         _ = self[index]

    def _get_folder(self, id):
        """
        This is a helper function which gets the folder a
        given image resides in.
        """
        pattern = r"\d+"
        n = int(re.findall(pattern, id)[0])
        if n <= 17407:
            return 0
        if n <= 34815:
            return 1
        if n <= 52223:
            return 2
        if n <= 69631:
            return 3
        if n <= 87039:
            return 4
        else:
            return 5

    def oversample(self):
        """
        This method performs over sampling on the dataset
        such that the positive and negative classes
        are balanced. Augmentation for oversampling occurs
        when fetching the item.
        """
        # Select Images to oversample and add them to the dataframe
        # The augmentation is performed in the getitem method
        self.is_oversample = True
        df = self.df
        df["oversample"] = False
        pos = df[df["label"] == 1]
        n_to_sample = len(self.df) - 2 * len(pos) # n_neg - n_pos

        new_rows = pos.sample(n = n_to_sample, replace = True)
        new_rows["oversample"] = True

        self.df = pd.concat([df, new_rows], ignore_index=True)

    def _oversample(self, img):
        """
        This is a helper method that performs data augmentation
        on the specified oversample images.
        """
        # tf.keras.layers.RandomFlip(),
        # tf.keras.layers.RandomRotation(0.2),
        return v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(0.2)
        ])(img)
    
    def save(self, save_dir):
        """
        This method performs the preprocessing steps on
        all images and saves them to the disk. This is done to
        prevent preprocessing occuring over every epoch.
        """
        save_dir = os.path.join(save_dir, self.type)
        os.makedirs(save_dir, exist_ok=True)
        df = self.df[self.df["oversample"] == False] if "oversample" in self.df.columns else self.df
        for path in df["path"]:
            # TODO add multiprocessing
            self._encode(path, save_dir)
    
    def _run_yolo(self, img):
        """
        This is a helper function that runs yolo and gets
        the center of the resulting bounding box. This is
        then converted to the original 2,000 x 2,000 pixel
        image.
        """
        img = v2.functional.to_dtype(img, torch.float32, scale=True)
        img = v2.functional.resize(img, self.yolo_size)
        img = torch.unsqueeze(img, 0) # add a batch of 1 for YOLO
        yolo = torch.jit.load(self.yolo_path, map_location=torch.device("cuda"))
        with torch.no_grad():
            output = yolo(img)

        # TODO Can add Non-Maximum Suppression (NMS) when exporting from Ultralytics
        boxes = output[0] # assuming batch of 1
        max_conf_idx = torch.argmax(boxes[4])
        x, y, w, h, c = boxes[:, max_conf_idx]
        
        # resize to (2000, 2000)
        x = x * self.resize[0] / self.yolo_size[0]
        y = y * self.resize[1] / self.yolo_size[1]

        return (x, y)
    
    # def _load_img(self, image_path):
    #     if self.is_test:
    #         # Don't want to cache test images as they will only
    #         # be used once.
    #         # Can change later depending on implementation of
    #         # training
    #         png = self._encode.__wrapped__(image_path)
    #     else:
    #         png = self._encode(image_path)
    #     return self._decode(png)
    #     # return self._preprocess(image_path)
    
    def _preprocess(self, image_path):
        """
        This is the helper method that performs all pf
        the preprocessing including CLAHE and YOLO."""
        # Load and resize image
        img = cv2.imread(image_path)
        # self.size = (2_000, 2_000) # TODO add to params
        img = cv2.resize(img, self.resize)
        # raise RuntimeError("Breakpoint")
        # return img

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        new_img = cv2.merge((l, a, b))
        img = cv2.cvtColor(new_img, cv2.COLOR_Lab2RGB)
        # return img

        img = v2.functional.to_image(img).to('cuda')
        # return img
        # Implemented in yolo method in order to perform RandAugment
        # img = v2.functional.to_dtype(img, torch.float32, scale=True)
        
        # YOLO
        x, y = self._run_yolo(img)
        w, h = self.target_size

        x = x - 0.5 * w
        y = y - 0.5 * h

        x, y, w, h = int(x), int(y), int(w), int(h)
        img = v2.functional.crop(img, y, x, h, w)
        # raise RuntimeError("Breakpoint")
        return img.cpu()
    
    def _encode(self, image_path, save_dir):
        """
        This method loads an image and encodes it as a JPEG.
        """
        # Make sure path exists, otherwise image is not saved
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, filename)

        img = self._preprocess(image_path)
        write_jpeg(img, save_path)

        # the following code might be faster some how?
        # would need to do a speed test
        # img = img.cpu().numpy().transpose((1, 2, 0))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(save_path, img)
    
    def get_pos_weight(self):
        """
        This method return the ratio of negative samples
        to positive samples. This is used to weight the loss function when training."""
        neg = self.df[self.df["label"] == 0]
        pos = self.df[self.df["label"] == 1]
        print(neg, pos)
        return len(neg) / len(pos)

    @staticmethod
    def split(ds, val_ratio, test_ratio):
        """
        This is a static method desined to split a given dataset of type
        GlaucomaDataset into train, val and test sets.
        """
        neg = ds.df[ds.df["label"] == 0]
        pos = ds.df[ds.df["label"] == 1]
        n_val = int(len(pos) * val_ratio)
        n_test = int(len(pos) * test_ratio)

        val_pos = pos.sample(n = n_val)
        pos = pos.drop(val_pos.index)
        val_neg = neg.sample(n = n_val)
        neg = neg.drop(val_neg.index)
        val_df = pd.concat([val_neg, val_pos], ignore_index=True)

        test_pos = pos.sample(n = n_test)
        pos = pos.drop(test_pos.index)
        test_neg = neg.sample(n = n_test)
        neg = neg.drop(test_neg.index)
        test_df = pd.concat([test_neg, test_pos], ignore_index=True)

        train_df = pd.concat([neg, pos], ignore_index=True)

        train = copy.copy(ds)
        train.df = train_df
        train.type = "train"
        val = copy.copy(ds)
        val.df = val_df
        val.type = "val"
        test = copy.copy(ds)
        test.df = test_df
        test.type = "test"

        return train, val, test