
kernel_mode = False

from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import cv2
import gc

import sys
if kernel_mode:
    sys.path.insert(0, "../input/hpa-cell-segmentation")

from hpacellseg.cellsegmentator import *

target_image_size = 512

IMAGE_SIZES = [1728, 2048, 3072, 4096]
if kernel_mode:
    dataset_folder = "/kaggle/input/hpa-single-cell-image-classification"
    # img_dir = f"{dataset_folder}/test"
    img_dir = "/kaggle/working/test_resized"
    output_folder = "/kaggle/working/test_cell_masks"
    NUC_MODEL = "/kaggle/input/hpa-cell-segmentation/dpn_unet_nuclei_v1.pth"
    CELL_MODEL = "/kaggle/input/hpa-cell-segmentation/dpn_unet_cell_3ch_v1.pth"
    BATCH_SIZE = {1728: 24, 2048: 22, 3072: 12, 4096: 12}
else:
    dataset_folder = "/workspace/Kaggle/HPA/hpa_2020"
    # img_dir = f"{dataset_folder}/test"
    img_dir = f"{dataset_folder}/test_resized"
    output_folder = f"{dataset_folder}/test_cell_masks"
    NUC_MODEL = "/workspace/Github/HPA-Cell-Segmentation/dpn_unet_nuclei_v1.pth"
    CELL_MODEL = "/workspace/Github/HPA-Cell-Segmentation/dpn_unet_cell_3ch_v1.pth"
    #     BATCH_SIZE = {1728: 24, 2048: 24, 3072: 12, 4096: 12}
    BATCH_SIZE = {1728: 20, 2048: 18, 3072: 8, 4096: 8}

os.makedirs(output_folder, exist_ok=True)

submit_df = pd.read_csv(f"{dataset_folder}/sample_submission.csv")

predict_df_1728 = submit_df[submit_df.ImageWidth == IMAGE_SIZES[0]]
predict_df_2048 = submit_df[submit_df.ImageWidth == IMAGE_SIZES[1]]
predict_df_3072 = submit_df[submit_df.ImageWidth == IMAGE_SIZES[2]]
predict_df_4096 = submit_df[submit_df.ImageWidth == IMAGE_SIZES[3]]

predict_ids_1728 = predict_df_1728.ID.to_list()
predict_ids_2048 = predict_df_2048.ID.to_list()
predict_ids_3072 = predict_df_3072.ID.to_list()
predict_ids_4096 = predict_df_4096.ID.to_list()


class CellSegmentator(object):
    """Uses pretrained DPN-Unet models to segment cells from images."""
    def __init__(
        self,
        nuclei_model="../input/hpacellsegmentatormodelweights/dpn_unet_nuclei_v1.pth",
        cell_model="../input/hpacellsegmentatormodelweights/dpn_unet_cell_3ch_v1.pth",
        scale_factor=1.0,
        device="cuda",
        padding=False,
        multi_channel_model=True,
    ):
        """Class for segmenting nuclei and whole cells from confocal microscopy images.
        It takes lists of images and returns the raw output from the
        specified segmentation model. Models can be automatically
        downloaded if they are not already available on the system.
        When working with images from the Huan Protein Cell atlas, the
        outputs from this class' methods are well combined with the
        label functions in the utils module.
        Note that for cell segmentation, there are two possible models
        available. One that works with 2 channeled images and one that
        takes 3 channels.
        Keyword arguments:
        nuclei_model -- A loaded torch nuclei segmentation model or the
                        path to a file which contains such a model.
                        If the argument is a path that points to a non-existant file,
                        a pretrained nuclei_model is going to get downloaded to the
                        specified path (default: './nuclei_model.pth').
        cell_model -- A loaded torch cell segmentation model or the
                      path to a file which contains such a model.
                      The cell_model argument can be None if only nuclei
                      are to be segmented (default: './cell_model.pth').
        scale_factor -- How much to scale images before they are fed to
                        segmentation models. Segmentations will be scaled back
                        up by 1/scale_factor to match the original image
                        (default: 0.25).
        device -- The device on which to run the models.
                  This should either be 'cpu' or 'cuda' or pointed cuda
                  device like 'cuda:0' (default: 'cuda').
        padding -- Whether to add padding to the images before feeding the
                   images to the network. (default: False).
        multi_channel_model -- Control whether to use the 3-channel cell model or not.
                               If True, use the 3-channel model, otherwise use the
                               2-channel version (default: True).
        """
        if device != "cuda" and device != "cpu" and "cuda" not in device:
            raise ValueError(f"{device} is not a valid device (cuda/cpu)")
        if device != "cpu":
            try:
                assert torch.cuda.is_available()
            except AssertionError:
                print("No GPU found, using CPU.", file=sys.stderr)
                device = "cpu"
        self.device = device

        if isinstance(nuclei_model, str):
            if not os.path.exists(nuclei_model):
                print(
                    f"Could not find {nuclei_model}. Downloading it now",
                    file=sys.stderr,
                )
                download_with_url(NUCLEI_MODEL_URL, nuclei_model)
            nuclei_model = torch.load(nuclei_model,
                                      map_location=torch.device(self.device))
        if isinstance(nuclei_model, torch.nn.DataParallel) and device == "cpu":
            nuclei_model = nuclei_model.module

        self.nuclei_model = nuclei_model.to(self.device).eval()

        self.multi_channel_model = multi_channel_model
        if isinstance(cell_model, str):
            if not os.path.exists(cell_model):
                print(f"Could not find {cell_model}. Downloading it now",
                      file=sys.stderr)
                if self.multi_channel_model:
                    download_with_url(MULTI_CHANNEL_CELL_MODEL_URL, cell_model)
                else:
                    download_with_url(TWO_CHANNEL_CELL_MODEL_URL, cell_model)
            cell_model = torch.load(cell_model,
                                    map_location=torch.device(self.device))
        self.cell_model = cell_model.to(self.device).eval()
        self.scale_factor = scale_factor
        self.padding = padding

    def _image_conversion(self, images):
        """Convert/Format images to RGB image arrays list for cell predictions.
        Intended for internal use only.
        Keyword arguments:
        images -- list of lists of image paths/arrays. It should following the
                 pattern if with er channel input,
                 [
                     [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                     [er_path0/image_array0, er_path1/image_array1, ...],
                     [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                 ]
                 or if without er input,
                 [
                     [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                     None,
                     [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                 ]
        """
        microtubule_imgs, er_imgs, nuclei_imgs = images
        if self.multi_channel_model:
            if not isinstance(er_imgs, list):
                raise ValueError(
                    "Please speicify the image path(s) for er channels!")
        else:
            if not er_imgs is None:
                raise ValueError(
                    "second channel should be None for two channel model predition!"
                )

        if not isinstance(microtubule_imgs, list):
            raise ValueError("The microtubule images should be a list")
        if not isinstance(nuclei_imgs, list):
            raise ValueError("The microtubule images should be a list")

        if er_imgs:
            if not len(microtubule_imgs) == len(er_imgs) == len(nuclei_imgs):
                raise ValueError(
                    "The lists of images needs to be the same length")
        else:
            if not len(microtubule_imgs) == len(nuclei_imgs):
                raise ValueError(
                    "The lists of images needs to be the same length")

        if not all(isinstance(item, np.ndarray) for item in microtubule_imgs):
            microtubule_imgs = [
                os.path.expanduser(item)
                for _, item in enumerate(microtubule_imgs)
            ]
            nuclei_imgs = [
                os.path.expanduser(item) for _, item in enumerate(nuclei_imgs)
            ]

            microtubule_imgs = list(
                map(lambda item: imageio.imread(item), microtubule_imgs))
            nuclei_imgs = list(
                map(lambda item: imageio.imread(item), nuclei_imgs))
            if er_imgs:
                er_imgs = [
                    os.path.expanduser(item) for _, item in enumerate(er_imgs)
                ]
                er_imgs = list(map(lambda item: imageio.imread(item), er_imgs))

        if not er_imgs:
            er_imgs = [
                np.zeros(item.shape, dtype=item.dtype)
                for _, item in enumerate(microtubule_imgs)
            ]
        cell_imgs = list(
            map(
                lambda item: np.dstack((item[0], item[1], item[2])),
                list(zip(microtubule_imgs, er_imgs, nuclei_imgs)),
            ))

        return cell_imgs

    def pred_nuclei(self, images, bs=24):
        """Predict the nuclei segmentation.
        Keyword arguments:
        images -- A list of image arrays or a list of paths to images.
                  If as a list of image arrays, the images could be 2d images
                  of nuclei data array only, or must have the nuclei data in
                  the blue channel; If as a list of file paths, the images
                  could be RGB image files or gray scale nuclei image file
                  paths.
        Returns:
        predictions -- A list of predictions of nuclei segmentation for each nuclei image.
        """
        def _preprocess(image):
            if isinstance(image, str):
                image = imageio.imread(image)
            self.target_shape = image.shape
            if len(image.shape) == 2:
                image = np.dstack((image, image, image))
            image = transform.rescale(image,
                                      self.scale_factor,
                                      multichannel=True)
            nuc_image = np.dstack((image[..., 2], image[..., 2], image[...,
                                                                       2]))
            if self.padding:
                rows, cols = nuc_image.shape[:2]
                self.scaled_shape = rows, cols
                nuc_image = cv2.copyMakeBorder(
                    nuc_image,
                    32,
                    (32 - rows % 32),
                    32,
                    (32 - cols % 32),
                    cv2.BORDER_REFLECT,
                )
            nuc_image = nuc_image.transpose([2, 0, 1])
            return nuc_image

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE["mean"], device=self.device)
                std = torch.as_tensor(NORMALIZE["std"], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.nuclei_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        preprocessed_imgs = list(map(_preprocess, images))
        predictions = []
        for i in range(0, len(preprocessed_imgs), bs):
            start = i
            end = min(len(preprocessed_imgs), i + bs)
            x = preprocessed_imgs[start:end]
            pred = _segment_helper(x).cpu().numpy()
            predictions.append(pred)
        predictions = list(np.concatenate(predictions, axis=0))
        predictions = map(util.img_as_ubyte, predictions)
        predictions = list(map(self._restore_scaling_padding, predictions))
        return predictions

    def _restore_scaling_padding(self, n_prediction):
        """Restore an image from scaling and padding.
        This method is intended for internal use.
        It takes the output from the nuclei model as input.
        """
        n_prediction = n_prediction.transpose([1, 2, 0])
        if self.padding:
            n_prediction = n_prediction[32:32 + self.scaled_shape[0],
                                        32:32 + self.scaled_shape[1], ...]
        if not self.scale_factor == 1:
            n_prediction[..., 0] = 0
            n_prediction = cv2.resize(
                n_prediction,
                (self.target_shape[0], self.target_shape[1]),
                # interpolation=cv2.INTER_AREA,
                interpolation=cv2.INTER_NEAREST,
            )
        return n_prediction

    def pred_cells(self, images, precombined=False, bs=24):
        """Predict the cell segmentation for a list of images.
        Keyword arguments:
        images -- list of lists of image paths/arrays. It should following the
                  pattern if with er channel input,
                  [
                      [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                      [er_path0/image_array0, er_path1/image_array1, ...],
                      [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                  ]
                  or if without er input,
                  [
                      [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                      None,
                      [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                  ]
                  The ER channel is required when multichannel is True
                  and required to be None when multichannel is False.
                  The images needs to be of the same size.
        precombined -- If precombined is True, the list of images is instead supposed to be
                       a list of RGB numpy arrays (default: False).
        Returns:
        predictions -- a list of predictions of cell segmentations.
        """
        def _preprocess(image):
            self.target_shape = image.shape
            if not len(image.shape) == 3:
                raise ValueError("image should has 3 channels")
            cell_image = transform.rescale(image,
                                           self.scale_factor,
                                           multichannel=True)
            if self.padding:
                rows, cols = cell_image.shape[:2]
                self.scaled_shape = rows, cols
                cell_image = cv2.copyMakeBorder(
                    cell_image,
                    32,
                    (32 - rows % 32),
                    32,
                    (32 - cols % 32),
                    cv2.BORDER_REFLECT,
                )
            cell_image = cell_image.transpose([2, 0, 1])
            return cell_image

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE["mean"], device=self.device)
                std = torch.as_tensor(NORMALIZE["std"], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.cell_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        if not precombined:
            images = self._image_conversion(images)
        preprocessed_imgs = list(map(_preprocess, images))
        predictions = []
        for i in range(0, len(preprocessed_imgs), bs):
            start = i
            end = min(len(preprocessed_imgs), i + bs)
            x = preprocessed_imgs[start:end]
            pred = _segment_helper(x).cpu().numpy()
            predictions.append(pred)
        predictions = list(np.concatenate(predictions, axis=0))
        predictions = map(self._restore_scaling_padding, predictions)
        predictions = list(map(util.img_as_ubyte, predictions))

        return predictions


import os.path
import urllib
import zipfile

import numpy as np
import scipy.ndimage as ndi
from skimage import filters, measure, segmentation
from skimage.morphology import (binary_erosion, closing, disk,
                                remove_small_holes, remove_small_objects)

HIGH_THRESHOLD = 0.4
LOW_THRESHOLD = HIGH_THRESHOLD - 0.25


def download_with_url(url_string, file_path, unzip=False):
    """Download file with a link."""
    with urllib.request.urlopen(url_string) as response, open(
            file_path, "wb") as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(data)

    if unzip:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))


def __fill_holes(image):
    """Fill_holes for labelled image, with a unique number."""
    boundaries = segmentation.find_boundaries(image)
    image = np.multiply(image, np.invert(boundaries))
    image = ndi.binary_fill_holes(image > 0)
    image = ndi.label(image)[0]
    return image


def label_cell(nuclei_pred, cell_pred):
    """Label the cells and the nuclei.
    Keyword arguments:
    nuclei_pred -- a 3D numpy array of a prediction from a nuclei image.
    cell_pred -- a 3D numpy array of a prediction from a cell image.
    Returns:
    A tuple containing:
    nuclei-label -- A nuclei mask data array.
    cell-label  -- A cell mask data array.
    0's in the data arrays indicate background while a continous
    strech of a specific number indicates the area for a specific
    cell.
    The same value in cell mask and nuclei mask refers to the identical cell.
    NOTE: The nuclei labeling from this function will be sligthly
    different from the values in :func:`label_nuclei` as this version
    will use information from the cell-predictions to make better
    estimates.
    """
    def __wsh(
        mask_img,
        threshold,
        border_img,
        seeds,
        threshold_adjustment=0.35,
        small_object_size_cutoff=10,
    ):
        img_copy = np.copy(mask_img)
        m = seeds * border_img  # * dt
        img_copy[m <= threshold + threshold_adjustment] = 0
        img_copy[m > threshold + threshold_adjustment] = 1
        img_copy = img_copy.astype(np.bool)
        img_copy = remove_small_objects(
            img_copy, small_object_size_cutoff).astype(np.uint8)

        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        mask_img = mask_img.astype(np.bool)
        mask_img = remove_small_holes(mask_img, 63)
        mask_img = remove_small_objects(mask_img, 1).astype(np.uint8)
        markers = ndi.label(img_copy, output=np.uint32)[0]
        labeled_array = segmentation.watershed(mask_img,
                                               markers,
                                               mask=mask_img,
                                               watershed_line=True)
        return labeled_array

    nuclei_label = __wsh(
        nuclei_pred[..., 2] / 255.0,
        0.4,
        1 - (nuclei_pred[..., 1] + cell_pred[..., 1]) / 255.0 > 0.05,
        nuclei_pred[..., 2] / 255,
        threshold_adjustment=-0.25,
        small_object_size_cutoff=32,
    )

    # for hpa_image, to remove the small pseduo nuclei
    nuclei_label = remove_small_objects(nuclei_label, 157)
    nuclei_label = measure.label(nuclei_label)
    # this is to remove the cell borders' signal from cell mask.
    # could use np.logical_and with some revision, to replace this func.
    # Tuned for segmentation hpa images
    threshold_value = max(
        0.22,
        filters.threshold_otsu(cell_pred[..., 2] / 255) * 0.5)
    # exclude the green area first
    cell_region = np.multiply(
        cell_pred[..., 2] / 255 > threshold_value,
        np.invert(np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8)),
    )
    sk = np.asarray(cell_region, dtype=np.int8)
    distance = np.clip(cell_pred[..., 2], 255 * threshold_value, cell_pred[...,
                                                                           2])
    cell_label = segmentation.watershed(-distance, nuclei_label, mask=sk)
    cell_label = remove_small_objects(cell_label, 344).astype(np.uint8)
    selem = disk(2)
    cell_label = closing(cell_label, selem)
    cell_label = __fill_holes(cell_label)
    # this part is to use green channel, and extend cell label to green channel
    # benefit is to exclude cells clear on border but without nucleus
    sk = np.asarray(
        np.add(
            np.asarray(cell_label > 0, dtype=np.int8),
            np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8),
        ) > 0,
        dtype=np.int8,
    )
    cell_label = segmentation.watershed(-distance, cell_label, mask=sk)
    cell_label = __fill_holes(cell_label)
    cell_label = np.asarray(cell_label > 0, dtype=np.uint8)
    cell_label = measure.label(cell_label)
    cell_label = remove_small_objects(cell_label, 344)
    cell_label = measure.label(cell_label)
    cell_label = np.asarray(cell_label, dtype=np.uint16)
    nuclei_label = np.multiply(cell_label > 0, nuclei_label) > 0
    nuclei_label = measure.label(nuclei_label)
    nuclei_label = remove_small_objects(nuclei_label, 157)
    nuclei_label = np.multiply(cell_label, nuclei_label > 0)

    return nuclei_label, cell_label


segmentator = CellSegmentator(NUC_MODEL,
                              CELL_MODEL,
                              device="cuda",
                              padding=True)
# segmentator = CellSegmentator(
#     NUC_MODEL,
#     CELL_MODEL,
#     scale_factor=0.25,
#     device="cuda",
#     padding=True,
#     multi_channel_model=True,
# )


def get_segment_mask(gray, rgb, bs=24):
    nuc_segmentations = segmentator.pred_nuclei(gray, bs=bs)  # blue
    cell_segmentations = segmentator.pred_cells(rgb, precombined=True, bs=bs)
    batch_cell_masks = [
        label_cell(nuc_seg, cell_seg)[1].astype(np.uint8)
        for nuc_seg, cell_seg in zip(nuc_segmentations, cell_segmentations)
    ]
    return batch_cell_masks


def load_images(image_ids, root=img_dir):
    gray = []
    rgb = []
    for ID in tqdm(image_ids, total=len(image_ids)):
        r = os.path.join(root, f'{ID}_red.png')
        y = os.path.join(root, f'{ID}_yellow.png')
        b = os.path.join(root, f'{ID}_blue.png')
        r = cv2.imread(r, 0)
        y = cv2.imread(y, 0)
        b = cv2.imread(b, 0)
        #         gray_image = cv2.resize(b, (target_image_size, target_image_size))
        #         rgb_image = cv2.resize(np.stack((r, y, b), axis=2),
        #                                (target_image_size, target_image_size))
        gray_image = b
        rgb_image = np.stack((r, y, b), axis=2)
        gray.append(gray_image)
        rgb.append(rgb_image)
    return gray, rgb


for size_idx, submission_ids in tqdm(enumerate(
    [predict_ids_1728, predict_ids_2048, predict_ids_3072, predict_ids_4096]),
                                     total=4):
    size = IMAGE_SIZES[size_idx]
    if submission_ids == []:
        print(f"\n...SKIPPING SIZE {size} AS THERE ARE NO IMAGE IDS ...\n")
        continue
    else:
        print(f"\n...WORKING ON IMAGE IDS FOR SIZE {size} ...\n")
    for i in tqdm(range(0, len(submission_ids), BATCH_SIZE[size]),
                  total=int(np.ceil(len(submission_ids) / BATCH_SIZE[size]))):

        r, y, b = [], [], []
        image_ids = submission_ids[i:(i + BATCH_SIZE[size])]
        gray, rgb = load_images(image_ids)
        batch_cell_masks = get_segment_mask(gray, rgb, bs=BATCH_SIZE[size])

        torch.cuda.empty_cache()
        gc.collect()

        for index, img_id in enumerate(image_ids):
            np.save(os.path.join(output_folder, f'{img_id}_cell_mask.npy'),
                    batch_cell_masks[index])