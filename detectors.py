from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from transformers import pipeline
import skimage
import numpy as np
import glob
import os
# add the grounding dino to the path
import sys
import matplotlib.pyplot as plt
from torchvision.ops import box_convert
import torch
from PIL import Image
import imutils
import cv2
import pandas as pd
from google.cloud import storage
import torch


checkpoint = "google/owlvit-base-patch32"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")


class BabyDetector:
    def __init__(self, model_path=r'/Users/matanb/Downloads/weights/best.pt', conf = 0.001):
        self.model = YOLO(model_path)
        self.conf = conf
        if model_path[-4:] == 'onnx':
            self.onnx = True
        else:
            self.onnx = False

    def __call__(self, frame, policy='smallest',rotate = False):
        # run inference without printing
        if not self.onnx:
            result = self.model(frame, verbose=False)[0]
            if len(result) == 0:
                # return none with a message of no detection
                return None

            sz = 10000000
            smallest = None
            # tensor to list
            for r in result:
                res = r.boxes.data.tolist()
                raw_roi = res
                roi = [int(x) for x in res[0]]
                roi = [roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]]
                if policy == 'smallest':
                    if roi[2] * roi[3] < sz:
                        sz = roi[2] * roi[3]
                        smallest = roi

            self.roi = smallest
        else:
            # if onnx
            roi = self.model.predict(frame, conf=self.conf, imgsz=(320, 320), verbose = False)
            try:
                conf = roi[0].boxes.conf.numpy()[0]
            except:
                conf = 0
            roi = roi[0].boxes.xyxy.data.tolist()
            if len(roi) == 0:
                roi = None
            else:
                roi = [int(x) for x in roi[0]]

            self.roi = roi
            return roi, conf

    def show(self, frame):
        cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[0]+self.roi[2], self.roi[1]+self.roi[3]), (0, 0, 255), 2)
        plt.imshow(frame)
        plt.show()




class ModelInferencer:
    def __init__(self, model_path=None, drop_confidence_threshold=False, predict_on_gray_image=False,
                 batch_predict=False, logger=None):
        if model_path is None:
            self.model_path = "baby_detector/yolov8_320320_int8_recall_0.87_map_0.94.onnx"
            # self.model_path = "baby_detector/yolov8_with_beta_320320_int8_recall_0.87_map_0.94.onnx"
        else:
            self.model_path = model_path
        """if 1:  # for torch ultralytics inference
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model2 = YOLO(self.model_path, task='detect')
        """
        if not os.path.exists(self.model_path):
            self.fetch_model_from_gcloud()

        if batch_predict:
            self.net = YOLO(self.model_path, task='detect')
        else:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)

        self.INPUT_WIDTH = 320
        self.INPUT_HEIGHT = 320
        self.SCORE_THRESHOLD = 0.01 if drop_confidence_threshold else 0.4
        self.NMS_THRESHOLD = 0.3
        self.USE_NMS = True
        self.predict_on_gray_image = predict_on_gray_image
        self.logger = logger

        self.display_rotated_image = False
        self.next_ax = 0
        self.axs = None

    def fetch_model_from_gcloud(self):
        if self.logger is not None:
            self.logger.info("Fetching baby detector model from gcloud")
        else:
            print("Fetching baby detector model from gcloud")
        project_name = 'ml-workspace-2'
        bucket_name = 'algo-misc'

        bucket = storage.Client(project=project_name).get_bucket(bucket_name)
        blob = bucket.blob(os.path.basename(self.model_path))
        blob.download_to_filename(self.model_path)

        if not os.path.exists(self.model_path):
            raise Exception(f"Model not found in {self.model_path}")

    def predict(self, image):

        if self.predict_on_gray_image:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        square_image = self.letter_to_square(image)
        blob = cv2.dnn.blobFromImage(square_image, 1 / 255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        preds = preds.transpose((0, 2, 1))

        confs, boxes = [], []
        image_height, image_width, _ = square_image.shape
        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT
        rows = preds[0].shape[0]
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]
            if conf > self.SCORE_THRESHOLD:
                confs.append(conf)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = max(int((x - 0.5 * w) * x_factor), 0)
                top = max(int((y - 0.5 * h) * y_factor), 0)
                width = min(int(w * x_factor), image.shape[1])
                height = min(int(h * y_factor), image.shape[0])
                box = np.array([left, top, left + width, top + height])
                boxes.append(box)

        if self.USE_NMS:
            idxs = cv2.dnn.NMSBoxes(boxes, confs, self.SCORE_THRESHOLD, self.NMS_THRESHOLD)

            confidence_list = [c for i, c in enumerate(confs) if i in idxs]
            box_list = [b for i, b in enumerate(boxes) if i in idxs]
        else:
            confidence_list, box_list = confs, boxes

        confidence_list = confidence_list if len(confidence_list) > 0 else None
        box_list = box_list if len(box_list) > 0 else None

        if confidence_list is not None:
            # sort by confidence
            try:
                confidence_list, box_list = zip(*sorted(zip(confidence_list, box_list), reverse=True))
            except (TypeError, ValueError) as e:
                if self.logger is not None:
                    self.logger.info(f"An error occurred: {e}")
                else:
                    print(f"An error occurred: {e}")
                confidence_list, box_list = None, None

        if self.display_rotated_image:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            if self.next_ax == 0:
                fig, axs = plt.subplots(2, 4, figsize=(15, 9))
                fig.suptitle('Frame inferences, YOLO nano model')
                self.axs = axs.flatten()
            colors_rgb = [tuple(mcolors.hsv_to_rgb(c) * 255) for c in [(h, 1, 1) for h in np.arange(0, 1, 0.125)]]
            frame_to_plot = np.ascontiguousarray(np.array(blob[0].transpose((1, 2, 0)) * 255, dtype=np.uint8))
            row = preds[0][np.argmax(preds[0][:, 4])]
            if row[4] > 0:
                cv2.rectangle(frame_to_plot, (int(row[0] - row[2] / 2), int(row[1] - row[3] / 2)),
                              (int(row[0] + row[2] / 2), int(row[1] + row[3] / 2)), colors_rgb[self.next_ax], 2)
            self.axs[self.next_ax].imshow(frame_to_plot)
            self.axs[self.next_ax].set_title(f"conf {row[4]:.2f}")
            self.next_ax += 1
            if self.next_ax == 8:
                plt.tight_layout()
                plt.savefig("rotated_frame.png")
                plt.close()
                self.next_ax = 0

        return confidence_list, box_list

    def predict_on_batch(self, batch):

        if self.predict_on_gray_image:
            batch = (0.299 * batch[:, :, :, 0] + 0.587 * batch[:, :, :, 1] + 0.114 * batch[:, :, :, 2]).astype(np.uint8)
            batch = np.concatenate([np.expand_dims(batch, axis=-1)] * 3, axis=-1)

        blobs = []
        for image in batch:
            blob = cv2.dnn.blobFromImage(self.letter_to_square(image), 1 / 255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
            blobs.append(blob)
        resized_batch = torch.from_numpy(np.concatenate(blobs, axis=0))

        preds = self.net(resized_batch, conf=self.SCORE_THRESHOLD, iou=self.NMS_THRESHOLD,
                         imgsz=(self.INPUT_HEIGHT, self.INPUT_WIDTH), max_det=1, verbose=False)

        image_width = max(image.shape)
        x_factor = image_width / self.INPUT_WIDTH
        y_factor = x_factor
        confidence_list, box_list = [], []

        for i in range(len(preds)):
            if len(preds[i]) > 0:
                conf = preds[i].boxes[0].conf.item()
                if conf > self.SCORE_THRESHOLD:
                    confidence_list.append(conf)
                    pred_box = preds[i].boxes[0].xyxy[0].tolist()
                    left = max(int(pred_box[0] * x_factor), 0)
                    top = max(int(pred_box[1] * y_factor), 0)
                    right = min(int(pred_box[2] * x_factor), image.shape[1])
                    bottom = min(int(pred_box[3] * y_factor), image.shape[0])
                    box = [left, top, right, bottom]
                    box_list.append(box)
                else:
                    confidence_list.append(0)
                    box_list.append(None)
            else:
                confidence_list.append(0)
                box_list.append(None)

            if self.display_rotated_image:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                if self.next_ax == 0:
                    fig, axs = plt.subplots(2, 4, figsize=(15, 9))
                    fig.suptitle('Batch inferences, YOLO X model')
                    self.axs = axs.flatten()
                frame_to_plot = np.ascontiguousarray(np.array(blob[0].transpose((1, 2, 0)) * 255, dtype=np.uint8))
                colors_rgb = [tuple(mcolors.hsv_to_rgb(c) * 255) for c in [(h, 1, 1) for h in np.arange(0, 1, 0.125)]]
                if conf > 0:
                    cv2.rectangle(frame_to_plot, (int(pred_box[0]), int(pred_box[1])),
                                  (int(pred_box[2]), int(pred_box[3])), colors_rgb[self.next_ax], 2)
                self.axs[self.next_ax].imshow(frame_to_plot)
                self.axs[self.next_ax].set_title(f"conf {conf:.2f}")
                self.next_ax += 1
                if self.next_ax == 8:
                    plt.tight_layout()
                    plt.savefig("rotated_batch.png")
                    plt.close()
                    self.next_ax = 0

        return confidence_list, box_list

    def rotate_and_predict(self, image, flip=False, rotation=0):
        rotated_image = imutils.rotate_bound(image, rotation)
        rotated_image = cv2.flip(rotated_image, 1) if flip else rotated_image
        confidence_list, box_list = self.predict(rotated_image)
        if confidence_list is not None:
            for box in box_list:
                if flip:
                    box[0], box[2] = rotated_image.shape[1] - box[0] - 1, rotated_image.shape[1] - box[2] - 1
                if rotation == 90:
                    box[0], box[1], box[2], box[3] = box[1], image.shape[0] - box[0], box[3], image.shape[0] - \
                                                             box[2]
                elif rotation == 180:
                    box[0], box[1], box[2], box[3] = image.shape[1] - box[0], image.shape[0] - box[1], \
                                                     image.shape[1] - box[2], image.shape[0] - box[3]
                elif rotation == 270:
                    box[0], box[1], box[2], box[3] = image.shape[1] - box[1], box[0], image.shape[1] - box[3], \
                        box[2]

                if box[2] < box[0]:
                    box[0], box[2] = box[2], box[0]
                if box[3] < box[1]:
                    box[1], box[3] = box[3], box[1]

            return confidence_list, box_list
        return None, None

    def rotate_and_predict_batch(self, batch, flip=False, rotation=0):
        if rotation == 0:
            rotated_batch = batch
        elif rotation == 90:
            rotated_batch = np.flip(batch.swapaxes(1, 2), axis=2)
        elif rotation == 180:
            rotated_batch = np.flip(np.flip(batch, axis=1), axis=2)
        elif rotation == 270:
            rotated_batch = np.flip(batch.swapaxes(1, 2), axis=1)
        else:
            raise Exception(f"Invalid rotation: {rotation}")

        rotated_batch = rotated_batch[:, :, ::-1, :] if flip else rotated_batch

        confidence_list, box_list = self.predict_on_batch(rotated_batch)
        for i in range(len(confidence_list)):
            if confidence_list[i] > 0:
                box = box_list[i]
                if flip:
                    box[0], box[2] = rotated_batch.shape[2] - box[0] - 1, rotated_batch.shape[2] - box[2] - 1
                if rotation == 90:
                    box[0], box[1], box[2], box[3] = box[1], batch.shape[1] - box[0], box[3], batch.shape[1] - \
                                                             box[2]
                elif rotation == 180:
                    box[0], box[1], box[2], box[3] = batch.shape[2] - box[0], batch.shape[1] - box[1], \
                                                     batch.shape[2] - box[2], batch.shape[1] - box[3]
                elif rotation == 270:
                    box[0], box[1], box[2], box[3] = batch.shape[2] - box[1], box[0], batch.shape[2] - box[3], \
                        box[2]

                if box[2] < box[0]:
                    box[0], box[2] = box[2], box[0]
                if box[3] < box[1]:
                    box[1], box[3] = box[3], box[1]

        return confidence_list, box_list
        return None, None

    def force_predict(self, image):
        rotations = [0, 90, 180, 270]
        flipped = [False, True]
        for rotation in rotations:
            for flip in flipped:
                confidence_list, box_list = self.rotate_and_predict(image, flip, rotation)
                if confidence_list is not None:
                    return confidence_list, box_list, rotation + flip

        return None, None, None

    def force_predict_all_orientations(self, image):
        prediction_df = pd.DataFrame(columns=['rotation', 'flip', 'confidences', 'boxes'])
        rotations = [0, 90, 180, 270]
        flipped = [False, True]
        for rotation in rotations:
            for flip in flipped:
                confidence_list, box_list = self.rotate_and_predict(image, flip, rotation)
                new_col = pd.DataFrame([{'rotation': rotation, 'flip': flip, 'confidences': confidence_list,
                                         'boxes': self.repack_box_list(box_list)}])
                prediction_df = pd.concat([prediction_df, new_col], ignore_index=True)

        return prediction_df

    def force_predict_all_batch_orientations(self, batch, filter_best=True):
        data_frames = []
        rotations = [0, 90, 180, 270]
        flipped = [False, True]
        for rotation in rotations:
            for flip in flipped:
                confidence_list, box_list = self.rotate_and_predict_batch(batch, flip, rotation)
                for i in range(len(confidence_list)):
                    new_col = pd.DataFrame([{'frame_idx': i, 'rotation': rotation, 'flip': flip,
                                             'confidence': confidence_list[i],
                                             'box': box_list[i]}])
                    data_frames.append(new_col)

        prediction_df = pd.concat(data_frames, ignore_index=True)
        if filter_best:
            prediction_df = prediction_df.loc[prediction_df.groupby('frame_idx')['confidence'].idxmax()]
        return prediction_df

    def repack_box_list(self, box_list):
        if box_list is None:
            return None
        else:
            return tuple(arr.tolist() for arr in box_list)

    def letter_to_square(self, image):
        h, w = image.shape[:2]
        if h < w:
            pad = w - h
            return cv2.copyMakeBorder(image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        elif h == w:
            return image
        else:  # h > w
            pad = h - w
            return cv2.copyMakeBorder(image, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=0)

