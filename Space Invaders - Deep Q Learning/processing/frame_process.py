from skimage.color import rgb2gray
from skimage import transform
import logging


class FramePreprocessor(object):
    def __init__(self, top_crop=0, bottom_crop=0, left_crop=0, right_crop=0, normalize=1, resize_width=0, resize_height=0):
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop
        self.left_crop = left_crop
        self.right_crop = right_crop
        self.normalize = normalize
        self.resize_width = resize_width
        self.resize_height = resize_height

    def convert_greyscale(self, frame):
        return rgb2gray(frame)

    def crop_frame(self, frame):
        return frame[self.top_crop: frame.height - self.bottom_crop, self.left_crop: frame.width - self.right_crop]

    def normalize_frame(self, frame):
        if self.normalize > 0:
            return frame/255.0
        else:
            return frame

    def resize_frame(self, frame):
        if self.resize_height > 0 and self.resize_width > 0:
            return transform.resize(frame, [self.resize_width, self.resize_height])
        else:
            logging.warning("unable to resize, resize height or width not set")

    def preprocess_frame(self, frame):
        gray_frame = self.convert_greyscale(frame)
        cropped_frame = self.crop_frame(gray_frame)
        normalized_frame = self.normalize_frame(cropped_frame)
        return self.resize_frame(normalized_frame)

    @classmethod
    def preprocess_frame_class(cls, env_object, frame):
        processor = cls(env_object.top_crop,
                        env_object.bottom_crop,
                        env_object.left_crop,
                        env_object.right_crop,
                        env_object.normalize,
                        env_object.resize_width,
                        env_object.resize_height)

        return processor.preprocess_frame(frame)

    @classmethod
    def create_class_with_param_object(cls, processing_params):
        return cls(top_crop=processing_params.top_crop,
                   bottom_crop=processing_params.bottom_crop,
                   left_crop=processing_params.left_crop,
                   right_crop=processing_params.right_crop,
                   normalize=processing_params.normalize,
                   resize_width=processing_params.resize_width,
                   resize_height=processing_params.resize_height)