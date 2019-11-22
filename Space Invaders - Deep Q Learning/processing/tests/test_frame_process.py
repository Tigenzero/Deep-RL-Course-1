import nose
import numpy as np
import os
import retro
import imageio
from processing.frame_process import FramePreprocessor


RETRO_GAME = 'SpaceInvaders-Atari2600'
CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(CUR_FILE_PATH, "images")


def get_retro_frame():
    env = retro.make(game=RETRO_GAME)
    env.reset()
    env.step([1, 0, 0, 0, 0, 0, 0, 0])
    next_state, reward, done, _ = env.step([1, 0, 0, 0, 0, 0, 0, 0])
    print(next_state.shape)
    return next_state


def test_frameprocessor_init():
    frame_processor = FramePreprocessor(None, None, None, None, None, None, None)


def test_frameprocessor_convert_grayscale():
    frame = get_retro_frame()
    frame_processor = FramePreprocessor(None, None, None, None, None, None, None)
    grayscale_frame = frame_processor.convert_grayscale(frame)
    expected_shape = (frame.shape[0], frame.shape[1])
    print(grayscale_frame.shape)
    print(expected_shape)
    save_image(grayscale_frame, "test_grayscale_frame.png")
    assert expected_shape == grayscale_frame.shape


def test_frameprocessor_crop_frame():
    frame = get_retro_frame()
    frame_processor = FramePreprocessor(8, 12, 4, 12, None, None, None)
    cropped_frame = frame_processor.crop_frame(frame)
    # imageio.imsave("./processing/tests/images/test_cropped_frame.png", cropped_frame)
    print(cropped_frame.shape)
    assert cropped_frame.shape == (190, 144, 3)


def test_frameprocessor_normalize_frame():
    frame = get_retro_frame()
    frame_processor = FramePreprocessor(8, 12, 4, 12, 255.0, None, None)
    frame = frame_processor.convert_grayscale(frame)
    frame = frame_processor.crop_frame(frame)
    normalized_frame = frame_processor.normalize_frame(frame)
    frame_min = np.amin(normalized_frame)
    frame_max = np.amax(normalized_frame)
    assert frame_min == 0
    assert frame_max == 1
    save_image(normalized_frame, "test_normalized_frame.png")


def test_frameprocessor_resize_frame():
    frame = get_retro_frame()
    resize_width = 110
    resized_height = 84
    frame_processor = FramePreprocessor(None, None, None, None, None, resize_width, resized_height)
    resized_frame = frame_processor.resize_frame(frame)
    print(resized_frame.shape)
    assert resized_frame.shape == (resize_width, resized_height, 3)


def test_frameprocessor_preprocess_frame():
    frame = get_retro_frame()
    frame_processor = FramePreprocessor(8, 12, 4, 12, 255.0, 110, 84)
    processed_frame = frame_processor.preprocess_frame(frame)
    save_image(processed_frame, "test_processed_frame.png")
    frame_min = np.amin(processed_frame)
    frame_max = np.amax(processed_frame)
    print("min: {}, max: {}".format(frame_min, frame_max))
    assert frame_min == 0
    assert frame_max == 1
    assert processed_frame.shape == (110, 84)


def save_image(frame, filename):
    if not os.path.exists(IMAGE_PATH):
        os.mkdir(IMAGE_PATH)
    frame_path = os.path.join(IMAGE_PATH, filename)
    print(frame_path)
    imageio.imsave(frame_path, frame)
