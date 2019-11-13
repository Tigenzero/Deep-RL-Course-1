import nose
import retro
import imageio
from processing.frame_process import FramePreprocessor


RETRO_GAME = 'SpaceInvaders-Atari2600'


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
    imageio.imsave("./images/test_grayscale_frame.png", grayscale_frame)
    assert expected_shape == grayscale_frame.shape


def test_frameprocessor_crop_frame():
    frame = get_retro_frame()
    frame_processor = FramePreprocessor(8, 12, 4, 12, None, None, None)
    cropped_frame = frame_processor.crop_frame(frame)
    # imageio.imsave("./processing/tests/images/test_cropped_frame.png", cropped_frame)
    print(cropped_frame.shape)
    assert cropped_frame.shape == (190, 144, 3)


# def test_frameprocessor_normalize_frame():
#     frame = get_retro_frame()
#     frame_processor = FramePreprocessor(None, None, None, None, 255.0, None, None)
#     frame = frame_processor.convert_grayscale(frame)
#     normalized_frame = frame_processor.normalize_frame(frame)
#     imageio.imsave("./images/test_normalized_frame.png", normalized_frame)


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
    imageio.imsave("./images/test_processed_frame.png", processed_frame)
    assert processed_frame.shape == (110, 84)
