"""Fixed RAW metadata and linear-sRGB rendering helpers, no learned processing."""
import os
os.environ['OMP_NUM_THREADS'] = '2'

import cv2
import exifread
import numpy as np
import rawpy


def encode_srgb(linear):
    linear = np.asarray(linear, dtype=np.float64)
    assert np.isfinite(linear).all() and (linear >= 0).all() and (linear <= 1).all()
    return np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)


def exif_metadata(path):
    with path.open('rb') as stream:
        tags = exifread.process_file(stream, details=False)
    def value(*keys):
        tag = next((tags[k] for k in keys if k in tags), None)
        assert tag is not None, f'Missing required EXIF {keys}: {path.name}'
        number = float(tag.values[0])
        assert np.isfinite(number) and number > 0
        return number
    assert 'Image Model' in tags
    return dict(model=str(tags['Image Model']), make=str(tags.get('Image Make', 'unknown')),
                exposure_seconds=value('EXIF ExposureTime'), f_number=value('EXIF FNumber'),
                ISO=value('EXIF ISOSpeedRatings', 'EXIF PhotographicSensitivity'),
                datetime_original=str(tags.get('EXIF DateTimeOriginal', 'unknown')),
                subsecond_original=str(tags.get('EXIF SubSecTimeOriginal', 'unknown')))


def raw_metadata(raw):
    assert raw.raw_image_visible.ndim == 2 and raw.raw_pattern is not None
    assert raw.color_desc == b'RGBG'
    return dict(sizes=dict(raw.sizes._asdict()), black_level_per_channel=list(raw.black_level_per_channel),
                white_level=int(raw.white_level), camera_white_level_per_channel=raw.camera_white_level_per_channel,
                camera_whitebalance=list(raw.camera_whitebalance), color_desc=raw.color_desc.decode(),
                raw_pattern=raw.raw_pattern.tolist(), rgb_xyz_matrix=raw.rgb_xyz_matrix.tolist())


def parameters(wb):
    assert len(wb) == 4 and np.isfinite(wb).all() and np.all(np.array(wb[:3]) > 0) and wb[3] >= 0
    return dict(half_size=True, user_flip=0, user_wb=wb, use_camera_wb=False, use_auto_wb=False,
                no_auto_bright=True, adjust_maximum_thr=0, bright=1, gamma=(1, 1),
                output_bps=16, output_color=rawpy.ColorSpace.sRGB, highlight_mode=rawpy.HighlightMode.Clip,
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off, noise_thr=None, median_filter_passes=0)


def make_view(linear16, gain):
    assert linear16.dtype == np.uint16 and linear16.shape[-1] == 3 and gain > 0
    linear = linear16.astype(np.float64)/65535*gain
    high = float(np.mean(linear > 1)); linear = np.clip(linear, 0, 1)
    h, w = linear.shape[:2]; scale = 640/max(h, w)
    shape = (round(w*scale), round(h*scale))
    assert max(shape) == 640
    small = cv2.resize(linear, shape, interpolation=cv2.INTER_AREA)
    output = np.clip(np.rint(encode_srgb(np.clip(small, 0, 1))*255), 0, 255).astype(np.uint8)
    return output, dict(exposure_gain=float(gain), pre_resize_above_one_fraction=high,
                        mean_u8=float(output.mean()), zero_channel_fraction=float(np.mean(output == 0)),
                        high_channel_fraction=float(np.mean(output == 255)))
