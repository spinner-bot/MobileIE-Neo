"""Channel-sensitive fixtures expose incorrect RGB/BGR routing or quantization."""
import unittest

import numpy as np
import torch

from stablellve_adapter_r15 import infer, input_tensor, output_rgb


class ColorModel:
    def __call__(self,x):return x*torch.tensor([1.,2.,3.],dtype=x.dtype)[None,:,None,None]


class StableLLVEAdapterTests(unittest.TestCase):
    def test_channel_sensitive_roundtrip_preserves_fractional_input(self):
        image=np.zeros((2,19,23,3),np.float32);image[:]=[10.25,20.5,30.75]
        encoded=input_tensor(image,'cpu');prediction=infer(ColorModel(),encoded);out=output_rgb(prediction)
        expected=np.broadcast_to([30.75,41.,30.75],image.shape)
        np.testing.assert_allclose(out,expected,rtol=0,atol=4e-6)
        self.assertEqual(encoded.dtype,torch.float32);self.assertTrue(out.flags.c_contiguous)

    def test_input_normalization_matches_official_test_path(self):
        rgb=np.random.default_rng(15).integers(0,256,(37,53,3),dtype=np.uint8)
        bgr=rgb[...,::-1];official=np.expand_dims((bgr/255.).transpose(2,0,1),0)
        reference=torch.from_numpy(official.copy()).to(dtype=torch.float32)
        torch.testing.assert_close(input_tensor(rgb[None],'cpu'),reference,rtol=0,atol=0)
        with self.assertRaises(ValueError):input_tensor(np.zeros((1,8,8,3)),'cpu')


if __name__=='__main__':unittest.main()
