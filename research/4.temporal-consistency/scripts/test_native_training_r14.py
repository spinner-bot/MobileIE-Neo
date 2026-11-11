"""Old stream identities, same ROI, boundary history, and full optimizer continuation."""
import json
import unittest

import numpy as np
import torch

from continuation_r14 import MODES, create_model, initial_runtime, restore
from exploration_r2_data import ROOT
from native_training_r14 import pair_identity, real_pair


class NativeTrainingTests(unittest.TestCase):
    def fixture(self):
        t=np.arange(16,dtype=np.uint8)[:,None,None,None]
        entry={'scene':'fixture','native_input':np.broadcast_to(t,(16,1080,1920,3)),
            'native_gt':np.broadcast_to(t+20,(16,1080,1920,3)),
            'resized_input':np.broadcast_to(t,(16,180,320,3)),
            'resized_gt':np.broadcast_to(t+20,(16,180,320,3)),
            'source_files':{s:[{'path':f'{s}/{i}','sha256':str(i)} for i in range(16)] for s in ['input','gt']}}
        return [entry]

    def test_all_old_real_identities_match(self):
        selection=json.loads((ROOT/'research/4.temporal-consistency/data/exploration-r4b/manifest.json').read_text())['selection']
        names=[r['scene'] for r in selection if r['split']=='train']
        self.assertEqual(len(names),12)
        for seed in [0,1]:
            path=ROOT/'research/4.temporal-consistency/results/exploration-r5-spatial'/f'plain-seed{seed}'/'training.jsonl'
            for step,line in enumerate(path.read_text().splitlines()):
                r=json.loads(line)['sample_identity'][0];i,t=pair_identity(seed,step,len(names))
                self.assertEqual((names[i],t),(r['scene'],r['frame_pair']))

    def test_matching_current_GT_history_and_registered_crop(self):
        data=self.fixture()
        for seed,step in [(0,0),(1,999),(0,3999)]:
            native=real_pair(data,seed,step,'native_crop');resized=real_pair(data,seed,step,'resized')
            for a,b in zip(native[:3],resized[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            low,gt,past,identity,meta=native;t=identity['frame_pair']
            self.assertEqual(meta['previous_indices'],[max(0,t-2),t-1]);self.assertEqual(meta['current_indices'],[t-1,t])
            x,y,w,h=meta['crop_xywh'];self.assertTrue(0<=x<=1600 and 0<=y<=900);self.assertEqual((w,h),(320,180))
            torch.testing.assert_close(low[1],torch.full_like(low[1],t/255),rtol=0,atol=1e-8)
            torch.testing.assert_close(gt[1],torch.full_like(gt[1],(t+20)/255),rtol=0,atol=1e-8)
            torch.testing.assert_close(past[1],low[0],rtol=0,atol=0)
        step=next(s for s in range(100) if pair_identity(0,s,1)[1]==1)
        low,_,past,_,meta=real_pair(data,0,step,'native_crop')
        self.assertTrue(meta['self_fallback']);torch.testing.assert_close(past[0],low[0],rtol=0,atol=0)

    def test_all_three_existing_Adam_continuations_preserve_moments(self):
        for variant in MODES:
            model=create_model(variant,'cpu');optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0)
            audit=restore(model,optimizer,initial_runtime(variant,0),'cpu')
            self.assertEqual(audit['optimizer_step_range'],[1000,1000]);self.assertTrue(audit['Adam_moments_exactly_preserved'])
            self.assertEqual(audit['parameters'],4075 if variant=='plain' else 4975)
            self.assertGreater(audit['optimizer_tensors_verified'],0)

    def test_native_crop_uses_recorded_spatial_coordinates(self):
        data=self.fixture();y,x=np.mgrid[:1080,:1920]
        spatial=np.stack([x%251,y%251,(x+y)%251],-1).astype(np.uint8)
        data[0]['native_input']=np.broadcast_to(spatial,(16,1080,1920,3))
        low,_,_,_,metadata=real_pair(data,1,177,'native_crop');cx,cy,_,_=metadata['crop_xywh']
        self.assertAlmostEqual(float(low[0,0,0,0]),(cx%251)/255,places=7)
        self.assertAlmostEqual(float(low[0,1,0,0]),(cy%251)/255,places=7)
        self.assertAlmostEqual(float(low[1,2,-1,-1]),((cx+319+cy+179)%251)/255,places=7)


if __name__=='__main__':unittest.main()
