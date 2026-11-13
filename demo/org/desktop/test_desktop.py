"""Portable core gates; GUI simulation is opt-in via LIGHTLAB_GUI_TEST=1."""
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image, ImageGrab

SPEC=importlib.util.spec_from_file_location("lightlab",Path(__file__).with_name("desktop.py"))
lab=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(lab)
REPO=Path(__file__).resolve().parents[3]
OUT=Path(os.environ.get("LIGHTLAB_TEST_OUTPUT",tempfile.gettempdir()))


class CoreTests(unittest.TestCase):
    def test_gamma_all_values(self):
        rgb=np.stack((np.arange(256),np.arange(255,-1,-1),(np.arange(256)*31)%256),axis=-1).astype(np.uint8)[None]
        actual=np.asarray(lab.gamma(Image.fromarray(rgb)))
        expected=np.floor(np.sqrt(rgb.astype(np.float64)/255)*255).astype(np.uint8)
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual([lab.GAMMA_LUT[i] for i in (0,64,128,255)],[0,127,180,255])

    def test_input_contract_and_errors(self):
        for index in range(1,4):
            path=lab.ASSETS/f"samples/{index:02d}.png"
            self.assertEqual(Image.open(path).convert("RGB").tobytes(),lab.prepare(path).tobytes())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"input.png"
            Image.new("RGBA",(300,1200),(255,0,0,0)).save(path)
            prepared=lab.prepare(path);self.assertEqual((600,400),prepared.size);self.assertEqual((255,255,255),prepared.getpixel((0,0)))
            path.write_bytes(b"bad image")
            with self.assertRaises(OSError):lab.prepare(path)
            with path.open("wb") as f:f.truncate(33*1024*1024)
            with self.assertRaisesRegex(ValueError,"超过"):lab.prepare(path)

    def test_official_model_reference(self):
        spec=importlib.util.spec_from_file_location("official_wrapper",REPO/"demo/org/scripts/run_official_tflite.py")
        wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper);wrapper.ensure_tflite_runtime()
        official=wrapper.load_official(REPO/"MobileIE/test_TFLite_RGB.py")
        reference=official.load_model(str(lab.ASSETS/"model.tflite"));engine=lab.Engine();records=[]
        for index in range(1,4):
            path=lab.ASSETS/f"samples/{index:02d}.png"
            result,raw,ms=engine.run(lab.prepare(path))
            reference.set_tensor(reference.get_input_details()[0]["index"],official.preprocess_image(str(path)));reference.invoke()
            expected=reference.get_tensor(reference.get_output_details()[0]["index"])
            maximum=float(np.abs(expected.astype(np.float64)-raw.astype(np.float64)).max())
            self.assertLessEqual(maximum,1e-4);self.assertTrue(np.isfinite(raw).all());self.assertTrue(math.isfinite(ms))
            np.testing.assert_array_equal(np.asarray(result),(np.clip(expected[0],0,1)*255).astype(np.uint8))
            records.append({"sample":index,"raw_max_abs":maximum,"uint8_exact":True})
        OUT.mkdir(parents=True,exist_ok=True);(OUT/"desktop-numerics.json").write_text(json.dumps({"pass":True,"cases":records,"runtime":"ai-edge-litert 2.3.0 CPU builtin","reference":"unchanged official preprocessing, same runtime"},indent=2)+"\n")

    def test_save_roundtrip_and_rollback(self):
        source=lab.prepare(lab.ASSETS/"samples/01.png");mie,_,_=lab.Engine().run(source)
        images={"Input":source,"MobileIE":mie,"Gamma 0.5":lab.gamma(source)};record={"model_sha256":lab.MODEL_SHA,"gamma_exponent":.5}
        with tempfile.TemporaryDirectory() as directory:
            destination=lab.save_results(Path(directory),images,record)
            for key,name in (("Input","input.png"),("MobileIE","mobileie.png"),("Gamma 0.5","gamma-0.5.png")):
                self.assertEqual(images[key].tobytes(),Image.open(destination/name).tobytes())
            with Image.open(destination/"comparison.png") as saved:self.assertEqual((1800,478),saved.size)
            self.assertEqual(record,json.loads((destination/"record.json").read_text()))
            before=set(Path(directory).iterdir())
            with self.assertRaises(TypeError):lab.save_results(Path(directory),images,{"bad":object()})
            self.assertEqual(before,set(Path(directory).iterdir()))


@unittest.skipUnless(os.environ.get("LIGHTLAB_GUI_TEST")=="1","GUI simulation requires a display")
class GuiTests(unittest.TestCase):
    def test_complete_gui_flow_and_failure(self):
        import tkinter as tk
        root=tk.Tk();app=lab.Desktop(root)
        def wait():
            end=time.monotonic()+20
            while app.busy and time.monotonic()<end:root.update();time.sleep(.01)
            self.assertFalse(app.busy);root.update();time.sleep(.15);root.update()
        def shot(name):
            root.update();time.sleep(.2);root.update()
            ImageGrab.grab(bbox=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())).save(OUT/name)
        try:
            OUT.mkdir(parents=True,exist_ok=True);root.update();shot("desktop-01-home.png")
            self.assertEqual("disabled",str(app.actions[2].cget("state")))
            app.choose_sample();self.assertTrue(app.busy);wait();self.assertIsNotNone(app.input)
            app.process();self.assertTrue(app.busy);self.assertIn("正在",app.metrics.get());wait();self.assertIsNotNone(app.images["MobileIE"])
            shot("desktop-02-result.png")
            app.select("Gamma 0.5");root.update();x,y,w,h=app.rect;app.drag(SimpleNamespace(x=x+w*.8,y=y+h/2));self.assertAlmostEqual(.8,app.split,places=2)
            app.select("三图总览");shot("desktop-03-overview.png")
            original=app.input.tobytes();app.filedialog.askopenfilename=lambda **_:"";app.choose();self.assertEqual(original,app.input.tobytes())
            with tempfile.TemporaryDirectory() as directory:
                app.filedialog.askdirectory=lambda **_:directory;app.save();self.assertTrue(any(Path(directory).glob("MobileIE-*/record.json")))
                invalid=Path(directory)/"bad.png";invalid.write_bytes(b"not an image");app.load(invalid);wait();self.assertIn("未加载",app.status.get());self.assertEqual(original,app.input.tobytes())
            app.load(lab.ASSETS/"samples/02.png");wait();self.assertIsNone(app.images["MobileIE"]);self.assertEqual({},app.record);self.assertIn("处理后",app.metrics.get())
            class Failure:
                def run(self,_):raise RuntimeError("controlled failure")
            app.engine=Failure();app.process();wait();self.assertIsNone(app.images["MobileIE"]);self.assertIsNotNone(app.images["Gamma 0.5"])
            self.assertEqual("disabled",str(app.chips["MobileIE"].cget("state")));self.assertIn("未生成",app.status.get());app.select("三图总览");shot("desktop-04-model-failure.png")
            root.geometry("900x650");root.update();shot("desktop-05-small-window.png")
            self.assertGreater(app.canvas.winfo_height(),100)
            for widget in (app.actions[3],app.chips["Gamma 0.5"]):self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),root.winfo_rooty()+root.winfo_height())
            for widget in app.rail.winfo_children():self.assertGreaterEqual(widget.winfo_height(),widget.winfo_reqheight())
            self.assertGreater(app.info_button.winfo_height(),20)
            app.rail_scroll.yview_moveto(1);root.update();last=app.rail.winfo_children()[-1]
            self.assertLessEqual(last.winfo_rooty()+last.winfo_height(),app.rail_scroll.winfo_rooty()+app.rail_scroll.winfo_height())
            shot("desktop-06-small-window-scrolled.png")
        finally:app.close();app.worker_thread.join(timeout=5)
        self.assertFalse(app.worker_thread.is_alive())
        (OUT/"desktop-gui.json").write_text(json.dumps({"pass":True,"platform":"Linux/Xvfb Tk 8.6 simulation","native_windows_tested":False,"checks":["initial controls","async processing","slider","overview","cancel","PNG/JSON save","invalid input retention","new input clears old timing","model failure baseline","disabled unavailable method","900x650 layout","clean worker exit"]},indent=2)+"\n")


if __name__=="__main__":unittest.main(verbosity=2)
