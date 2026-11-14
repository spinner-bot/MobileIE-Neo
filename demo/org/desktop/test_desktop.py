"""Core and actual Tk GUI acceptance for the three-task desktop application."""
import importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageGrab
sys.path.insert(0,str(Path(__file__).parent))
import desktop as lab
import lightlab_tasks as tasks
REPO=Path(__file__).resolve().parents[3]
OUT=Path(os.environ.get('LIGHTLAB_TEST_OUTPUT',tempfile.gettempdir()))
REF=Path(os.environ.get('LIGHTLAB_REFERENCE_DIR','/data/Users/zrh/data/MobileIE-Neo/demo/org/three-task/models-v2'))


class CoreTests(unittest.TestCase):
    def test_gamma_all_values(self):
        rgb=np.stack((np.arange(256),np.arange(255,-1,-1),(np.arange(256)*31)%256),axis=-1).astype(np.uint8)[None]
        np.testing.assert_array_equal(np.floor(np.sqrt(rgb.astype(float)/255)*255).astype(np.uint8),np.asarray(lab.gamma(Image.fromarray(rgb))))
    def test_rgb_input_and_errors(self):
        for path in tasks.TASKS['lowlight']['sample_assets']:
            p=tasks.ASSETS/path;self.assertEqual(Image.open(p).convert('RGB').tobytes(),lab.prepare(p).tobytes())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.png';Image.new('RGBA',(300,1200),(255,0,0,0)).save(p);self.assertEqual((255,255,255),lab.prepare(p).getpixel((0,0)))
            p.write_bytes(b'bad');self.assertRaises(OSError,lab.prepare,p)
            with p.open('wb') as f:f.truncate(33*1024*1024)
            self.assertRaises(ValueError,lab.prepare,p)
    def test_registry_and_native_input_contracts(self):
        self.assertEqual({'lowlight','underwater','isp'},set(tasks.TASKS))
        for task in tasks.TASKS:
            e=tasks.Engine(task);self.assertEqual(1,len(e.interpreter.get_input_details()));self.assertEqual([],tasks.TASKS[task]['native_control_inputs'])
        self.assertEqual([1,224,224,4],tasks.TASKS['isp']['input_shape']);self.assertEqual([1,448,448,3],tasks.TASKS['isp']['output_shape'])
    def test_lossless_raw_and_invalid_formats(self):
        p=tasks.ASSETS/tasks.TASKS['isp']['sample_assets'][0];value=tasks.prepare(p,'isp');raw=value.raw
        self.assertEqual((448,448),raw.shape);self.assertEqual(p.read_bytes(),value.original_bytes)
        for y,x in ((0,0),(5,11),(100,200)):
            np.testing.assert_array_equal(value.tensor[0,y,x],raw[2*y:2*y+2,2*x:2*x+2].reshape(4).astype(np.float32)/np.float32(4095))
        self.assertRaises(ValueError,tasks.prepare,tasks.ASSETS/'samples/01.png','isp')
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/'bad.rggb';b=bytearray(p.read_bytes());b[0]=0;bad.write_bytes(b);self.assertRaises(ValueError,tasks.prepare,bad,'isp')
    def test_fixed_traditional_baselines(self):
        im=Image.new('RGB',(6,8),(40,80,120));out=np.asarray(tasks.gray_world(im));self.assertLessEqual(int(np.abs(out.astype(int)-80).max()),1)
        out=np.asarray(tasks.bilinear_isp(np.full((8,8),100,np.uint16)));self.assertTrue((out[:,:,0]==out[:,:,1]).all());self.assertTrue((out[:,:,1]==out[:,:,2]).all());self.assertTrue((out==out[0,0]).all())
    def test_all_models_against_independent_references(self):
        spec=importlib.util.spec_from_file_location('official_wrapper',REPO/'demo/org/scripts/run_official_tflite.py');wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper);wrapper.ensure_tflite_runtime();official=wrapper.load_official(REPO/'MobileIE/test_TFLite_RGB.py');reference=official.load_model(str(tasks.ASSETS/'model.tflite'));records=[]
        for task in tasks.TASKS:
            engine=tasks.Engine(task)
            for index,path in enumerate(tasks.TASKS[task]['sample_assets'],1):
                p=tasks.ASSETS/path;value=tasks.prepare(p,task);image,raw,ms=engine.run(value);self.assertTrue(np.isfinite(raw).all());self.assertGreaterEqual(ms,0)
                if task=='lowlight':
                    reference.set_tensor(reference.get_input_details()[0]['index'],official.preprocess_image(str(p)));reference.invoke();expected=reference.get_tensor(reference.get_output_details()[0]['index'])
                else:expected=np.load(REF/f'{task}-sample{index}-reference.npy')
                delta=float(np.abs(raw.astype(float)-expected.astype(float)).max());pixel=int(np.abs(np.asarray(image).astype(int)-(np.clip(expected[0],0,1)*255).astype(np.uint8).astype(int)).max());self.assertLessEqual(delta,1e-4);self.assertLessEqual(pixel,1)
                if task=='lowlight':self.assertEqual(0,pixel)
                records.append({'task':task,'sample':index,'raw_max_abs':delta,'uint8_max_abs':pixel})
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'desktop-numerics.json').write_text(json.dumps({'pass':True,'cases':records,'native_windows_tested':sys.platform.startswith('win')},indent=2)+'\n')
    def test_save_roundtrip_task_identity_and_rollback(self):
        for task in tasks.TASKS:
            value=tasks.prepare(tasks.ASSETS/tasks.TASKS[task]['sample_assets'][0],task);mie,_,_=tasks.Engine(task).run(value);images={'Input':value.image,'MobileIE':mie,'Baseline':tasks.baseline(value)};record={'task_id':task,'task_name':tasks.TASKS[task]['name'],'baseline_name':tasks.TASKS[task]['baseline_name']}
            with tempfile.TemporaryDirectory() as d:
                target=lab.save_results(d,images,record,value)
                for key,name in (('Input','input.png'),('MobileIE','mobileie.png'),('Baseline','baseline.png')):self.assertEqual(images[key].tobytes(),Image.open(target/name).tobytes())
                self.assertEqual((3*value.image.width,value.image.height+110),Image.open(target/'comparison.png').size);self.assertEqual(record,json.loads((target/'record.json').read_text(encoding='utf-8')))
                if task=='isp':self.assertEqual(value.original_bytes,(target/'input.rggb').read_bytes())
                before=set(Path(d).iterdir());self.assertRaises(TypeError,lab.save_results,d,images,{'bad':object()},value);self.assertEqual(before,set(Path(d).iterdir()))


@unittest.skipUnless(os.environ.get('LIGHTLAB_GUI_TEST')=='1','GUI requires an isolated display')
class GuiTests(unittest.TestCase):
    def test_three_tasks_two_displays_and_failure(self):
        import tkinter as tk
        root=tk.Tk();app=lab.Desktop(root)
        def wait():
            end=time.monotonic()+30
            while app.busy and time.monotonic()<end:root.update();time.sleep(.01)
            self.assertFalse(app.busy);root.update();time.sleep(.12);root.update()
        def shot(name):
            # 先置顶，避免截到压在窗口上的其他程序；再按系统缩放换算物理像素
            root.lift();root.attributes('-topmost',True);root.update();time.sleep(.2);root.update()
            full=ImageGrab.grab();sx=full.width/root.winfo_screenwidth();sy=full.height/root.winfo_screenheight()
            box=(int(root.winfo_rootx()*sx),int(root.winfo_rooty()*sy),int((root.winfo_rootx()+root.winfo_width())*sx),int((root.winfo_rooty()+root.winfo_height())*sy))
            full.crop(box).save(OUT/name)
        try:
            OUT.mkdir(parents=True,exist_ok=True);root.update();shot('desktop-home.png')
            for task in tasks.TASKS:
                app.change_task(task);self.assertIsNone(app.input);self.assertEqual({},app.record);self.assertIn('待处理',app.result_task.get());wait()
                app.process();self.assertTrue(app.busy);self.assertEqual('disabled',str(app.task_box.cget('state')));self.assertEqual('disabled',str(app.info_button.cget('state')));app.change_task('isp' if task!='isp' else 'lowlight');self.assertEqual(task,app.active_task);wait()
                self.assertIsNotNone(app.images['MobileIE']);self.assertIsNotNone(app.images['Baseline']);self.assertEqual(task,app.record['task_id']);self.assertEqual('本次任务：'+tasks.TASKS[task]['name'],app.result_task.get())
                app.select(lab.MODES[0]);root.update();self.assertEqual(2,len(app.rects));rectangles=list(app.rects)
                for index,fraction in ((0,.22),(1,.78)):
                    x,y,w,h=app.rects[index];app.drag(SimpleNamespace(x=x+w*fraction,y=y+h/2));self.assertAlmostEqual(fraction,app.split,places=2)
                    lines=[app.canvas.coords(item) for item in app.canvas.find_all() if app.canvas.type(item)=='line'];self.assertEqual(2,len(lines))
                    for line,rect in zip(lines,app.rects):self.assertAlmostEqual(fraction,(line[0]-rect[0])/rect[2],places=2)
                shot(task+'-dynamic.png');app.select(lab.MODES[1]);root.update();self.assertEqual(3,len(app.rects));self.assertEqual(rectangles[0][3],rectangles[1][3]);shot(task+'-static.png')
                with tempfile.TemporaryDirectory() as d:
                    app.filedialog.askdirectory=lambda **_:d;app.save();record=json.loads(next(Path(d).glob('MobileIE-*/record.json')).read_text(encoding='utf-8'));self.assertEqual(task,record['task_id'])
            app.change_task('underwater');wait()
            class Failure:
                def run(self,_):raise RuntimeError('controlled model failure')
            app.engines['underwater']=Failure();app.process();wait();self.assertIsNone(app.images['MobileIE']);self.assertIsNotNone(app.images['Baseline']);self.assertIn('未生成',app.status.get());self.assertIn('水下增强',app.result_task.get());shot('desktop-model-failure.png')
            root.geometry('900x650');root.update();shot('desktop-small.png');app.rail_scroll.yview_moveto(1);root.update();shot('desktop-small-scrolled.png');self.assertLessEqual(app.info_button.winfo_rooty()+app.info_button.winfo_height(),root.winfo_rooty()+root.winfo_height())
        finally:app.close();app.worker_thread.join(timeout=5)
        self.assertFalse(app.worker_thread.is_alive());(OUT/'desktop-gui.json').write_text(json.dumps({'pass':True,'tasks':3,'display_modes':2,'task_label_bound_to_record':True,'paired_slider_sync':True,'task_switch_clears_old_result':True,'model_failure_baseline_preserved':True,'small_window_checked':True,'native_windows_tested':sys.platform.startswith('win')},indent=2)+'\n')


if __name__=='__main__':unittest.main(verbosity=2)
