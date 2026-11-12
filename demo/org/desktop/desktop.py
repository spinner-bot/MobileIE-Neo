"""MobileIE Light Lab: portable Python desktop companion (no native build tools)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageCms, ImageFont

VERSION = "0.1.0"
MODEL_SHA = "2ea09b840634469eceab5263bf7b549c33fcd5df2a25701ce5fefc585402ad44"
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets" if (HERE / "assets").is_dir() else HERE.parent / "app/app/src/main/assets"
GAMMA_LUT = [int(255 * (i / 255.0) ** .5) for i in range(256)]


def prepare(path: Path) -> Image.Image:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("图片超过 32 MB，请选择较小的 JPG / PNG")
    with Image.open(path) as image:
        if max(image.size) > 50000:
            raise ValueError("图片尺寸过大，请选择较小的照片")
        image.draft("RGB", (2048, 2048))
        corrected = ImageOps.exif_transpose(image)
        profile = image.info.get("icc_profile")
        if corrected.mode in ("RGBA", "LA") or "transparency" in corrected.info:
            rgba = corrected.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            corrected = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            corrected = corrected.convert("RGB")
        if profile:
            corrected = ImageCms.profileToProfile(corrected, ImageCms.ImageCmsProfile(io.BytesIO(profile)), ImageCms.createProfile("sRGB"), outputMode="RGB")
        corrected.thumbnail((2048, 2048), Image.Resampling.BILINEAR)
        w, h = corrected.size
        cw, ch = (max(1, h * 3 // 2), h) if w * 2 > h * 3 else (w, max(1, w * 2 // 3))
        x, y = (w-cw)//2, (h-ch)//2
        return corrected.crop((x, y, x+cw, y+ch)).resize((600, 400), Image.Resampling.BILINEAR)


def gamma(image: Image.Image) -> Image.Image:
    return image.point(GAMMA_LUT * 3)


class Engine:
    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        start = time.perf_counter()
        model = ASSETS / "model.tflite"
        if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA:
            raise ValueError("模型校验失败，请重新获取演示包")
        self.interpreter = Interpreter(model_path=str(model), num_threads=4,
                                       experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        assert self.input["shape"].tolist() == self.output["shape"].tolist() == [1, 400, 600, 3]
        assert self.input["dtype"] == self.output["dtype"] == np.float32
        self.initialization_ms = (time.perf_counter()-start)*1000

    def run(self, image: Image.Image):
        if image.size != (600, 400) or image.mode != "RGB":
            raise ValueError("需要统一的 600 × 400 RGB 输入")
        values = np.asarray(image, dtype=np.float32)[None] / np.float32(255)
        self.interpreter.set_tensor(self.input["index"], values)
        start = time.perf_counter()
        self.interpreter.invoke()
        milliseconds = (time.perf_counter()-start)*1000
        raw = self.interpreter.get_tensor(self.output["index"]).copy()
        if not np.isfinite(raw).all():
            raise ValueError("模型输出异常，请重新处理")
        result = Image.fromarray((np.clip(raw[0], 0, 1)*np.float32(255)).astype(np.uint8))
        return result, raw, milliseconds


def comparison(images: dict[str, Image.Image | None]):
    sheet = Image.new("RGB", (1800, 478), "#f6f5ef")
    draw = ImageDraw.Draw(sheet)
    label_font, footer_font = ImageFont.load_default(size=24), ImageFont.load_default(size=18)
    for i, name in enumerate(("Input", "MobileIE", "Gamma 0.5")):
        draw.text((i*600+16, 10), name, fill="#243d35", font=label_font)
        if images[name] is not None:
            sheet.paste(images[name], (i*600, 44))
        else:
            draw.text((i*600+180, 220), "Not generated", fill="#68766c", font=label_font)
    draw.text((16, 450), "Same 600 x 400 input | local CPU | fixed Gamma exponent 0.5, no denoising", fill="#68766c", font=footer_font)
    return sheet


def save_results(directory: Path, images: dict, record: dict) -> Path:
    # Build in a new directory and rename only after every output succeeds.
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    temporary = directory / (".MobileIE-"+token)
    target = directory / ("MobileIE-"+token)
    temporary.mkdir()
    try:
        for name, filename in (("Input", "input.png"), ("MobileIE", "mobileie.png"), ("Gamma 0.5", "gamma-0.5.png")):
            if images[name] is not None:
                images[name].save(temporary / filename)
        comparison(images).save(temporary / "comparison.png")
        (temporary / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        temporary.rename(target)
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


class Desktop:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from PIL import ImageTk
        self.tk, self.filedialog, self.messagebox, self.ImageTk = tk, filedialog, messagebox, ImageTk
        self.root, self.input, self.engine = root, None, None
        self.images = {"Input": None, "MobileIE": None, "Gamma 0.5": None}
        self.record, self.mode, self.split, self.busy = {}, "MobileIE", .5, False
        self.tasks, self.results = queue.Queue(), queue.Queue()
        self.closed = False
        self.font = "Microsoft YaHei" if sys.platform == "win32" else "Noto Sans CJK SC"
        root.title("微光实验室 · MobileIE Python"); root.geometry("1180x850"); root.minsize(960, 760); root.configure(bg="#f6f5ef")
        header=tk.Frame(root,bg="#f6f5ef");header.pack(fill="x",padx=28,pady=(24,18))
        tk.Label(header,text="◒  微光实验室",font=(self.font,24,"bold"),fg="#243d35",bg="#f6f5ef").pack(side="left")
        tk.Label(header,text="PYTHON DESKTOP  /  本机 · 离线",font=(self.font,11),fg="#68766c",bg="#f6f5ef").pack(side="right")
        body=tk.Frame(root,bg="#f6f5ef");body.pack(fill="both",expand=True,padx=28)
        rail=tk.Frame(body,bg="#e8ecdf",width=250);rail.pack(side="left",fill="y",padx=(0,20));rail.pack_propagate(False)
        tk.Label(rail,text="同一张照片，\n看见不同处理。",justify="left",font=(self.font,20,"bold"),fg="#243d35",bg="#e8ecdf").pack(anchor="w",padx=20,pady=(24,12))
        self.actions=[]
        for label,command,primary in (("选择图片",self.choose,True),("内置暗光样例",self.choose_sample,False),("开始对比",self.process,True),("保存结果",self.save,False)):
            button=self.button(rail,label,command,primary);button.pack(fill="x",padx=20,pady=6);self.actions.append(button)
        tk.Label(rail,text="01  选图并确认处理范围\n02  处理后拖动分界线\n03  切换传统 Gamma 对照",justify="left",font=(self.font,11),fg="#68766c",bg="#e8ecdf").pack(anchor="w",padx=20,pady=18)
        tk.Label(rail,text="统一输入 600 × 400\nGamma 指数固定为 0.5\nGamma 不包含降噪",justify="left",font=(self.font,10),fg="#68766c",bg="#e8ecdf").pack(anchor="w",padx=20)
        self.button(rail,"查看计算记录",self.info,False).pack(side="bottom",fill="x",padx=20,pady=20)
        content=tk.Frame(body,bg="white");content.pack(side="left",fill="both",expand=True)
        self.status=tk.StringVar(value="选一张暗光照片，开始本机对比")
        tk.Label(content,textvariable=self.status,anchor="w",wraplength=760,font=(self.font,12),bg="white",fg="#243d35").pack(fill="x",padx=20,pady=(18,6))
        self.source=tk.StringVar(value="原图指校正方向、居中裁剪后的同一处理范围")
        tk.Label(content,textvariable=self.source,anchor="w",font=(self.font,10),bg="white",fg="#68766c").pack(fill="x",padx=20)
        self.canvas=tk.Canvas(content,bg="#243d35",highlightthickness=0,height=480)
        self.canvas.pack(fill="both",expand=True,padx=20,pady=16)
        self.canvas.bind("<Configure>",self.redraw);self.canvas.bind("<B1-Motion>",self.drag);self.canvas.bind("<Button-1>",self.drag)
        chips=tk.Frame(content,bg="white");chips.pack(fill="x",padx=20,pady=(0,12));self.chips={}
        for label in ("MobileIE","Gamma 0.5","三图总览"):
            b=self.button(chips,label,lambda name=label:self.select(name),False);b.pack(side="left",expand=True,fill="x",padx=4);self.chips[label]=b
        self.metrics=tk.StringVar(value="处理后显示本次 CPU 耗时；安卓测速请使用 AI Benchmark")
        tk.Label(content,textvariable=self.metrics,font=(self.font,10),fg="#68766c",bg="white",wraplength=740).pack(padx=20,pady=(0,18))
        tk.Label(root,text="MobileIE 官方模型  ·  原图 / 学习增强 / 固定 Gamma  ·  单张本地处理",font=(self.font,10),fg="#68766c",bg="#f6f5ef").pack(pady=14)
        self.worker_thread=threading.Thread(target=self.worker,daemon=True);self.worker_thread.start();root.after(40,self.poll);root.protocol("WM_DELETE_WINDOW",self.close)
        self.update_controls()

    def button(self,parent,label,command,primary):
        return self.tk.Button(parent,text=label,command=command,font=(self.font,11,"bold"),bg="#c45738" if primary else "#e8ecdf",fg="white" if primary else "#243d35",activebackground="#d7b098",relief="flat",borderwidth=0,padx=14,pady=11,cursor="hand2")

    def choose(self):
        filename=self.filedialog.askopenfilename(filetypes=[("照片","*.png *.jpg *.jpeg *.webp"),("所有文件","*.*")])
        if filename:self.load(Path(filename))

    def choose_sample(self):
        self.load(ASSETS/"samples/01.png")

    def load(self,path):
        if self.busy:return
        self.busy=True;self.status.set("正在准备图片…");self.update_controls();self.tasks.put(("load",Path(path)))

    def process(self):
        if self.busy or self.input is None:return
        self.busy=True;self.status.set("正在本机处理，请稍候…");self.metrics.set("正在计算本次结果…");self.images={"Input":self.input,"MobileIE":None,"Gamma 0.5":None};self.update_controls();self.tasks.put(("process",self.input.copy()))

    def worker(self):
        while True:
            kind,value=self.tasks.get()
            if kind=="stop":self.engine=None;return
            try:
                if kind=="load":self.results.put((kind,(prepare(value),value.name)))
                elif kind=="process":
                    start=time.perf_counter();g=gamma(value);gamma_ms=(time.perf_counter()-start)*1000;error=None;mie=None;init=self.engine is None
                    try:
                        if self.engine is None:self.engine=Engine()
                        mie,_,ms=self.engine.run(value)
                    except Exception as exc:error=type(exc).__name__;ms=None;self.engine=None
                    record={"app_version":VERSION,"model_sha256":MODEL_SHA,"width":600,"height":400,"gamma_exponent":.5,"gamma_ms":gamma_ms,"inference_ms":ms,"initialization_in_this_run":init,"runtime":"ai-edge-litert 2.3.0, CPU builtin, 4 threads","mobileie_complete":mie is not None,"gamma_complete":True,"mobileie_error":error,"input_rgb_sha256":hashlib.sha256(value.tobytes()).hexdigest()}
                    self.results.put((kind,(mie,g,record)))
            except Exception as exc:self.results.put(("error",type(exc).__name__))

    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,value=self.results.get_nowait();self.busy=False
                if kind=="load":
                    self.input,name=value;self.images={"Input":self.input,"MobileIE":None,"Gamma 0.5":None};self.record={};self.mode="MobileIE";self.split=.5;self.metrics.set("处理后显示本次 CPU 耗时；安卓测速请使用 AI Benchmark");self.source.set(name+"  ·  600 × 400 居中处理范围");self.status.set("图片已准备，请确认取景范围再开始对比")
                elif kind=="process":
                    mie,g,self.record=value;self.images={"Input":self.input,"MobileIE":mie,"Gamma 0.5":g};self.mode="MobileIE" if mie else "Gamma 0.5";self.status.set("处理完成 · 两种方法均在本机计算" if mie else "MobileIE 未生成；Gamma 结果仍可查看，请核对依赖或模型")
                    self.metrics.set(f"MobileIE {self.record['inference_ms']:.1f} ms  ·  Gamma {self.record['gamma_ms']:.1f} ms" if mie else "MobileIE 本次未生成；Gamma 已完成")
                else:self.status.set("图片未加载或处理失败，请重新选择有效的 JPG / PNG")
                self.update_controls();self.redraw()
        except queue.Empty:pass
        self.root.after(40,self.poll)

    def update_controls(self):
        for b in self.actions:b.configure(state="disabled" if self.busy else "normal")
        self.actions[2].configure(state="normal" if self.input is not None and not self.busy else "disabled")
        self.actions[3].configure(state="normal" if self.images["Gamma 0.5"] is not None and not self.busy else "disabled")
        for label,b in self.chips.items():b.configure(bg="#243d35" if label==self.mode else "#e8ecdf",fg="white" if label==self.mode else "#243d35",state="disabled" if self.images["Gamma 0.5"] is None or (label=="MobileIE" and self.images["MobileIE"] is None) else "normal")

    def select(self,mode):
        self.mode=mode;self.update_controls();self.redraw()

    def redraw(self,event=None):
        c=self.canvas;c.delete("all");w,h=max(c.winfo_width(),1),max(c.winfo_height(),1)
        if self.input is None:
            c.create_oval(w*.55,h*.08,w*.85,h*.50,fill="#e9c28e",outline="");c.create_oval(w*.61,h*.02,w*.91,h*.44,fill="#243d35",outline="")
            c.create_text(32,h*.68,text="让暗处的细节，重新被看见。",anchor="w",fill="#f6f5ef",font=(self.font,18,"bold"));return
        if self.mode=="三图总览":image=comparison(self.images)
        else:
            image=self.input.copy();processed=self.images.get(self.mode)
            if processed is not None:
                x=int(600*self.split);image.paste(processed.crop((x,0,600,400)),(x,0))
        ratio=min(w/image.width,h/image.height);size=(max(1,int(image.width*ratio)),max(1,int(image.height*ratio)))
        self.rect=((w-size[0])//2,(h-size[1])//2,size[0],size[1]);self._photo=self.ImageTk.PhotoImage(image.resize(size,Image.Resampling.BILINEAR));c.create_image(w//2,h//2,image=self._photo)
        if self.mode!="三图总览" and self.images.get(self.mode) is not None:
            x0,y0,iw,ih=self.rect;x=x0+int(iw*self.split);c.create_line(x,y0,x,y0+ih,fill="white",width=2);c.create_oval(x-16,h/2-16,x+16,h/2+16,fill="white",outline="");c.create_text(x,h/2,text="↔",fill="#243d35",font=(self.font,12))

    def drag(self,event):
        if self.input is None or self.mode=="三图总览":return
        x,y,w,h=self.rect;self.split=max(.03,min(.97,(event.x-x)/max(w,1)));self.redraw()

    def save(self):
        if self.images["Gamma 0.5"] is None or self.busy:return
        directory=self.filedialog.askdirectory(title="选择结果保存目录")
        if not directory:return
        try:target=save_results(Path(directory),self.images,self.record);self.status.set("已保存："+str(target))
        except Exception:self.status.set("保存失败，请检查目录与可用空间，当前结果仍可查看")

    def info(self):
        self.messagebox.showinfo("计算记录",json.dumps(self.record,ensure_ascii=False,indent=2) if self.record else "选择图片并处理后可查看本次记录。")

    def close(self):
        self.closed=True;self.tasks.put(("stop",None));self._photo=None;self.root.destroy()


def main():
    import tkinter as tk
    root=tk.Tk();desktop=Desktop(root);root.mainloop();desktop.worker_thread.join(timeout=5)


if __name__=="__main__":main()
