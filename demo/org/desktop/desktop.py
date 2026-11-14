"""MobileIE Light Lab: three tasks, paired dynamic or static comparison."""
from __future__ import annotations
import hashlib,json,os,queue,shutil,sys,threading,time,uuid
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from lightlab_tasks import ASSETS,TASKS,Engine,Prepared,prepare as prepare_task,prepare_rgb as prepare,gamma,GAMMA_LUT,baseline
VERSION='0.2.0'
MODEL_SHA=TASKS['lowlight']['model_sha256']
MODES=('动态对比','静态总览')


def export_font(size):
    for p in (Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc',Path('/data/Users/zrh/.envs/android/validation/desktop-host/root/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')):
        if p.exists():return ImageFont.truetype(str(p),size)
    return ImageFont.load_default(size=size)


def comparison(images,record=None):
    record=record or {};source=next(im for im in images.values() if im is not None);w,h=source.size
    sheet=Image.new('RGB',(3*w,h+110),'#f6f5ef');draw=ImageDraw.Draw(sheet)
    draw.text((16,6),'本次任务：'+record.get('task_name','低光增强'),fill='#243d35',font=export_font(24))
    labels=(('Input','RAW 可视化' if record.get('task_id')=='isp' else '原图'),('MobileIE','MobileIE'),('Baseline',record.get('baseline_name','Gamma 0.5')))
    for i,(key,label) in enumerate(labels):
        draw.text((i*w+16,40),label,fill='#243d35',font=export_font(20))
        if images.get(key) is not None:sheet.paste(images[key],(i*w,76))
        else:draw.text((i*w+w//3,76+h//2),'未生成',fill='#68766c',font=export_font(20))
    draw.text((16,h+82),'同一输入与坐标 · 本地计算 · 固定传统 baseline',fill='#68766c',font=export_font(17));return sheet


def save_results(directory,images,record,prepared=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);token=uuid.uuid4().hex[:10]
    temporary=directory/('.MobileIE-'+token+'.part');target=directory/('MobileIE-'+token);temporary.mkdir()
    try:
        for key,name in (('Input','input.png'),('MobileIE','mobileie.png'),('Baseline','baseline.png')):
            if images.get(key) is not None:images[key].save(temporary/name)
        comparison(images,record).save(temporary/'comparison.png')
        if prepared is not None and prepared.original_bytes is not None:(temporary/'input.rggb').write_bytes(prepared.original_bytes)
        (temporary/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');temporary.rename(target);return target
    except Exception:shutil.rmtree(temporary,ignore_errors=True);raise


class Desktop:
    def __init__(self,root):
        import tkinter as tk
        from tkinter import ttk,filedialog,messagebox
        from PIL import ImageTk
        self.tk,self.filedialog,self.messagebox,self.ImageTk=tk,filedialog,messagebox,ImageTk
        self.root=root;self.active_task='lowlight';self.prepared=self.input=None;self.images={'Input':None,'MobileIE':None,'Baseline':None}
        self.record={};self.mode=MODES[0];self.split=.5;self.busy=self.closed=False;self.rects=[];self._photos=[]
        self.tasks,self.results=queue.Queue(),queue.Queue();self.engines={};self.font='Microsoft YaHei' if sys.platform=='win32' else 'Noto Sans CJK SC'
        root.title('微光实验室 · 三任务对比');root.geometry('1280x850');root.minsize(900,650);root.configure(bg='#f6f5ef')
        head=tk.Frame(root,bg='#f6f5ef');head.pack(fill='x',padx=24,pady=(18,14))
        tk.Label(head,text='◒  微光实验室',font=(self.font,23,'bold'),fg='#243d35',bg='#f6f5ef').pack(side='left')
        tk.Label(head,text='三类任务 / 本机 · 离线 / v'+VERSION,font=(self.font,10),fg='#68766c',bg='#f6f5ef').pack(side='right')
        body=tk.Frame(root,bg='#f6f5ef');body.pack(fill='both',expand=True,padx=24)
        shell=tk.Frame(body,bg='#e8ecdf',width=235);shell.pack(side='left',fill='y',padx=(0,16));shell.pack_propagate(False)
        self.info_button=self.button(shell,'查看计算记录',self.info,False);self.info_button.pack(side='bottom',fill='x',padx=16,pady=10)
        bar=tk.Scrollbar(shell);bar.pack(side='right',fill='y')
        self.rail_scroll=tk.Canvas(shell,bg='#e8ecdf',highlightthickness=0,yscrollcommand=bar.set);self.rail_scroll.pack(side='left',fill='both',expand=True);bar.configure(command=self.rail_scroll.yview)
        self.rail=tk.Frame(self.rail_scroll,bg='#e8ecdf');window=self.rail_scroll.create_window((0,0),window=self.rail,anchor='nw')
        self.rail.bind('<Configure>',lambda e:self.rail_scroll.configure(scrollregion=self.rail_scroll.bbox('all')));self.rail_scroll.bind('<Configure>',lambda e:self.rail_scroll.itemconfigure(window,width=e.width))
        def wheel(e):
            if self.rail_scroll.winfo_rootx()<=e.x_root<=self.rail_scroll.winfo_rootx()+self.rail_scroll.winfo_width():self.rail_scroll.yview_scroll(-1 if e.delta>0 else 1,'units')
        root.bind('<MouseWheel>',wheel)
        tk.Label(self.rail,text='选择任务',font=(self.font,15,'bold'),fg='#243d35',bg='#e8ecdf').pack(anchor='w',padx=16,pady=(18,8))
        names=[t['name'] for t in TASKS.values()];self.task_choice=tk.StringVar(value=names[0]);self.task_box=ttk.Combobox(self.rail,textvariable=self.task_choice,values=names,state='readonly',font=(self.font,11))
        self.task_box.pack(fill='x',padx=16,pady=(0,12));self.task_box.bind('<<ComboboxSelected>>',lambda e:self.change_task(next(k for k,t in TASKS.items() if t['name']==self.task_choice.get())))
        self.actions=[]
        for label,command,primary in (('选择图片',self.choose,True),('内置样例',self.show_samples,False),('开始对比',self.process,True),('保存结果',self.save,False)):
            b=self.button(self.rail,label,command,primary);b.pack(fill='x',padx=16,pady=4);self.actions.append(b)
        tk.Label(self.rail,text='01  选择任务与匹配输入\n02  开始对比\n03  两组分界线同步滑动\n04  保存图像与任务记录',justify='left',font=(self.font,10),fg='#68766c',bg='#e8ecdf').pack(anchor='w',padx=16,pady=16)
        self.task_note=tk.StringVar();tk.Label(self.rail,textvariable=self.task_note,wraplength=186,justify='left',font=(self.font,10),fg='#68766c',bg='#e8ecdf').pack(anchor='w',padx=16,pady=(0,18))
        content=tk.Frame(body,bg='white');content.pack(side='left',fill='both',expand=True)
        self.result_task=tk.StringVar(value='当前任务：低光增强 · 待处理')
        tk.Label(content,textvariable=self.result_task,anchor='w',font=(self.font,17,'bold'),fg='#243d35',bg='white').pack(fill='x',padx=18,pady=(16,5))
        self.status=tk.StringVar(value='选图或使用内置样例，确认后开始对比');self.source=tk.StringVar(value='三种任务分别使用对应官方权重和输入格式')
        status_label=tk.Label(content,textvariable=self.status,anchor='w',font=(self.font,11),fg='#243d35',bg='white');status_label.pack(fill='x',padx=18)
        source_label=tk.Label(content,textvariable=self.source,anchor='w',font=(self.font,10),fg='#68766c',bg='white');source_label.pack(fill='x',padx=18,pady=5)
        chips=tk.Frame(content,bg='white');chips.pack(fill='x',padx=14,pady=(6,0));self.chips={}
        for label in MODES:
            b=self.button(chips,label,lambda name=label:self.select(name),False);b.pack(side='left',expand=True,fill='x',padx=4);self.chips[label]=b
        self.canvas=tk.Canvas(content,bg='#eef0e7',highlightthickness=0,height=300,takefocus=True);self.canvas.pack(fill='both',expand=True,padx=18,pady=14)
        self.canvas.bind('<Configure>',self.redraw);self.canvas.bind('<B1-Motion>',self.drag);self.canvas.bind('<Button-1>',self.drag)
        self.canvas.bind('<Left>',lambda e:self.move_split(-.05));self.canvas.bind('<Right>',lambda e:self.move_split(.05))
        self.metrics=tk.StringVar(value='处理后显示本次计算耗时');metrics_label=tk.Label(content,textvariable=self.metrics,fg='#68766c',bg='white',font=(self.font,10));metrics_label.pack(fill='x',padx=18,pady=(0,12))
        content.bind('<Configure>',lambda e:[label.configure(wraplength=max(260,e.width-36)) for label in (status_label,source_label,metrics_label)])
        tk.Label(root,text='动态：两组原图对照同步滑动     静态：原图 / MobileIE / baseline 三图总览',font=(self.font,10),fg='#68766c',bg='#f6f5ef').pack(pady=10)
        self.worker_thread=threading.Thread(target=self.worker,daemon=True);self.worker_thread.start();root.after(40,self.poll);root.protocol('WM_DELETE_WINDOW',self.close);self.update_controls()

    def button(self,parent,label,command,primary):
        return self.tk.Button(parent,text=label,command=command,font=(self.font,11,'bold'),bg='#c45738' if primary else '#e8ecdf',fg='white' if primary else '#243d35',activebackground='#d7b098',relief='flat',borderwidth=0,padx=12,pady=10,cursor='hand2')
    def change_task(self,task_id):
        if self.busy or task_id not in TASKS:return
        self.active_task=task_id;self.task_choice.set(TASKS[task_id]['name']);self.prepared=self.input=None;self.images={'Input':None,'MobileIE':None,'Baseline':None};self.record={};self.split=.5
        self.result_task.set('当前任务：'+TASKS[task_id]['name']+' · 待处理');self.metrics.set('处理后显示本次计算耗时');self.choose_sample()
    def choose(self):
        kinds=[('Huawei RAW 样例文件','*.rggb')] if self.active_task=='isp' else [('照片','*.png *.jpg *.jpeg *.webp')]
        filename=self.filedialog.askopenfilename(filetypes=kinds+[('所有文件','*.*')])
        if filename:self.load(Path(filename))
    def show_samples(self):
        if self.busy:return
        menu=self.tk.Menu(self.root,tearoff=False)
        for i in range(3):menu.add_command(label=f'{i+1:02d} · '+TASKS[self.active_task]['name']+'样例',command=lambda n=i:self.choose_sample(n))
        menu.tk_popup(self.actions[1].winfo_rootx(),self.actions[1].winfo_rooty()+self.actions[1].winfo_height())
    def choose_sample(self,index=0):self.load(ASSETS/TASKS[self.active_task]['sample_assets'][index])
    def load(self,path):
        if self.busy:return
        self.busy=True;self.status.set('正在准备本任务输入…');self.update_controls();self.tasks.put(('load',(Path(path),self.active_task)))
    def process(self):
        if self.busy or self.prepared is None:return
        self.busy=True;self.status.set('正在本机处理，请稍候…');self.metrics.set('正在计算本次结果…');self.update_controls();self.tasks.put(('process',self.prepared))
    def worker(self):
        while True:
            kind,value=self.tasks.get()
            if kind=='stop':self.engines.clear();return
            try:
                if kind=='load':path,task=value;self.results.put((kind,(prepare_task(path,task),path.name)))
                elif kind=='process':
                    task=TASKS[value.task_id];start=time.perf_counter();traditional=baseline(value);baseline_ms=(time.perf_counter()-start)*1000;error=None;mie=None;init=value.task_id not in self.engines
                    try:
                        if init:self.engines[value.task_id]=Engine(value.task_id)
                        mie,_,ms=self.engines[value.task_id].run(value)
                    except Exception as exc:error=type(exc).__name__;ms=None;self.engines.pop(value.task_id,None)
                    record={'app_version':VERSION,'task_id':value.task_id,'task_name':task['name'],'model_sha256':task['model_sha256'],'input_shape':task['input_shape'],'output_shape':task['output_shape'],'native_control_inputs':[],'baseline_id':task['baseline_id'],'baseline_name':task['baseline_name'],'baseline_ms':baseline_ms,'inference_ms':ms,'initialization_in_this_run':init,'runtime':'ai-edge-litert 2.3.0, CPU builtin, 1 thread','mobileie_complete':mie is not None,'baseline_complete':True,'mobileie_error':error,'input_tensor_sha256':hashlib.sha256(value.tensor.tobytes()).hexdigest(),'input_visualization':'10-bit RAW intensity (not RGB model input)' if value.task_id=='isp' else 'shared RGB input'}
                    if value.original_bytes is not None:record['original_raw_sha256']=hashlib.sha256(value.original_bytes).hexdigest()
                    self.results.put((kind,(value,mie,traditional,record)))
            except Exception as exc:self.results.put(('error',str(exc)))
    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,value=self.results.get_nowait();self.busy=False
                if kind=='load':
                    self.prepared,name=value;self.input=self.prepared.image;self.images={'Input':self.input,'MobileIE':None,'Baseline':None};self.record={};self.split=.5;self.metrics.set('处理后显示本次计算耗时')
                    task=TASKS[self.prepared.task_id];self.result_task.set('当前任务：'+task['name']+' · 待处理');self.source.set(name+f'  ·  {self.input.width} × {self.input.height}'+(' RAW 强度可视化' if self.prepared.task_id=='isp' else ' 共同取景范围'));self.status.set('输入已准备，请确认后开始对比')
                elif kind=='process':
                    self.prepared,mie,traditional,self.record=value;self.input=self.prepared.image;self.images={'Input':self.input,'MobileIE':mie,'Baseline':traditional}
                    self.result_task.set('本次任务：'+self.record['task_name']);self.status.set('处理完成 · 两组对照使用同一输入' if mie else 'MobileIE 未生成，固定 baseline 仍可查看')
                    self.metrics.set((f"MobileIE {self.record['inference_ms']:.1f} ms" if mie else 'MobileIE 未生成')+f"  ·  {self.record['baseline_name']} {self.record['baseline_ms']:.1f} ms")
                else:self.status.set('输入未加载：'+value)
                self.update_controls();self.redraw()
        except queue.Empty:pass
        self.root.after(40,self.poll)
    def update_controls(self):
        for b in self.actions:b.configure(state='disabled' if self.busy else 'normal')
        self.task_box.configure(state='disabled' if self.busy else 'readonly');self.actions[0].configure(text='选择 RAW 文件' if self.active_task=='isp' else '选择图片')
        self.actions[2].configure(state='normal' if self.prepared is not None and not self.busy else 'disabled');self.actions[3].configure(state='normal' if self.images['Baseline'] is not None and not self.busy else 'disabled');self.info_button.configure(state='disabled' if self.busy else 'normal')
        for label,b in self.chips.items():b.configure(bg='#243d35' if label==self.mode else '#e8ecdf',fg='white' if label==self.mode else '#243d35',state='disabled' if self.busy else 'normal')
        self.task_note.set('本任务 baseline：'+TASKS[self.active_task]['baseline_name']+('\nISP 使用真实 Huawei RAW 样例或配套 .rggb 文件，普通照片不能作为 RAW。' if self.active_task=='isp' else '\n三种任务使用不同官方权重。'))
    def select(self,mode):
        if self.busy or mode not in MODES:return
        self.mode=mode;self.update_controls();self.redraw()
    def redraw(self,event=None):
        c=self.canvas;c.delete('all');self._photos=[];self.rects=[];w,h=max(c.winfo_width(),1),max(c.winfo_height(),1)
        if self.input is None:
            c.create_text(w/2,h/2,text='选择对应任务的图片或内置样例\n在同一位置，看清两种处理',justify='center',fill='#243d35',font=(self.font,15,'bold'));return
        dynamic=self.mode==MODES[0];keys=('Baseline','MobileIE') if dynamic else ('Input','MobileIE','Baseline');n=len(keys);gap=14;cell=(w-(n-1)*gap)/n
        ratio=min(cell/self.input.width,max(1,h-56)/self.input.height);iw,ih=max(1,int(self.input.width*ratio)),max(1,int(self.input.height*ratio));y=(h-ih)//2+12
        task=TASKS[self.prepared.task_id];original_label='RAW 可视化' if self.prepared.task_id=='isp' else '原图'
        for index,key in enumerate(keys):
            x=int(index*(cell+gap)+(cell-iw)/2);self.rects.append((x,y,iw,ih));label=original_label if key=='Input' else 'MobileIE' if key=='MobileIE' else task['baseline_name']
            c.create_text(x+iw/2,max(14,y-19),text=original_label+' ↔ '+label if dynamic else label,fill='#243d35',font=(self.font,11,'bold'));processed=self.images[key]
            if dynamic:
                image=self.input.copy();cut=int(image.width*self.split)
                if processed is not None:image.paste(processed.crop((cut,0,image.width,image.height)),(cut,0))
                else:ImageDraw.Draw(image).rectangle((cut,0,image.width,image.height),fill='#e8ecdf')
            else:image=processed or Image.new('RGB',self.input.size,'#e8ecdf')
            photo=self.ImageTk.PhotoImage(image.resize((iw,ih),Image.Resampling.BILINEAR));self._photos.append(photo);c.create_image(x,y,image=photo,anchor='nw')
            if processed is None and key!='Input':c.create_text(x+iw*.75 if dynamic else x+iw/2,y+ih/2,text='未生成' if self.record else '待处理',fill='#68766c',font=(self.font,11))
            if dynamic:
                xx=x+iw*self.split;c.create_line(xx,y,xx,y+ih,fill='white',width=2);c.create_oval(xx-12,y+ih/2-12,xx+12,y+ih/2+12,fill='white',outline='#dce0d4');c.create_text(xx,y+ih/2,text='↔',fill='#243d35',font=(self.font,11))
    def move_split(self,amount):
        if self.busy or self.mode!=MODES[0] or self.input is None:return
        self.split=max(0,min(1,self.split+amount));self.redraw()
    def drag(self,event):
        if self.busy or self.mode!=MODES[0] or self.input is None:return
        for x,y,w,h in self.rects:
            if x<=event.x<=x+w and y<=event.y<=y+h:self.canvas.focus_set();self.split=max(0,min(1,(event.x-x)/w));self.redraw();break
    def save(self):
        if self.busy or self.images['Baseline'] is None:return
        directory=self.filedialog.askdirectory(title='选择保存目录')
        if not directory:return
        try:target=save_results(directory,self.images,self.record,self.prepared);self.status.set('已保存：'+str(target))
        except Exception:self.status.set('保存失败，当前结果仍可查看')
    def info(self):self.messagebox.showinfo('本次计算记录',json.dumps(self.record,ensure_ascii=False,indent=2) if self.record else '选择任务与输入，处理后查看本次记录。')
    def close(self):self.closed=True;self.tasks.put(('stop',None));self._photos=[];self.root.destroy()


def main():
    import tkinter as tk
    root=tk.Tk();app=Desktop(root);root.mainloop();app.worker_thread.join(timeout=5)


if __name__=='__main__':main()
