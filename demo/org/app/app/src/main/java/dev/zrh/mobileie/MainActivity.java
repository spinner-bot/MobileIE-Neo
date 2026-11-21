package dev.zrh.mobileie;

import android.app.*;
import android.content.*;
import android.content.res.ColorStateList;
import android.graphics.*;
import android.graphics.drawable.*;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.*;
import android.widget.*;
import androidx.core.content.FileProvider;
import androidx.core.view.WindowInsetsCompat;
import java.io.*;
import java.util.*;

public final class MainActivity extends Activity {
    static final int CAMERA=41, PICK=42;
    private static final int INK=0xff243d35, MUTED=0xff68766c, ACCENT=0xffc45738, BACKGROUND=0xfff6f5ef;
    private LightLabApplication app;
    private final Runnable observer = this::render;
    private ComparisonView comparison;
    private TextView status, source, metrics, process, save, camera, pick, sample, displayDynamic, displayStatic, hint, resultTask, taskNote;
    private final java.util.List<TextView> taskButtons=new java.util.ArrayList<>();
    private ProgressBar progress;
    private LinearLayout modes;
    private int mode;
    private File pendingCapture;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        app=(LightLabApplication)getApplication();
        if(state!=null) {
            mode=state.getInt("display_mode",ComparisonView.DYNAMIC)==ComparisonView.STATIC?ComparisonView.STATIC:ComparisonView.DYNAMIC;
            String pending=state.getString("pendingCapture");
            if(pending!=null) {
                File candidate=new File(pending);
                try { if(candidate.getCanonicalPath().startsWith(new File(getFilesDir(),"captures").getCanonicalPath()+File.separator)) pendingCapture=candidate; }
                catch(IOException ignored) {}
            }
        }
        createUi();
        if(state!=null) comparison.position(state.getFloat("split",.5f));
    }

    // A newer Activity can replace the process's observer. Rebind on every return
    // so this page receives both the current snapshot and later worker results.
    @Override protected void onResume() { super.onResume(); app.attach(observer); }
    @Override protected void onPause() { app.detach(observer); super.onPause(); }

    private void createUi() {
        ScrollView scroll=new ScrollView(this); scroll.setFillViewport(true); scroll.setBackgroundColor(BACKGROUND);
        LinearLayout body=new LinearLayout(this); body.setOrientation(LinearLayout.VERTICAL); body.setPadding(dp(20),dp(18),dp(20),dp(24));
        scroll.addView(body,new ScrollView.LayoutParams(-1,-2));
        scroll.setOnApplyWindowInsetsListener((view,insets)-> {
            androidx.core.graphics.Insets bars=WindowInsetsCompat.toWindowInsetsCompat(insets,view).getInsets(WindowInsetsCompat.Type.systemBars()|WindowInsetsCompat.Type.displayCutout());
            view.setPadding(bars.left,bars.top,bars.right,bars.bottom); return insets;
        });
        setContentView(scroll); scroll.requestApplyInsets();

        LinearLayout header=row();
        TextView logo=text("◒",28,INK,true); logo.setGravity(Gravity.CENTER); logo.setBackground(round(0xffe6dcc3,14));
        header.addView(logo,new LinearLayout.LayoutParams(dp(48),dp(48)));
        LinearLayout titles=new LinearLayout(this); titles.setOrientation(LinearLayout.VERTICAL); titles.setPadding(dp(12),0,0,0);
        titles.addView(text("微光实验室",22,INK,true)); titles.addView(text("MOBILEIE  /  LIGHT LAB",10,MUTED,false));
        header.addView(titles,new LinearLayout.LayoutParams(0,-2,1));
        TextView local=text("本机 · 离线",11,INK,true); local.setPadding(dp(10),dp(8),dp(10),dp(8)); local.setBackground(round(0xffe5eadf,20)); header.addView(local);
        add(body,header,0);
        add(body,text("三类任务，同屏看清两种处理。",17,INK,true),18);
        LinearLayout taskRow=row();String[] ids={"lowlight","underwater","isp"};String[] names={"低光增强","水下增强","RAW→RGB ISP"};int[] viewIds={R.id.task_lowlight,R.id.task_underwater,R.id.task_isp};
        for(int i=0;i<3;i++){String id=ids[i];TextView button=button(names[i],viewIds[i],false);button.setTextSize(11);button.setTag(id);button.setOnClickListener(v->app.selectTask(id));taskButtons.add(button);weighted(taskRow,button,i==0?0:6);}add(body,taskRow,12);

        LinearLayout inputs=row();
        camera=button("现场拍照",R.id.capture,true); camera.setOnClickListener(v->capture());
        pick=button("相册选图",R.id.pick,false); pick.setOnClickListener(v->pick());
        sample=button("内置样例",R.id.sample,false); sample.setOnClickListener(v->{String title=app.activeTask.name;new AlertDialog.Builder(this).setTitle("选择"+title+"样例")
            .setItems(app.activeTask.id.equals("lowlight")?new String[]{"01 · 书柜","02 · 暗光样例","03 · 仓库样例"}:new String[]{"01 · "+title+"样例","02 · "+title+"样例","03 · "+title+"样例"},(d,index)->app.loadSample(index+1)).show();});
        weighted(inputs,camera,0); weighted(inputs,pick,8); weighted(inputs,sample,8); add(body,inputs,20);

        LinearLayout panel=new LinearLayout(this); panel.setOrientation(LinearLayout.VERTICAL); panel.setPadding(dp(12),dp(14),dp(12),dp(14)); panel.setBackground(round(Color.WHITE,20));
        resultTask=text("当前任务：低光增强 · 待处理",16,INK,true);resultTask.setId(R.id.result_task);add(panel,resultTask,0);
        LinearLayout stateRow=row(); progress=new ProgressBar(this); progress.setIndeterminateTintList(ColorStateList.valueOf(ACCENT));
        stateRow.addView(progress,new LinearLayout.LayoutParams(dp(18),dp(18)));
        status=text("准备好照片",12,INK,true); status.setId(R.id.status); status.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        LinearLayout.LayoutParams statusParams=new LinearLayout.LayoutParams(0,-2,1); statusParams.leftMargin=dp(7); stateRow.addView(status,statusParams); panel.addView(stateRow);
        source=text("600 × 400 处理范围 · 原图与结果保持同一取景",11,MUTED,false); add(panel,source,7);
        comparison=new ComparisonView(this); comparison.setId(R.id.comparison); add(panel,comparison,12);
        modes=row(); displayDynamic=modeButton("动态对比",R.id.display_dynamic,ComparisonView.DYNAMIC); displayStatic=modeButton("静态总览",R.id.display_static,ComparisonView.STATIC);
        weighted(modes,displayDynamic,0);weighted(modes,displayStatic,6);add(panel,modes,12);
        hint=text("选择照片并处理后，可拖动对比同一位置",11,MUTED,false); hint.setGravity(Gravity.CENTER); add(panel,hint,8);
        process=button("开始对比",R.id.process,true); process.setOnClickListener(v->app.process()); add(panel,process,14); add(body,panel,16);

        LinearLayout detail=new LinearLayout(this); detail.setOrientation(LinearLayout.VERTICAL); detail.setPadding(dp(16),dp(16),dp(16),dp(16)); detail.setBackground(round(0xffe8ecdf,16));
        detail.addView(text("两条方法，一份任务输入",15,INK,true));
        taskNote=text("",12,MUTED,false);add(detail,taskNote,7);
        metrics=text("处理后显示本次耗时，点击可查看计算记录",12,INK,false); metrics.setId(R.id.metrics); metrics.setPadding(0,dp(8),0,dp(8)); metrics.setOnClickListener(v->showRecord()); add(detail,metrics,3); add(body,detail,14);

        LinearLayout actions=row(); save=button("保存对比结果",R.id.save,true); save.setOnClickListener(v->app.save());
        TextView benchmark=button("AI Benchmark ↗",R.id.benchmark,false); benchmark.setOnClickListener(v->benchmark());
        weighted(actions,save,0); weighted(actions,benchmark,8); add(body,actions,16);
        TextView about=text("关于模型与演示  ·  v"+BuildConfig.VERSION_NAME,11,MUTED,false); about.setGravity(Gravity.CENTER); about.setPadding(0,dp(12),0,dp(12));
        about.setOnClickListener(v->new AlertDialog.Builder(this).setTitle("微光实验室")
            .setMessage("三种任务分别使用官方训练权重：低光增强、水下增强、RAW→RGB ISP。\n\n低光和水下使用 600×400 RGB；ISP 使用配套真实 Huawei RAW，四通道输入、448×448 RGB 输出。RAW 视图仅作强度可视化，普通相机照片不是 RAW。\n\n动态模式两条分界线同步，静态模式为三图总览。所有处理均在本机完成。\n\nAI Benchmark 独立测速，需要手动选择相应模型，不能与本 App 计时混用。\n\n固定 baseline 不是最强传统方法，不保证模型对所有照片更好。MobileIE 上游许可：Apache-2.0。\n\n当前任务："+app.activeTask.name+"\n模型 SHA-256：\n"+app.activeTask.modelSha)
            .setPositiveButton("知道了",null).show()); add(body,about,4);
    }

    private void render() {
        if(comparison==null) return;
        LightLabApplication.Session s=app.session; boolean loaded=s!=null, done=loaded&&s.baseline!=null;TaskSpec task=loaded?s.task:app.activeTask;
        status.setText(app.status); progress.setVisibility(app.busy?View.VISIBLE:View.GONE);
        resultTask.setText((done?"本次任务：":"当前任务：")+task.name+(done?"":" · 待处理"));
        source.setText(loaded?s.label+" · "+task.width()+" × "+task.height()+(task.raw?" RAW 强度可视化":" 共同取景范围"):task.raw?"请导入配套 .rggb 或使用真实 RAW 样例":"选择照片或对应任务样例");
        comparison.images(loaded?s.input:null,loaded?s.mobileie:null,loaded?s.baseline:null,mode,task);
        camera.setEnabled(!app.busy&&!task.raw);pick.setEnabled(!app.busy);sample.setEnabled(!app.busy);pick.setText(task.raw?"导入 RAW":"相册选图");
        for(TextView b:taskButtons){b.setEnabled(!app.busy);select(b,b.getTag().equals(app.activeTask.id));}
        process.setEnabled(loaded&&!app.busy); process.setText(app.busy?"处理中…":done?"重新处理":"开始对比");
        save.setEnabled(done&&!app.busy);modes.setVisibility(View.VISIBLE);
        displayDynamic.setEnabled(!app.busy);displayStatic.setEnabled(!app.busy);
        displayDynamic.setContentDescription(app.busy?"任务进行中，暂不可切换":"动态并列对比，两窗口同步滑动");
        comparison.setEnabled(!app.busy); metrics.setEnabled(!app.busy);
        hint.setText(mode==ComparisonView.STATIC?"独立三图总览：原图 / MobileIE / baseline":"拖动任一窗口，两条分界线同步移动");
        select(displayDynamic,mode==ComparisonView.DYNAMIC);select(displayStatic,mode==ComparisonView.STATIC);
        taskNote.setText("MobileIE："+task.name+"官方权重\nbaseline："+task.baselineName+(task.raw?"\n本任务仅接收配套 RAW，系统相机照片可用于低光与水下任务。":"\n同一输入与坐标，比较细节与颜色。"));
        if(app.busy) metrics.setText("任务进行中，完成后显示本次耗时");
        else if(done) metrics.setText(String.format(Locale.CHINA,"MobileIE %s   ·   baseline %.1f ms\n点击查看本次任务与分项计时",s.mobileie==null?"未生成":String.format(Locale.CHINA,"%.1f ms",s.metadata.optDouble("inference_ms")),s.metadata.optDouble("baseline_ms",s.metadata.optDouble("gamma_ms"))));
        else metrics.setText("处理后显示本次耗时，点击可查看计算记录");
        for(TextView b:new TextView[]{camera,pick,sample,process,save}) b.setAlpha(b.isEnabled()?1f:.45f);
    }

    private void capture() {
        if(app.busy) return;
        if(app.activeTask.raw){app.message("ISP 请使用配套 RAW，普通相机照片不能作为 RAW");return;}
        try {
            File directory=new File(getFilesDir(),"captures"); if(!directory.exists()&&!directory.mkdirs()) throw new IOException("无法创建拍照文件");
            pendingCapture=File.createTempFile("capture-",".jpg",directory);
            Uri uri=FileProvider.getUriForFile(this,getPackageName()+".files",pendingCapture);
            Intent intent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE).putExtra(MediaStore.EXTRA_OUTPUT,uri);
            intent.setClipData(ClipData.newRawUri("photo-output",uri));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            startActivityForResult(intent,CAMERA);
        } catch(Exception e) { if(pendingCapture!=null) pendingCapture.delete(); pendingCapture=null; app.message("无法打开相机，请使用相册或内置样例"); }
    }
    private void pick() {
        Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT).setType(app.activeTask.raw?"*/*":"image/*").addCategory(Intent.CATEGORY_OPENABLE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        try { startActivityForResult(intent,PICK); }
        catch(ActivityNotFoundException e) { app.message("没有可用的图片选择器，请使用拍照或内置样例"); }
    }
    @Override protected void onActivityResult(int request,int result,Intent data) {
        super.onActivityResult(request,result,data);
        if(request==CAMERA) {
            File captured=pendingCapture; pendingCapture=null;
            if(result==RESULT_OK && captured!=null && captured.length()>0) {
                Uri uri=FileProvider.getUriForFile(this,getPackageName()+".files",captured); app.load(uri,"现场拍摄");
            } else { if(captured!=null) captured.delete(); app.message(result==RESULT_OK?"相机未返回完整照片，请重新拍摄；此前结果已保留":"已取消拍照，保留此前结果"); }
        } else if(request==PICK) {
            if(result==RESULT_OK && data!=null && data.getData()!=null) {
                Uri uri=data.getData();
                try { getContentResolver().takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION); } catch(SecurityException ignored) {}
                app.load(uri,app.activeTask.raw?"导入 RAW":"相册照片");
            } else app.message("已取消选图，保留此前结果");
        }
    }
    private void benchmark() {
        try {
            Intent intent=getPackageManager().getLaunchIntentForPackage("org.benchmark.demo");
            if(intent==null) { app.message("未安装 AI Benchmark；图片对比仍可独立使用"); return; }
            startActivity(intent);
        } catch(ActivityNotFoundException | SecurityException e) { app.message("无法打开 AI Benchmark，请手动打开"); }
    }
    private void showRecord() {
        if(app.session==null) return;
        try { new AlertDialog.Builder(this).setTitle("本次计算记录").setMessage(app.session.metadata.toString(2)+"\n\nMobileIE 推理仅计解释器调用；Gamma 仅计像素变换。总处理时间不包含拍照、准备、保存或屏幕绘制，不能直接作为速度排名。").setPositiveButton("关闭",null).show(); }
        catch(org.json.JSONException ignored) {}
    }
    @Override protected void onSaveInstanceState(Bundle state) {
        super.onSaveInstanceState(state); state.putInt("display_mode",mode); state.putFloat("split",comparison.position());
        if(pendingCapture!=null) state.putString("pendingCapture",pendingCapture.getPath());
    }
    @Override protected void onDestroy() { app.detach(observer); super.onDestroy(); }
    private TextView modeButton(String label,int id,int index) { TextView b=button(label,id,false); b.setTextSize(12); b.setOnClickListener(v->{mode=index;render();}); return b; }
    private void select(TextView v,boolean active) { v.setBackground(round(active?INK:0xfff0f1ea,10)); v.setTextColor(active?Color.WHITE:INK); v.setSelected(active); }
    private TextView button(String label,int id,boolean primary) {
        TextView v=text(label,14,primary?Color.WHITE:INK,true); v.setId(id); v.setGravity(Gravity.CENTER); v.setMinHeight(dp(48)); v.setPadding(dp(8),dp(12),dp(8),dp(12));
        v.setBackground(new RippleDrawable(ColorStateList.valueOf(primary?0xffe6aa91:0xffd4ded0),round(primary?ACCENT:0xffe9eade,13),null));
        v.setClickable(true); v.setFocusable(true); v.setAccessibilityDelegate(new View.AccessibilityDelegate(){ @Override public void onInitializeAccessibilityNodeInfo(View host,android.view.accessibility.AccessibilityNodeInfo info){super.onInitializeAccessibilityNodeInfo(host,info);info.setClassName("android.widget.Button");} }); return v;
    }
    private TextView text(String value,int size,int color,boolean bold) { TextView v=new TextView(this); v.setText(value); v.setTextSize(size); v.setTextColor(color); v.setIncludeFontPadding(false); if(bold)v.setTypeface(Typeface.create("sans-serif-medium",Typeface.NORMAL)); return v; }
    private GradientDrawable round(int color,int radius) { GradientDrawable d=new GradientDrawable(); d.setColor(color); d.setCornerRadius(dp(radius)); return d; }
    private LinearLayout row() { LinearLayout r=new LinearLayout(this); r.setOrientation(LinearLayout.HORIZONTAL); r.setGravity(Gravity.CENTER_VERTICAL); return r; }
    private void weighted(LinearLayout parent,View child,int leftMargin) { LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(0,-2,1); p.leftMargin=dp(leftMargin); parent.addView(child,p); }
    private void add(LinearLayout parent,View child,int top) { LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,-2); p.topMargin=dp(top); parent.addView(child,p); }
    private int dp(int value) { return Math.round(value*getResources().getDisplayMetrics().density); }
}
