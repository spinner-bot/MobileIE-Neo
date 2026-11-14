package dev.zrh.mobileie;

import android.content.*;
import android.database.Cursor;
import android.graphics.*;
import android.net.Uri;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.widget.TextView;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.*;
import org.json.*;
import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class) @FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class ThreeTaskInstrumentedTest {
    private final Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
    private final LightLabApplication app=(LightLabApplication)context.getApplicationContext();
    private final UiDevice device=UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());
    private ActivityScenario<MainActivity> scenario;
    private static final String PKG="dev.zrh.mobileie";
    @Before public void launch(){scenario=ActivityScenario.launch(MainActivity.class);idle();new File(context.getFilesDir(),"gates/tasks").mkdirs();}
    @After public void close(){scenario.close();}
    private void main(Runnable r){InstrumentationRegistry.getInstrumentation().runOnMainSync(r);}
    private void idle(){long end=SystemClock.uptimeMillis()+30000;do{SystemClock.sleep(50);}while(app.busy&&SystemClock.uptimeMillis()<end);assertFalse(app.status,app.busy);device.waitForIdle();}
    private void prepared(String task){main(()->app.selectTask(task));idle();main(()->app.loadSample(1));idle();assertEquals(task,app.session.task.id);}
    private void processed(String task){prepared(task);main(app::process);idle();assertNotNull(app.status,app.session.mobileie);assertNotNull(app.session.baseline);}
    private File asset(String path) throws Exception {File file=new File(context.getCacheDir(),"task-input");try(InputStream in=context.getAssets().open(path)){Files.copy(in,file.toPath(),java.nio.file.StandardCopyOption.REPLACE_EXISTING);}return file;}
    private UiObject2 find(String name) throws Exception {UiScrollable s=new UiScrollable(new UiSelector().scrollable(true));s.scrollToBeginning(8);s.scrollIntoView(new UiSelector().resourceId(PKG+":id/"+name));UiObject2 found=device.wait(Until.findObject(By.res(PKG,name)),3000);assertNotNull(name,found);return found;}
    private void shot(String name) {device.waitForIdle();assertTrue(device.takeScreenshot(new File(context.getFilesDir(),"gates/tasks/"+name+".png")));}
    private static void floats(File file,float[] values) throws Exception {ByteBuffer b=ByteBuffer.allocate(values.length*4).order(ByteOrder.LITTLE_ENDIAN);b.asFloatBuffer().put(values);Files.write(file.toPath(),b.array());}

    @Test public void aExportAllThreeTasksForIndependentNumericalReference() throws Exception {
        File dir=new File(context.getFilesDir(),"gates/tasks");JSONArray results=new JSONArray();
        for(String id:new String[]{"lowlight","underwater","isp"}){
            TaskSpec task=TaskSpec.get(context,id);
            try(MobileEngine engine=new MobileEngine(context,task)){
                for(int i=0;i<3;i++){
                    File source=asset(task.samples[i]);RawInput raw=task.raw?RawInput.read(source):null;Bitmap input=raw==null?ImagePipeline.prepare(source):raw.preview;
                    Bitmap base=raw!=null?raw.baseline():id.equals("underwater")?ImagePipeline.grayWorld(input):ImagePipeline.gamma(input);
                    MobileEngine.Output output=engine.run(input,raw==null?null:raw.tensor);String name=id+"-sample"+(i+1);
                    assertEquals(task.width(),output.bitmap.getWidth());assertEquals(task.height(),output.bitmap.getHeight());for(float f:output.raw)assertTrue(Float.isFinite(f));
                    ImagePipeline.savePng(input,new File(dir,name+"-input.png"));ImagePipeline.savePng(base,new File(dir,name+"-baseline.png"));ImagePipeline.savePng(output.bitmap,new File(dir,name+"-android.png"));floats(new File(dir,name+"-android.f32"),output.raw);
                    if(raw!=null){floats(new File(dir,name+"-input.f32"),raw.tensor);Files.copy(source.toPath(),new File(dir,name+".rggb").toPath(),java.nio.file.StandardCopyOption.REPLACE_EXISTING);}
                    results.put(new JSONObject().put("task_id",id).put("task_name",task.name).put("sample",i+1).put("model_sha256",task.modelSha).put("inference_ms",output.inferenceMs));
                    input.recycle();base.recycle();output.bitmap.recycle();
                }
            }
        }
        assertEquals(9,results.length());Files.write(new File(dir,"cases.json").toPath(),results.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }
    @Test public void bRawPackingIsLosslessAndRgbIsRejected() throws Exception {
        TaskSpec task=TaskSpec.get(context,"isp");File file=asset(task.samples[0]);byte[] bytes=Files.readAllBytes(file.toPath());RawInput raw=RawInput.read(file);ByteBuffer source=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for(int y:new int[]{0,5,100})for(int x:new int[]{0,11,200})for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){
            int value=source.getShort(RawInput.HEADER+2*((2*y+dy)*448+2*x+dx))&65535;assertEquals(value/4095f,raw.tensor[(y*224+x)*4+dy*2+dx],0);
        }
        file=asset("samples/01.png");try{RawInput.read(file);fail("RGB must not become RAW");}catch(IOException expected){}
        File bad=asset(task.samples[0]);byte[] broken=Files.readAllBytes(bad.toPath());broken[0]=0;Files.write(bad.toPath(),broken);try{RawInput.read(bad);fail("Bad header");}catch(IOException expected){}
    }
    @Test public void cTaskSwitchClearsOldResultsAndHeaderTracksActualJob() throws Exception {
        processed("lowlight");scenario.onActivity(a->{app.selectTask("underwater");assertNull(app.session);assertFalse(a.findViewById(R.id.process).isEnabled());assertTrue(((TextView)a.findViewById(R.id.result_task)).getText().toString().contains("水下增强"));});idle();
        assertNull(app.session.mobileie);main(app::process);idle();scenario.onActivity(a->assertEquals("本次任务：水下增强",((TextView)a.findViewById(R.id.result_task)).getText().toString()));
        processed("isp");scenario.onActivity(a->{assertEquals("本次任务：RAW→RGB ISP",((TextView)a.findViewById(R.id.result_task)).getText().toString());assertFalse(a.findViewById(R.id.capture).isEnabled());assertEquals("导入 RAW",((TextView)a.findViewById(R.id.pick)).getText().toString());});assertEquals("isp",app.session.metadata.getString("task_id"));
    }
    @Test public void dBothPanesMoveTogetherAndStaticIsIndependent() throws Exception {
        for(String id:new String[]{"lowlight","underwater","isp"}){
            processed(id);find("display_dynamic").click();find("comparison");
            for(int pane=0;pane<2;pane++){
                float target=pane==0?.22f:.78f;int selected=pane;int[] swipe=new int[4];
                scenario.onActivity(a->{ComparisonView v=a.findViewById(R.id.comparison);RectF r=v.paneBounds(selected);int[] location=new int[2];v.getLocationOnScreen(location);swipe[0]=location[0]+(int)(r.left+r.width()*.5f);swipe[1]=location[1]+(int)r.centerY();swipe[2]=location[0]+(int)(r.left+r.width()*target);swipe[3]=swipe[1];});
                device.swipe(swipe[0],swipe[1],swipe[2],swipe[3],12);scenario.onActivity(a->{ComparisonView v=a.findViewById(R.id.comparison);assertEquals(target,v.position(),.04);assertEquals(v.paneBounds(0).width(),v.paneBounds(1).width(),.01);});
            }
            shot(id+"-dynamic");find("display_static").click();scenario.onActivity(a->assertEquals(ComparisonView.STATIC,((ComparisonView)a.findViewById(R.id.comparison)).displayMode()));shot(id+"-static");
        }
    }
    @Test public void eInvalidRawKeepsTheCorrectPreviousTask() throws Exception {
        processed("isp");String previous=app.session.id;File rgb=asset("samples/01.png");main(()->app.load(Uri.fromFile(rgb),"invalid RGB"));idle();assertEquals(previous,app.session.id);assertEquals("isp",app.session.task.id);assertTrue(app.status.contains("图片未加载"));scenario.onActivity(a->assertEquals("本次任务：RAW→RGB ISP",((TextView)a.findViewById(R.id.result_task)).getText().toString()));
    }
    @Test public void fRawAndTaskRecordSurviveMediaStoreRoundtrip() throws Exception {
        processed("isp");main(app::save);idle();assertTrue(app.status,app.status.startsWith("已保存"));String prefix="MobileIE_"+app.session.id.substring(0,8)+"_%";
        try(Cursor c=context.getContentResolver().query(MediaStore.Downloads.EXTERNAL_CONTENT_URI,new String[]{"_id","_display_name","is_pending"},"_display_name LIKE ?",new String[]{prefix},null)){
            assertNotNull(c);assertEquals(2,c.getCount());while(c.moveToNext()){
                assertEquals(0,c.getInt(2));try(InputStream in=context.getContentResolver().openInputStream(ContentUris.withAppendedId(MediaStore.Downloads.EXTERNAL_CONTENT_URI,c.getLong(0)));ByteArrayOutputStream out=new ByteArrayOutputStream()){
                    byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);
                    if(c.getString(1).endsWith(".rggb"))assertArrayEquals(Files.readAllBytes(new File(app.session.directory,"source.image").toPath()),out.toByteArray());
                    else {JSONObject j=new JSONObject(out.toString("UTF-8"));assertEquals("isp",j.getString("task_id"));assertEquals("RAW→RGB ISP",j.getString("task_name"));assertEquals(448,j.getJSONArray("output_shape").getInt(1));assertFalse(j.has("gamma_exponent"));}
                }
            }
        }
        Bitmap comparison=ResultExporter.comparison(app.session);assertEquals(1344,comparison.getWidth());assertEquals(558,comparison.getHeight());assertEquals(app.session.input.getPixel(200,200),comparison.getPixel(200,276));comparison.recycle();
    }
    @Test public void gRecreateKeepsTaskDisplayAndSplit() throws Exception {
        processed("underwater");String id=app.session.id;scenario.onActivity(a->{((ComparisonView)a.findViewById(R.id.comparison)).position(.73f);a.findViewById(R.id.display_static).performClick();});scenario.recreate();idle();assertEquals(id,app.session.id);
        scenario.onActivity(a->{assertEquals("本次任务：水下增强",((TextView)a.findViewById(R.id.result_task)).getText().toString());ComparisonView v=a.findViewById(R.id.comparison);assertEquals(ComparisonView.STATIC,v.displayMode());assertEquals(.73f,v.position(),.001);});
    }
    @Test public void hActualSystemPickerImportsRawWithoutRgbConversion() throws Exception {
        prepared("isp");ContentValues v=new ContentValues();v.put(MediaStore.MediaColumns.DISPLAY_NAME,"MobileIE_raw_gate.rggb");v.put(MediaStore.MediaColumns.MIME_TYPE,"application/octet-stream");v.put(MediaStore.MediaColumns.RELATIVE_PATH,"Download");Uri uri=context.getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI,v);assertNotNull(uri);
        try{
            try(InputStream in=context.getAssets().open("samples/isp/01.rggb");OutputStream out=context.getContentResolver().openOutputStream(uri)){byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);}
            find("pick").click();assertTrue(device.wait(Until.hasObject(By.pkg("com.google.android.documentsui")),7000));UiObject2 list=device.findObject(By.desc("List view"));if(list!=null)list.click();
            UiObject2 item=device.wait(Until.findObject(By.textContains("MobileIE_raw_gate")),4000);
            if(item==null){UiObject2 drawer=device.findObject(By.desc("Show roots"));if(drawer!=null)drawer.click();UiObject2 downloads=device.wait(Until.findObject(By.text("Downloads")),3000);if(downloads!=null)downloads.click();item=device.wait(Until.findObject(By.textContains("MobileIE_raw_gate")),5000);}
            assertNotNull("RAW file in system picker",item);item.click();assertTrue(device.wait(Until.hasObject(By.pkg(PKG)),5000));idle();assertEquals("导入 RAW",app.session.label);assertNotNull(app.session.raw);main(app::process);idle();assertNotNull(app.session.mobileie);
        }finally{context.getContentResolver().delete(uri,null,null);}
    }
    @Test public void iThreeNativeImageInputsAndFixedBaselines() throws Exception {
        for(String id:new String[]{"lowlight","underwater","isp"})try(MobileEngine engine=new MobileEngine(context,TaskSpec.get(context,id))){assertEquals(id,engine.task.id);}
        Bitmap b=Bitmap.createBitmap(6,8,Bitmap.Config.ARGB_8888);b.eraseColor(Color.rgb(40,80,120));Bitmap result=ImagePipeline.grayWorld(b);int p=result.getPixel(2,2);assertTrue(Math.abs(Color.red(p)-80)<=1);assertTrue(Math.abs(Color.green(p)-80)<=1);assertTrue(Math.abs(Color.blue(p)-80)<=1);b.recycle();result.recycle();
    }
}
