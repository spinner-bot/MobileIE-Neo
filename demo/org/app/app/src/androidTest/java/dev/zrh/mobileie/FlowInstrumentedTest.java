package dev.zrh.mobileie;

import android.app.Activity;
import android.content.*;
import android.database.Cursor;
import android.graphics.*;
import android.net.Uri;
import android.os.SystemClock;
import android.provider.MediaStore;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.*;
import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import static org.junit.Assert.*;
import java.io.*;
import java.lang.reflect.Field;
import java.util.regex.Pattern;

@RunWith(AndroidJUnit4.class) @FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class FlowInstrumentedTest {
    private final Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
    private final LightLabApplication app=(LightLabApplication)context.getApplicationContext();
    private final UiDevice device=UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());
    private ActivityScenario<MainActivity> scenario;
    private static final String PKG="dev.zrh.mobileie";
    @Before public void launch() { new File(context.getFilesDir(),"gates").mkdirs(); scenario=ActivityScenario.launch(MainActivity.class); idle(); }
    @After public void close() { if(scenario!=null) scenario.close(); }
    private void main(Runnable r) { InstrumentationRegistry.getInstrumentation().runOnMainSync(r); }
    private void idle() { long end=SystemClock.uptimeMillis()+30000; do {SystemClock.sleep(100);}while(app.busy&&SystemClock.uptimeMillis()<end); assertFalse("worker did not finish: "+app.status,app.busy); device.waitForIdle(); }
    private UiObject2 find(String id) throws Exception {
        UiScrollable scroll=new UiScrollable(new UiSelector().scrollable(true));
        scroll.scrollToBeginning(8);
        scroll.scrollIntoView(new UiSelector().resourceId(PKG+":id/"+id));
        UiObject2 object=device.wait(Until.findObject(By.res(PKG,id)),3000); assertNotNull(id,object); return object;
    }
    private void prepared(int index) { main(()->app.loadSample(index)); idle(); assertNotNull(app.session); }
    private void processed(int index) { prepared(index); main(()->app.process()); idle(); assertNotNull("MobileIE: "+app.status,app.session.mobileie); assertNotNull(app.session.gamma); }
    private void shot(String name) {
        device.waitForIdle(); SystemClock.sleep(200);
        File directory=new File(context.getFilesDir(),"gates"); directory.mkdirs();
        assertTrue(device.takeScreenshot(new File(directory,name+".png")));
    }

    @Test public void aUiSampleProcessCompareSaveAndNoNetworkPermission() throws Exception {
        find("sample").click(); device.wait(Until.findObject(By.text("01 · 书柜")),3000).click(); idle();
        assertNull(app.session.mobileie); assertNull(app.session.gamma);
        find("process").click(); idle(); assertNotNull(app.session.mobileie); assertNotNull(app.session.gamma);
        find("mode_gamma").click();
        scenario.onActivity(a->assertTrue(a.findViewById(R.id.mode_gamma).isSelected()));
        UiObject2 image=find("comparison"); android.graphics.Rect bounds=image.getVisibleBounds();
        device.swipe(bounds.left+bounds.width()/4,bounds.centerY(),bounds.left+bounds.width()*4/5,bounds.centerY(),12);
        scenario.onActivity(a->assertTrue(((ComparisonView)a.findViewById(R.id.comparison)).position()>.7f));
        find("mode_mobileie").click(); device.waitForIdle();
        scenario.onActivity(a->assertTrue(a.findViewById(R.id.mode_mobileie).isSelected())); shot("02-mobileie-comparison");
        find("mode_overview").click();
        assertNotNull(device.wait(Until.findObject(By.res(PKG,"mode_overview").selected(true)),3000));
        shot("03-three-method-overview");
        find("save").click(); idle(); assertTrue(app.status,app.status.startsWith("已保存"));
        int images=0;
        String prefix="MobileIE_"+app.session.id.substring(0,8)+"_%";
        try(Cursor c=context.getContentResolver().query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,new String[]{"_id","_display_name","is_pending"},"_display_name LIKE ?",new String[]{prefix},null)) {
            assertNotNull(c); images=c.getCount();
            while(c.moveToNext()) {
                assertEquals(0,c.getInt(2));
                Uri saved=ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,c.getLong(0));
                try(InputStream in=context.getContentResolver().openInputStream(saved)) {
                    Bitmap decoded=BitmapFactory.decodeStream(in); assertNotNull(decoded);
                    String name=c.getString(1);
                    if(name.endsWith("_comparison.png")) { assertEquals(1800,decoded.getWidth());assertEquals(478,decoded.getHeight());assertEquals(app.session.input.getPixel(300,200),decoded.getPixel(300,244)); }
                    else {
                        Bitmap expected=name.endsWith("_input.png")?app.session.input:name.endsWith("_mobileie.png")?app.session.mobileie:app.session.gamma;
                        assertArrayEquals(LightLabApplication.bitmapBytes(expected),LightLabApplication.bitmapBytes(decoded));
                    }
                    decoded.recycle();
                }
            }
        }
        assertEquals(4,images);
        try(Cursor c=context.getContentResolver().query(MediaStore.Downloads.EXTERNAL_CONTENT_URI,new String[]{"_id","is_pending"},"_display_name LIKE ?",new String[]{prefix},null)) {
            assertNotNull(c);assertEquals(1,c.getCount());assertTrue(c.moveToFirst());assertEquals(0,c.getInt(1));
            try(InputStream in=context.getContentResolver().openInputStream(ContentUris.withAppendedId(MediaStore.Downloads.EXTERNAL_CONTENT_URI,c.getLong(0)));ByteArrayOutputStream bytes=new ByteArrayOutputStream()) {
                byte[] buffer=new byte[8192];int n;while((n=in.read(buffer))!=-1)bytes.write(buffer,0,n);
                org.json.JSONObject record=new org.json.JSONObject(bytes.toString("UTF-8"));
                assertEquals(MobileEngine.MODEL_SHA,record.getString("model_sha256"));assertEquals(.5,record.getDouble("gamma_exponent"),0);
                assertTrue(record.getBoolean("mobileie_complete"));assertTrue(record.getBoolean("gamma_complete"));
            }
        }
        String[] permissions=context.getPackageManager().getPackageInfo(PKG,android.content.pm.PackageManager.GET_PERMISSIONS).requestedPermissions;
        if(permissions!=null) for(String permission:permissions) { assertNotEquals("android.permission.INTERNET",permission); assertNotEquals("android.permission.CAMERA",permission); }
        find("benchmark").click(); assertTrue(app.status,app.status.contains("未安装 AI Benchmark")); shot("04-benchmark-unavailable");
    }

    @Test public void bRecreationPreservesResultAndSlider() throws Exception {
        processed(2); String id=app.session.id;
        scenario.onActivity(a->{ ((ComparisonView)a.findViewById(R.id.comparison)).position(.73f); a.findViewById(R.id.mode_gamma).performClick(); });
        scenario.recreate(); idle(); assertEquals(id,app.session.id);
        scenario.onActivity(a->{assertEquals(.73f,((ComparisonView)a.findViewById(R.id.comparison)).position(),.001f); assertTrue(a.findViewById(R.id.mode_gamma).isSelected());});
        assertNotNull(app.session.mobileie); assertNotNull(app.session.gamma);
    }

    @Test public void cCameraAndPickerCancelKeepResult() throws Exception {
        processed(1); String id=app.session.id;
        find("capture").click(); assertTrue(device.wait(Until.hasObject(By.pkg("com.android.camera2")),10000));
        device.pressBack(); device.wait(Until.hasObject(By.pkg(PKG)),5000);
        if(!device.hasObject(By.pkg(PKG))) device.pressBack();
        assertTrue(device.wait(Until.hasObject(By.pkg(PKG)),5000)); idle();
        assertEquals(id,app.session.id); assertNotNull(app.session.mobileie);
        find("pick").click(); assertTrue(device.wait(Until.hasObject(By.pkg("com.google.android.documentsui")),7000));
        device.pressBack(); assertTrue(device.wait(Until.hasObject(By.pkg(PKG)),5000)); idle();
        assertEquals(id,app.session.id); assertNotNull(app.session.mobileie);
    }

    @Test public void dSystemCameraReturnsFullImageAndCanProcess() throws Exception {
        find("capture").click(); assertTrue(device.wait(Until.hasObject(By.pkg("com.android.camera2")),10000));
        device.dumpWindowHierarchy(new File(context.getFilesDir(),"gates/camera-before.xml"));
        UiObject2 shutter=device.wait(Until.findObject(By.res("com.android.camera2","shutter_button")),10000);
        if(shutter==null) shutter=device.findObject(By.desc(Pattern.compile("(?i).*(shutter|capture photo|take photo).*")));
        assertNotNull("Camera shutter",shutter); shutter.click(); SystemClock.sleep(1800);
        device.dumpWindowHierarchy(new File(context.getFilesDir(),"gates/camera-review.xml"));
        UiObject2 done=device.wait(Until.findObject(By.res(Pattern.compile(".*:id/(done_button|btn_done|review_done|done)"))),5000);
        if(done==null) done=device.findObject(By.desc(Pattern.compile("(?i)(done|完成|确定|ok)")));
        if(done==null) done=device.findObject(By.text(Pattern.compile("(?i)(done|完成|确定|ok)")));
        if(done!=null) done.click();
        assertTrue("Camera did not return",device.wait(Until.hasObject(By.pkg(PKG)),10000)); idle();
        assertEquals("现场拍摄",app.session.label); assertNull(app.session.mobileie);
        BitmapFactory.Options o=new BitmapFactory.Options(); o.inJustDecodeBounds=true; BitmapFactory.decodeFile(new File(app.session.directory,"source.image").getPath(),o);
        assertTrue("Must not use thumbnail",o.outWidth>=600&&o.outHeight>=400);
        main(()->app.process()); idle(); assertNotNull(app.session.mobileie); shot("05-captured-photo-result");
    }

    @Test public void eDocumentPickerSelectsRealImage() throws Exception {
        ContentValues v=new ContentValues(); v.put(MediaStore.Images.Media.DISPLAY_NAME,"MobileIE_picker_gate.png"); v.put(MediaStore.Images.Media.MIME_TYPE,"image/png"); v.put(MediaStore.Images.Media.RELATIVE_PATH,"Pictures/MobileIE");
        Uri uri=context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,v); assertNotNull(uri);
        try(InputStream in=context.getAssets().open("samples/01.png");OutputStream out=context.getContentResolver().openOutputStream(uri)) { byte[] buffer=new byte[8192]; int n; while((n=in.read(buffer))!=-1) out.write(buffer,0,n); }
        find("pick").click(); assertTrue(device.wait(Until.hasObject(By.pkg("com.google.android.documentsui")),7000));
        UiObject2 list=device.findObject(By.desc("List view")); if(list!=null) { list.click(); device.waitForIdle(); }
        device.dumpWindowHierarchy(new File(context.getFilesDir(),"gates/picker.xml"));
        UiObject2 photo=device.wait(Until.findObject(By.textContains("MobileIE_picker_gate")),7000);
        if(photo==null) photo=device.findObject(By.desc(Pattern.compile("^MobileIE_picker_gate.*")));
        assertNotNull("Picker image",photo); photo.click();
        assertTrue(device.wait(Until.hasObject(By.pkg(PKG)),5000)); idle(); assertEquals("相册照片",app.session.label);
        main(()->app.process()); idle(); assertNotNull(app.session.mobileie);
        context.getContentResolver().delete(uri,null,null);
    }

    @Test public void fBadInputAndDoubleTapDoNotCreateFalseResults() throws Exception {
        processed(1); String id=app.session.id;
        File invalid=new File(context.getCacheDir(),"not-an-image.png"); try(FileOutputStream out=new FileOutputStream(invalid)){out.write(new byte[]{1,2,3});}
        main(()->app.load(Uri.fromFile(invalid),"invalid")); idle(); assertEquals(id,app.session.id); assertTrue(app.status.contains("图片未加载"));
        int previous=app.session.metadata.optInt("run_number");
        main(()->{app.process(); assertTrue(app.busy); app.process();});
        scenario.onActivity(a->{
            if(app.busy) {
                assertFalse(a.findViewById(R.id.mode_mobileie).isEnabled());assertFalse(a.findViewById(R.id.mode_gamma).isEnabled());assertFalse(a.findViewById(R.id.mode_overview).isEnabled());
                assertFalse(a.findViewById(R.id.metrics).isEnabled());
                assertTrue(((android.widget.TextView)a.findViewById(R.id.metrics)).getText().toString().contains("任务进行中"));
            }
        });idle();
        assertEquals(previous+1,app.session.metadata.getInt("run_number")); assertNotNull(app.session.mobileie);
        invalid.delete();
    }

    @Test public void fEmptyCameraOutputIsAnErrorAndKeepsResult() throws Exception {
        processed(1); String id=app.session.id;
        File dir=new File(context.getFilesDir(),"captures"); dir.mkdirs();
        File empty=File.createTempFile("empty-",".jpg",dir);
        scenario.onActivity(a->{
            try { Field f=MainActivity.class.getDeclaredField("pendingCapture"); f.setAccessible(true); f.set(a,empty); }
            catch(Exception e) {throw new AssertionError(e);}
            a.onActivityResult(MainActivity.CAMERA,Activity.RESULT_OK,null);
        });
        assertEquals(id,app.session.id); assertNotNull(app.session.mobileie);
        assertTrue(app.status,app.status.contains("未返回完整照片")); assertFalse(empty.exists());
    }

    @Test public void gModelFailureKeepsOnlyHonestBaseline() throws Exception {
        processed(1);
        Field field=LightLabApplication.class.getDeclaredField("engine");field.setAccessible(true);
        MobileEngine engine=(MobileEngine)field.get(app); engine.close();field.set(app,null);
        LightLabApplication.EngineFactory previous=app.engineFactory;
        try {
            app.engineFactory=c->{throw new IOException("controlled model failure");};
            main(()->app.process()); idle();
            assertNull(app.session.mobileie); assertNotNull(app.session.gamma);
            assertFalse(app.session.metadata.getBoolean("mobileie_complete")); assertFalse(app.session.metadata.has("inference_ms"));
            scenario.onActivity(a->assertFalse(a.findViewById(R.id.mode_mobileie).isEnabled()));
            find("mode_overview").click();
            assertNotNull(device.wait(Until.findObject(By.res(PKG,"mode_overview").selected(true)),3000));
            scenario.onActivity(a->assertTrue(a.findViewById(R.id.comparison).getContentDescription().toString().contains("本次未生成")));
            assertTrue(app.status.contains("Gamma")); shot("06-model-failure-honest-baseline");
        } finally { app.engineFactory=previous; }
    }

    @Test public void hExportFailureRollsBackPartialFiles() throws Exception {
        processed(1);
        LightLabApplication.Session broken=new LightLabApplication.Session(); broken.id="rollback-test-0000"; broken.input=app.session.input;broken.mobileie=app.session.mobileie;broken.gamma=app.session.gamma;broken.metadata=null;
        try { ResultExporter.save(context,broken); fail("Export must fail"); } catch(IOException expected) {}
        try(Cursor c=context.getContentResolver().query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,new String[]{"_id"},"_display_name LIKE ?",new String[]{"MobileIE_rollback_%"},null)) { assertNotNull(c); assertEquals(0,c.getCount()); }
    }

    @Test public void zHealthyResultReadyForProcessRestoreReview() throws Exception {
        processed(1); find("mode_mobileie").click(); shot("07-final-result");
        assertTrue(new File(app.session.directory,"record.json").isFile());
    }
}
