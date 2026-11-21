package dev.zrh.mobileie;

import android.content.Context;
import android.graphics.*;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.*;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import org.json.*;

@RunWith(AndroidJUnit4.class)
public class CoreInstrumentedTest {
    private final Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();

    @Test public void gammaAllValuesAndChannels() {
        int[] values = new int[256];
        for(int i=0;i<256;i++) values[i]=Color.rgb(i,255-i,(i*31)%256);
        Bitmap source=Bitmap.createBitmap(values,256,1,Bitmap.Config.ARGB_8888);
        Bitmap result=ImagePipeline.gamma(source);
        for(int i=0;i<256;i++) {
            int p=result.getPixel(i,0);
            assertEquals((int)Math.floor(255*Math.pow(i/255.0,.5)),Color.red(p));
            assertEquals((int)Math.floor(255*Math.pow((255-i)/255.0,.5)),Color.green(p));
            assertEquals((int)Math.floor(255*Math.pow(((i*31)%256)/255.0,.5)),Color.blue(p));
        }
        Bitmap extremes=Bitmap.createBitmap(new int[]{Color.BLACK,Color.WHITE},2,1,Bitmap.Config.ARGB_8888);
        Bitmap gammaExtremes=ImagePipeline.gamma(extremes);
        assertEquals(Color.BLACK,gammaExtremes.getPixel(0,0)); assertEquals(Color.WHITE,gammaExtremes.getPixel(1,0));
        extremes.recycle(); gammaExtremes.recycle();
        source.recycle(); result.recycle();
    }

    @Test public void allExifOrientations() {
        int[] values={1,2,3,4,5,6}; for(int i=0;i<6;i++) values[i]|=0xff000000;
        int[][] expected={{1,2,3,4,5,6},{2,1,4,3,6,5},{6,5,4,3,2,1},{5,6,3,4,1,2},
            {1,3,5,2,4,6},{5,3,1,6,4,2},{6,4,2,5,3,1},{2,4,6,1,3,5}};
        Bitmap b=Bitmap.createBitmap(values,2,3,Bitmap.Config.ARGB_8888);
        for(int orientation=1;orientation<=8;orientation++) {
            Bitmap out=ImagePipeline.orient(b,orientation);
            assertEquals(orientation>=5?3:2,out.getWidth()); assertEquals(orientation>=5?2:3,out.getHeight());
            for(int i=0;i<6;i++) assertEquals("orientation "+orientation+" index "+i,expected[orientation-1][i],out.getPixel(i%out.getWidth(),i/out.getWidth())&0xffffff);
            if(out!=b) out.recycle();
        }
        b.recycle();
    }

    @Test public void sharedInputRejectsCorruptionAndKeepsExactSamples() throws Exception {
        File temporary=new File(context.getCacheDir(),"input-contract.png");
        try(InputStream in=context.getAssets().open("samples/01.png")) { Files.copy(in,temporary.toPath(),java.nio.file.StandardCopyOption.REPLACE_EXISTING); }
        Bitmap source=BitmapFactory.decodeFile(temporary.getPath()), prepared=ImagePipeline.prepare(temporary);
        assertArrayEquals(LightLabApplication.bitmapBytes(source),LightLabApplication.bitmapBytes(prepared));
        Bitmap portrait=Bitmap.createBitmap(300,1200,Bitmap.Config.ARGB_8888); portrait.eraseColor(Color.RED);
        ImagePipeline.savePng(portrait,temporary); Bitmap crop=ImagePipeline.prepare(temporary);
        assertEquals(600,crop.getWidth()); assertEquals(400,crop.getHeight()); assertEquals(Color.RED,crop.getPixel(300,200));
        Files.write(temporary.toPath(),new byte[]{1,2,3,4});
        try { ImagePipeline.prepare(temporary); fail("Corrupt input must fail"); } catch(IOException expected) {}
        source.recycle(); prepared.recycle(); portrait.recycle(); crop.recycle(); temporary.delete();
    }

    @Test public void exportConformanceOutputsForIndependentReference() throws Exception {
        File dir=new File(context.getFilesDir(),"gates"); if(!dir.exists()) assertTrue(dir.mkdirs());
        JSONArray evidence=new JSONArray();
        try(MobileEngine engine=new MobileEngine(context)) {
            for(int i=1;i<=6;i++) {
                Bitmap input;
                if(i<=3) {
                    try(InputStream in=context.getAssets().open(String.format(java.util.Locale.ROOT,"samples/%02d.png",i))) { input=BitmapFactory.decodeStream(in); }
                } else {
                    int[] pixels=new int[600*400];
                    for(int y=0;y<400;y++) for(int x=0;x<600;x++) pixels[y*600+x]=i==4?Color.BLACK:i==5?Color.WHITE:Color.rgb(x%256,y%256,(x+y)%256);
                    input=Bitmap.createBitmap(pixels,600,400,Bitmap.Config.ARGB_8888);
                }
                MobileEngine.Output output=engine.run(input);
                assertEquals(600*400*3,output.raw.length);
                for(float value:output.raw) assertTrue(Float.isFinite(value));
                String name="case"+i;
                ImagePipeline.savePng(input,new File(dir,name+"-input.png"));
                ImagePipeline.savePng(output.bitmap,new File(dir,name+"-android.png"));
                ByteBuffer bytes=ByteBuffer.allocate(output.raw.length*4).order(ByteOrder.LITTLE_ENDIAN);
                bytes.asFloatBuffer().put(output.raw);
                Files.write(new File(dir,name+"-android.f32").toPath(),bytes.array());
                evidence.put(new JSONObject().put("case",name).put("inference_ms",output.inferenceMs).put("input_rgb_sha256",MobileEngine.sha256(LightLabApplication.bitmapBytes(input))));
                input.recycle(); output.bitmap.recycle();
            }
        }
        Files.write(new File(dir,"android-cases.json").toPath(),evidence.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
        assertEquals(6,evidence.length());
    }
}
