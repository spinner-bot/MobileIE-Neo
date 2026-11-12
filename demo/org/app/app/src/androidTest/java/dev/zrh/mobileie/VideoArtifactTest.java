package dev.zrh.mobileie;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.json.JSONObject;
import java.io.File;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import static org.junit.Assert.*;

/** Delivery recording check; no recording/player functionality in the product. */
@RunWith(AndroidJUnit4.class)
public class VideoArtifactTest {
    @Test public void recordingDecodesToRealFrame() throws Exception {
        Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
        File video=new File(c.getFilesDir(),"demo-video.mp4");assertTrue(video.length()>10000);
        MediaMetadataRetriever decoder=new MediaMetadataRetriever();
        try {
            decoder.setDataSource(video.getPath());
            long duration=Long.parseLong(decoder.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION));
            int width=Integer.parseInt(decoder.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH));
            int height=Integer.parseInt(decoder.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT));
            assertTrue(duration>3000);assertEquals(720,width);assertEquals(1560,height);
            Bitmap frame=decoder.getFrameAtTime(duration*500,MediaMetadataRetriever.OPTION_CLOSEST);assertNotNull(frame);
            int[] pixels=new int[frame.getWidth()*frame.getHeight()];frame.getPixels(pixels,0,frame.getWidth(),0,0,frame.getWidth(),frame.getHeight());
            double sum=0,square=0;for(int pixel:pixels){double value=(((pixel>>16)&255)+((pixel>>8)&255)+(pixel&255))/3.0;sum+=value;square+=value*value;}
            double standardDeviation=Math.sqrt(square/pixels.length-Math.pow(sum/pixels.length,2));assertTrue("Recording must not be blank",standardDeviation>5);
            File out=new File(c.getFilesDir(),"video-gate");out.mkdirs();ImagePipeline.savePng(frame,new File(out,"decoded-midpoint.png"));
            JSONObject result=new JSONObject().put("pass",true).put("duration_ms",duration).put("width",width).put("height",height).put("decoded_midpoint",true).put("frame_luminance_std",standardDeviation);
            Files.write(new File(out,"video-decoder.json").toPath(),result.toString(2).getBytes(StandardCharsets.UTF_8));frame.recycle();
        } finally {decoder.release();}
    }
}
