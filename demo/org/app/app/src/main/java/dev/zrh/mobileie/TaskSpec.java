package dev.zrh.mobileie;

import android.content.Context;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Task and display mode are independent concepts. All model inputs are images. */
public final class TaskSpec {
    public final String id,name,modelAsset,modelSha,baselineId,baselineName;
    public final int[] inputShape,outputShape;
    public final String[] samples;
    public final boolean raw;
    private static Map<String,TaskSpec> registry;
    private TaskSpec(JSONObject j) throws JSONException {
        id=j.getString("id");name=j.getString("name");modelAsset=j.getString("model_asset");modelSha=j.getString("model_sha256");
        baselineId=j.getString("baseline_id");baselineName=j.getString("baseline_name");raw=id.equals("isp");
        inputShape=shape(j.getJSONArray("input_shape"));outputShape=shape(j.getJSONArray("output_shape"));
        if(j.getJSONArray("native_control_inputs").length()!=0)throw new JSONException("Unsupported control input");
        JSONArray a=j.getJSONArray("sample_assets");samples=new String[a.length()];for(int i=0;i<a.length();i++)samples[i]=a.getString(i);
    }
    private static int[] shape(JSONArray a) throws JSONException {int[] s=new int[a.length()];for(int i=0;i<s.length;i++)s[i]=a.getInt(i);return s;}
    public static synchronized TaskSpec get(Context context,String id) throws IOException {
        if(registry==null) {
            try(InputStream in=context.getAssets().open("tasks.json");ByteArrayOutputStream out=new ByteArrayOutputStream()) {
                byte[] b=new byte[4096];int n;while((n=in.read(b))!=-1)out.write(b,0,n);
                JSONArray a=new JSONObject(out.toString(StandardCharsets.UTF_8.name())).getJSONArray("tasks");Map<String,TaskSpec> loaded=new LinkedHashMap<>();
                for(int i=0;i<a.length();i++){TaskSpec t=new TaskSpec(a.getJSONObject(i));loaded.put(t.id,t);}
                if(!loaded.keySet().equals(new LinkedHashSet<>(Arrays.asList("lowlight","underwater","isp"))))throw new IOException("任务清单不完整");registry=loaded;
            } catch(JSONException e){throw new IOException("任务清单无效",e);}
        }
        TaskSpec task=registry.get(id);if(task==null)throw new IOException("未知任务");return task;
    }
    public int width(){return outputShape[2];}
    public int height(){return outputShape[1];}
    public String originalLabel(){return raw?"RAW 可视化":"原图";}
}
