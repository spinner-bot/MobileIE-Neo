package dev.zrh.mobileie;

import android.content.Context;
import android.graphics.*;
import android.os.SystemClock;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;
import java.io.*;
import java.nio.*;
import java.security.*;
import java.util.*;

/** Official FP32 model, CPU built-in kernels; all calls belong to one worker. */
public final class MobileEngine implements AutoCloseable {
    public static final String MODEL_SHA = "2ea09b840634469eceab5263bf7b549c33fcd5df2a25701ce5fefc585402ad44";
    private final Interpreter interpreter;
    public final TaskSpec task;
    public final double initializationMs;

    public MobileEngine(Context context) throws IOException {this(context,TaskSpec.get(context,"lowlight"));}
    public MobileEngine(Context context,TaskSpec selected) throws IOException {
        task=selected;
        long start = SystemClock.elapsedRealtimeNanos();
        byte[] model;
        try (InputStream in = context.getAssets().open(task.modelAsset); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] block = new byte[8192]; int n;
            while ((n = in.read(block)) != -1) out.write(block, 0, n);
            model = out.toByteArray();
        }
        if (!task.modelSha.equals(sha256(model))) throw new IOException("模型校验失败，请重新安装应用");
        ByteBuffer buffer = ByteBuffer.allocateDirect(model.length).order(ByteOrder.nativeOrder());
        buffer.put(model).rewind();
        Interpreter.Options options = new Interpreter.Options().setNumThreads(4).setUseNNAPI(false).setUseXNNPACK(false);
        interpreter = new Interpreter(buffer, options);
        interpreter.allocateTensors();
        if (interpreter.getInputTensorCount()!=1 || interpreter.getOutputTensorCount()!=1 ||
            !Arrays.equals(task.inputShape, interpreter.getInputTensor(0).shape()) ||
            !Arrays.equals(task.outputShape, interpreter.getOutputTensor(0).shape()) ||
            interpreter.getInputTensor(0).dataType() != DataType.FLOAT32 ||
            interpreter.getOutputTensor(0).dataType() != DataType.FLOAT32) {
            interpreter.close();
            throw new IOException("模型输入输出不匹配");
        }
        initializationMs = elapsed(start);
    }

    public static final class Output {
        public final Bitmap bitmap;
        public final float[] raw;
        public final double packingMs, inferenceMs, renderingMs;
        Output(Bitmap bitmap, float[] raw, double packing, double inference, double rendering) {
            this.bitmap = bitmap; this.raw = raw;
            packingMs = packing; inferenceMs = inference; renderingMs = rendering;
        }
    }

    public Output run(Bitmap input) throws IOException {
        if(task.raw)throw new IOException("ISP 需要 RAW 输入");return run(input,null);
    }
    public Output run(Bitmap input,float[] rawInput) throws IOException {
        int width=task.width(),height=task.height();
        if (input.getWidth() != width || input.getHeight() != height) throw new IOException("输入尺寸与任务不匹配");
        long t = SystemClock.elapsedRealtimeNanos();
        int[] pixels = new int[width*height];
        input.getPixels(pixels, 0, width, 0, 0, width, height);
        int count=1;for(int n:task.inputShape)count*=n;
        ByteBuffer in = ByteBuffer.allocateDirect(count*4).order(ByteOrder.nativeOrder());
        ByteBuffer out = ByteBuffer.allocateDirect(pixels.length * 12).order(ByteOrder.nativeOrder());
        if(task.raw){if(rawInput==null||rawInput.length!=count)throw new IOException("RAW 张量不匹配");for(float f:rawInput){if(!Float.isFinite(f)||f<0||f>1)throw new IOException("RAW 张量无效");in.putFloat(f);}}
        else for (int p : pixels) { in.putFloat(((p >>> 16) & 255) / 255f); in.putFloat(((p >>> 8) & 255) / 255f); in.putFloat((p & 255) / 255f); }
        in.rewind();
        double packing = elapsed(t);
        t = SystemClock.elapsedRealtimeNanos();
        interpreter.run(in, out);
        double inference = elapsed(t);
        t = SystemClock.elapsedRealtimeNanos();
        out.rewind();
        float[] raw = new float[pixels.length * 3];
        out.asFloatBuffer().get(raw);
        for (int i = 0; i < pixels.length; i++) {
            float r = raw[3 * i], g = raw[3 * i + 1], b = raw[3 * i + 2];
            if (!Float.isFinite(r) || !Float.isFinite(g) || !Float.isFinite(b)) throw new IOException("模型输出异常，请重新处理");
            pixels[i] = Color.rgb(toByte(r), toByte(g), toByte(b));
        }
        Bitmap result = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
        return new Output(result, raw, packing, inference, elapsed(t));
    }

    private static int toByte(float x) { return (int) (Math.max(0f, Math.min(1f, x)) * 255f); }
    public static double elapsed(long start) { return (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0; }
    public static String sha256(byte[] value) throws IOException {
        try {
            StringBuilder s = new StringBuilder();
            for (byte b : MessageDigest.getInstance("SHA-256").digest(value)) s.append(String.format(Locale.ROOT, "%02x", b & 255));
            return s.toString();
        } catch (NoSuchAlgorithmException e) { throw new IOException(e); }
    }
    @Override public void close() { interpreter.close(); }
}
