package dev.zrh.mobileie;

import android.app.Application;
import android.graphics.*;
import android.net.Uri;
import android.os.*;
import android.util.AtomicFile;
import org.json.*;
import java.io.*;
import java.nio.file.Files;
import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.*;

/** Process-level coordinator: survive Activity recreation without duplicate jobs. */
public final class LightLabApplication extends Application {
    public static final class Session {
        public String id, label;
        public File directory;
        public Bitmap input, mobileie, gamma;
        public JSONObject metadata = new JSONObject();
    }
    public volatile Session session;
    public volatile boolean busy;
    public String status = "选一张暗光照片，开始本机对比";
    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private Runnable listener;
    private boolean restoring = true;
    private Runnable queuedAcquisition;
    private MobileEngine engine;
    interface EngineFactory { MobileEngine create(android.content.Context context) throws IOException; }
    volatile EngineFactory engineFactory = MobileEngine::new;
    private File sessions;

    @Override public void onCreate() {
        super.onCreate();
        sessions = new File(getFilesDir(), "sessions");
        if (!sessions.exists()) sessions.mkdirs();
        busy = true;
        worker.execute(() -> {
            Session restored = null;
            try {
                File active = new File(getFilesDir(), "current.json");
                if (active.isFile()) {
                    String id = readJson(active).getString("id");
                    if (!id.matches("[0-9a-f-]{36}")) throw new IOException("Invalid session");
                    File dir = new File(sessions, id);
                    JSONObject m = readJson(new File(dir, "record.json"));
                    restored = new Session(); restored.id = id; restored.directory = dir; restored.label = m.optString("source_label", "上次照片");
                    restored.metadata = m;
                    restored.input = BitmapFactory.decodeFile(new File(dir, "input.png").getPath());
                    if (restored.input == null) throw new IOException("Missing input");
                    if (m.optBoolean("gamma_complete")) restored.gamma = BitmapFactory.decodeFile(new File(dir, "gamma.png").getPath());
                    if (m.optBoolean("mobileie_complete")) restored.mobileie = BitmapFactory.decodeFile(new File(dir, "mobileie.png").getPath());
                }
            } catch (Exception e) { restored = null; }
            Session result = restored;
            main.post(() -> {
                session = result; busy = false; restoring = false;
                if (queuedAcquisition != null) { Runnable next = queuedAcquisition; queuedAcquisition = null; next.run(); }
                else { status = result == null ? "选一张暗光照片，开始本机对比" : "已恢复上次照片，可继续查看或重新处理"; emit(); }
            });
        });
    }

    public void attach(Runnable observer) { listener = observer; emit(); }
    public void detach(Runnable observer) { if (listener == observer) listener = null; }
    private void emit() { if (listener != null) listener.run(); }
    public void message(String message) { status = message; emit(); }
    private void finish(String message) { busy = false; status = message; emit(); }

    public void load(Uri source, String label) { beginLoad(() -> getContentResolver().openInputStream(source), label); }
    public void loadSample(int index) { beginLoad(() -> getAssets().open(String.format(java.util.Locale.ROOT, "samples/%02d.png", index)), "内置暗光样例 " + index); }
    interface StreamSource { InputStream open() throws IOException; }
    private void beginLoad(StreamSource source, String label) {
        if (busy) {
            if (restoring) queuedAcquisition = () -> beginLoad(source, label);
            return;
        }
        busy = true; status = "正在准备图片…"; emit();
        worker.execute(() -> {
            File dir = new File(sessions, UUID.randomUUID().toString());
            try {
                if (!dir.mkdirs()) throw new IOException("无法创建本地图片记录");
                File file = new File(dir, "source.image");
                try (InputStream in = source.open(); OutputStream out = new FileOutputStream(file)) {
                    if (in == null) throw new IOException("无法读取这张图片");
                    byte[] buffer = new byte[8192]; int n; long total = 0;
                    while ((n = in.read(buffer)) != -1) {
                        total += n; if (total > 32L * 1024 * 1024) throw new IOException("图片超过 32 MB，请选择较小的照片");
                        out.write(buffer, 0, n);
                    }
                }
                long start = SystemClock.elapsedRealtimeNanos();
                Bitmap input = ImagePipeline.prepare(file);
                double prep = MobileEngine.elapsed(start);
                Session s = new Session(); s.id = dir.getName(); s.directory = dir; s.label = label; s.input = input;
                s.metadata.put("source_label", label).put("created_at_utc", Instant.now().toString()).put("width", 600).put("height", 400)
                    .put("crop", "EXIF upright, centered 3:2 crop, RGB/sRGB").put("gamma_exponent", 0.5).put("preparation_ms", prep)
                    .put("input_sha256", MobileEngine.sha256(bitmapBytes(input))).put("model_sha256", MobileEngine.MODEL_SHA)
                    .put("app_version", BuildConfig.VERSION_NAME).put("runtime", "LiteRT 1.4.2 CPU built-in, 4 threads")
                    .put("gamma_complete", false).put("mobileie_complete", false);
                ImagePipeline.savePng(input, new File(dir, "input.png"));
                write(s);
                atomicJson(new File(getFilesDir(), "current.json"), new JSONObject().put("id", s.id));
                main.post(() -> { session = s; finish("图片已准备 · 请确认居中裁剪范围后开始对比"); });
            } catch (Exception e) {
                deleteTree(dir);
                main.post(() -> finish("图片未加载：" + readable(e)));
            }
        });
    }

    public void process() {
        Session s = session;
        if (busy || s == null) return;
        busy = true; status = "正在本机处理，请稍候…"; emit();
        worker.execute(() -> {
            try {
                s.metadata.put("run_number", s.metadata.optInt("run_number", 0) + 1).put("gamma_complete", false).put("mobileie_complete", false);
                for (String field : new String[]{"initialization_ms", "initialization_in_this_run", "packing_ms", "inference_ms", "rendering_ms"}) s.metadata.remove(field);
                write(s);
                long total = SystemClock.elapsedRealtimeNanos();
                long t = SystemClock.elapsedRealtimeNanos();
                Bitmap gamma = ImagePipeline.gamma(s.input);
                double gammaMs = MobileEngine.elapsed(t);
                s.metadata.put("gamma_ms", gammaMs);
                MobileEngine.Output output = null;
                String failure = null;
                boolean first = engine == null;
                try {
                    if (engine == null) engine = engineFactory.create(this);
                    output = engine.run(s.input);
                    s.metadata.put("initialization_ms", engine.initializationMs).put("initialization_in_this_run", first)
                        .put("packing_ms", output.packingMs).put("inference_ms", output.inferenceMs).put("rendering_ms", output.renderingMs);
                    s.metadata.remove("mobileie_error");
                } catch (Exception | LinkageError error) {
                    failure = "MobileIE 暂不可用，请重新处理或重新安装应用";
                    s.metadata.put("mobileie_error", error.getClass().getSimpleName());
                    if (engine != null) { engine.close(); engine = null; }
                }
                s.metadata.put("processing_total_ms", MobileEngine.elapsed(total)).put("processed_at_utc", Instant.now().toString());
                ImagePipeline.savePng(gamma, new File(s.directory, "gamma.png"));
                if (output != null) ImagePipeline.savePng(output.bitmap, new File(s.directory, "mobileie.png"));
                s.metadata.put("gamma_complete", true).put("mobileie_complete", output != null);
                write(s);
                MobileEngine.Output done = output; String message = failure;
                main.post(() -> {
                    s.gamma = gamma; s.mobileie = done == null ? null : done.bitmap;
                    finish(message == null ? "处理完成 · 两种方法均在本机计算" : message + "；Gamma 结果可查看");
                });
            } catch (Exception e) { main.post(() -> { s.mobileie = null; s.gamma = null; finish("处理未完成：" + readable(e)); }); }
        });
    }

    public void save() {
        Session s = session;
        if (busy || s == null || s.gamma == null) return;
        busy = true; status = "正在保存对比结果…"; emit();
        worker.execute(() -> {
            try { ResultExporter.save(this, s); main.post(() -> finish("已保存至相册 Pictures/MobileIE；处理记录在 Download/MobileIE")); }
            catch (Exception e) { main.post(() -> finish("保存失败：" + readable(e))); }
        });
    }

    static byte[] bitmapBytes(Bitmap bitmap) {
        int[] pixels = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        byte[] bytes = new byte[pixels.length * 3];
        for (int i = 0; i < pixels.length; i++) { bytes[3*i]=(byte)(pixels[i]>>16); bytes[3*i+1]=(byte)(pixels[i]>>8); bytes[3*i+2]=(byte)pixels[i]; }
        return bytes;
    }
    static void write(Session s) throws IOException { atomicJson(new File(s.directory, "record.json"), s.metadata); }
    static JSONObject readJson(File file) throws IOException, JSONException { return new JSONObject(new String(Files.readAllBytes(file.toPath()), java.nio.charset.StandardCharsets.UTF_8)); }
    static void atomicJson(File file, JSONObject value) throws IOException {
        AtomicFile atomic = new AtomicFile(file);
        FileOutputStream out = null;
        try { out = atomic.startWrite(); out.write(value.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)); atomic.finishWrite(out); }
        catch (IOException e) { if (out != null) atomic.failWrite(out); throw e; }
    }
    static String readable(Throwable e) {
        if (e instanceof SecurityException) return "图片授权已失效，请重新选择";
        String message = e.getMessage();
        return message != null && message.matches("(?s).*[\\u4e00-\\u9fff].*") ? message : "请重试或更换图片";
    }
    static void deleteTree(File dir) { File[] children = dir.listFiles(); if (children != null) for (File f : children) deleteTree(f); dir.delete(); }
}
