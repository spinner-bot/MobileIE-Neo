package dev.zrh.mobileie;

import android.content.*;
import android.graphics.*;
import android.net.Uri;
import android.provider.MediaStore;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Export only newly generated results, with rollback on a partial write. */
public final class ResultExporter {
    private ResultExporter() {}
    public static void save(Context context, LightLabApplication.Session s) throws IOException {
        ContentResolver resolver = context.getContentResolver();
        List<Uri> inserted = new ArrayList<>();
        String prefix = "MobileIE_" + s.id.substring(0, 8) + "_" + System.currentTimeMillis();
        try {
            put(resolver, inserted, prefix+"_input.png", "image/png", "Pictures/MobileIE", s.input, null);
            if (s.mobileie != null) put(resolver, inserted, prefix+"_mobileie.png", "image/png", "Pictures/MobileIE", s.mobileie, null);
            if (s.baseline != null) put(resolver, inserted, prefix+"_baseline.png", "image/png", "Pictures/MobileIE", s.baseline, null);
            Bitmap sheet = comparison(s);
            put(resolver, inserted, prefix+"_comparison.png", "image/png", "Pictures/MobileIE", sheet, null);
            sheet.recycle();
            put(resolver, inserted, prefix+"_record.json", "application/json", "Download/MobileIE", null, s.metadata.toString(2).getBytes(StandardCharsets.UTF_8));
            if(s.task.raw)put(resolver,inserted,prefix+"_input.rggb","application/octet-stream","Download/MobileIE",null,java.nio.file.Files.readAllBytes(new File(s.directory,"source.image").toPath()));
            ContentValues published = new ContentValues(); published.put(MediaStore.MediaColumns.IS_PENDING, 0);
            for (Uri uri : inserted) resolver.update(uri, published, null, null);
        } catch (Exception e) {
            for (Uri uri : inserted) { try { resolver.delete(uri, null, null); } catch (Exception ignored) {} }
            throw new IOException("未能完整写入相册，请检查可用空间后重试", e);
        }
    }

    private static void put(ContentResolver resolver, List<Uri> inserted, String name, String mime, String directory, Bitmap image, byte[] bytes) throws IOException {
        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, name); values.put(MediaStore.MediaColumns.MIME_TYPE, mime);
        values.put(MediaStore.MediaColumns.RELATIVE_PATH, directory); values.put(MediaStore.MediaColumns.IS_PENDING, 1);
        Uri collection = image == null ? MediaStore.Downloads.EXTERNAL_CONTENT_URI : MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
        Uri uri = resolver.insert(collection, values);
        if (uri == null) throw new IOException("无法创建文件");
        inserted.add(uri);
        try (OutputStream out = resolver.openOutputStream(uri)) {
            if (out == null) throw new IOException("无法写入文件");
            if (image != null) { if (!image.compress(Bitmap.CompressFormat.PNG, 100, out)) throw new IOException("图片写入失败"); }
            else out.write(bytes);
        }
    }

    public static Bitmap comparison(LightLabApplication.Session s) {
        int width=s.task.width(),height=s.task.height();
        Bitmap b = Bitmap.createBitmap(width*3, height+110, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(b); c.drawColor(Color.rgb(246,245,239));
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG); p.setColor(Color.rgb(28,48,40)); p.setTextSize(24);
        c.drawText("本次任务："+s.task.name,16,29,p);p.setTextSize(20);
        String[] labels = {s.task.originalLabel(), "MobileIE · CPU", s.task.baselineName};
        Bitmap[] images = {s.input, s.mobileie, s.baseline};
        for (int i = 0; i < 3; i++) {
            c.drawText(labels[i], 16 + width*i, 63, p);
            if (images[i] != null) c.drawBitmap(images[i], width*i, 76, null);
            else c.drawText("本次未生成", width*i+width/3f, 76+height/2f, p);
        }
        p.setTextSize(18); c.drawText("同一任务输入与坐标 · MobileIE / 固定传统 baseline · 本机计算", 16, height+100, p);
        return b;
    }
}
