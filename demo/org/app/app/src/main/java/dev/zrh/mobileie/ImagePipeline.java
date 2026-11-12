package dev.zrh.mobileie;

import android.graphics.*;
import androidx.exifinterface.media.ExifInterface;
import java.io.*;

/** One common RGB input for every method and every comparison view. */
public final class ImagePipeline {
    public static final int WIDTH = 600, HEIGHT = 400;
    private static final int[] GAMMA = new int[256];
    static { for (int i = 0; i < 256; i++) GAMMA[i] = (int) (255.0 * Math.pow(i / 255.0, 0.5)); }
    private ImagePipeline() {}

    public static Bitmap prepare(File source) throws IOException {
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(source.getPath(), o);
        if (o.outWidth <= 0 || o.outHeight <= 0 || o.outWidth > 50000 || o.outHeight > 50000)
            throw new IOException("无法读取这张图片，请选择 JPG 或 PNG 照片");
        o.inJustDecodeBounds = false;
        o.inSampleSize = 1;
        while (Math.max(o.outWidth, o.outHeight) / o.inSampleSize > 2048) o.inSampleSize *= 2;
        o.inPreferredConfig = Bitmap.Config.ARGB_8888;
        o.inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB);
        Bitmap decoded = BitmapFactory.decodeFile(source.getPath(), o);
        if (decoded == null) throw new IOException("图片解码失败，请换一张照片");
        int orientation = ExifInterface.ORIENTATION_NORMAL;
        try { orientation = new ExifInterface(source.getPath()).getAttributeInt(ExifInterface.TAG_ORIENTATION, orientation); }
        catch (IOException ignored) { /* An otherwise decoded image can legitimately have no EXIF. */ }
        Bitmap upright = orient(decoded, orientation);
        int w = upright.getWidth(), h = upright.getHeight();
        int cw = w, ch = h;
        if (w * 2L > h * 3L) cw = Math.max(1, h * 3 / 2);
        else ch = Math.max(1, w * 2 / 3);
        Bitmap crop = Bitmap.createBitmap(upright, (w - cw) / 2, (h - ch) / 2, cw, ch);
        Bitmap resized = Bitmap.createScaledBitmap(crop, WIDTH, HEIGHT, true);
        // Composite transparent inputs once, before both methods split.
        Bitmap rgb = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(rgb);
        canvas.drawColor(Color.WHITE);
        canvas.drawBitmap(resized, 0, 0, null);
        if (resized != crop && resized != upright && resized != decoded) resized.recycle();
        if (crop != upright && crop != decoded) crop.recycle();
        if (upright != decoded) upright.recycle();
        decoded.recycle();
        return rgb;
    }

    public static Bitmap orient(Bitmap b, int orientation) {
        Matrix m = new Matrix();
        switch (orientation) {
            case 2: m.setScale(-1, 1); break;
            case 3: m.setRotate(180); break;
            case 4: m.setScale(1, -1); break;
            case 5: m.setRotate(90); m.postScale(-1, 1); break;
            case 6: m.setRotate(90); break;
            case 7: m.setRotate(270); m.postScale(-1, 1); break;
            case 8: m.setRotate(270); break;
            default: return b;
        }
        return Bitmap.createBitmap(b, 0, 0, b.getWidth(), b.getHeight(), m, false);
    }

    public static Bitmap gamma(Bitmap input) {
        int w = input.getWidth(), h = input.getHeight();
        int[] pixels = new int[w * h];
        input.getPixels(pixels, 0, w, 0, 0, w, h);
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            pixels[i] = Color.rgb(GAMMA[(p >>> 16) & 255], GAMMA[(p >>> 8) & 255], GAMMA[p & 255]);
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888);
    }

    public static void savePng(Bitmap bitmap, File destination) throws IOException {
        try (OutputStream out = new FileOutputStream(destination)) {
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) throw new IOException("图片保存失败");
        }
    }
}
