package dev.zrh.mobileie;

import android.content.Context;
import android.graphics.*;
import android.os.Bundle;
import android.view.*;
import android.view.accessibility.*;

/** Aspect-correct comparison, drawn in a shared coordinate system. */
public final class ComparisonView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final RectF bounds = new RectF(), cellBounds = new RectF();
    private final Path clip = new Path();
    private final Bitmap[] overviewImages = new Bitmap[3];
    private final String[] overviewLabels = {"原图","MobileIE","Gamma 0.5"};
    private Shader placeholder;
    private Bitmap original, mobileie, gamma;
    private int mode;
    private float split = 0.5f;
    private final float density;
    public ComparisonView(Context context) {
        super(context); density = getResources().getDisplayMetrics().density;
        setFocusable(true); setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_YES);
    }
    public void images(Bitmap input, Bitmap mie, Bitmap traditional, int selected) {
        boolean relayout = (mode == 2) != (selected == 2);
        original = input; mobileie = mie; gamma = traditional; mode = selected;
        overviewImages[0]=input; overviewImages[1]=mie; overviewImages[2]=traditional;
        setContentDescription(selected == 2 ? "原图、MobileIE 与 Gamma 三图总览" : "原图与" + (selected == 0 ? "MobileIE" : "Gamma") + "滑动对比，可左右调整分界线");
        if (relayout) requestLayout(); invalidate();
    }
    public float position() { return split; }
    public void position(float value) { split = Math.max(0.03f, Math.min(0.97f, value)); invalidate(); }
    @Override protected void onMeasure(int w, int h) {
        int width = MeasureSpec.getSize(w);
        setMeasuredDimension(width, Math.round(mode == 2 ? width * 2f / 9f + 42*density : width * 2f / 3f));
    }
    @Override protected void onDraw(Canvas c) {
        super.onDraw(c);
        float w = getWidth(), h = getHeight();
        c.save(); c.clipPath(clip);
        c.drawColor(Color.rgb(229,234,224));
        if (original == null) {
            paint.setShader(placeholder);
            c.drawRect(0,0,w,h,paint); paint.setShader(null);
            paint.setColor(0xffe9c28e); c.drawCircle(w*.78f,h*.34f,w*.15f,paint);
            paint.setColor(0xff355747); c.drawCircle(w*.84f,h*.25f,w*.15f,paint);
            paint.setColor(0xfff7f5e9); paint.setTextSize(22*density); paint.setTypeface(Typeface.DEFAULT_BOLD);
            c.drawText("让暗处的细节",24*density,h*.70f,paint);
            c.drawText("重新被看见",24*density,h*.70f+31*density,paint);
            paint.setTypeface(Typeface.DEFAULT);
        } else if (mode == 2) {
            float cell = w/3f;
            for(int i=0;i<3;i++) {
                paint.setColor(0xfff0f0e8); c.drawRect(i*cell,0,(i+1)*cell,h,paint);
                cellBounds.set(i*cell,0,(i+1)*cell,cell*2/3f);
                if(overviewImages[i]!=null) c.drawBitmap(overviewImages[i],null,cellBounds,paint);
                paint.setColor(0xff243d35); paint.setTextSize(12*density); paint.setTextAlign(Paint.Align.CENTER);
                c.drawText(overviewLabels[i],cell*(i+.5f),h-15*density,paint); paint.setTextAlign(Paint.Align.LEFT);
            }
        } else {
            c.drawBitmap(original,null,bounds,paint);
            Bitmap processed = mode == 0 ? mobileie : gamma;
            if(processed != null) {
                c.save(); c.clipRect(w*split,0,w,h); c.drawBitmap(processed,null,bounds,paint); c.restore();
                paint.setColor(Color.WHITE); paint.setStrokeWidth(2*density); c.drawLine(w*split,0,w*split,h,paint);
                c.drawCircle(w*split,h/2,18*density,paint);
                paint.setColor(0xff243d35); paint.setStrokeWidth(2*density);
                c.drawLine(w*split-4*density,h/2-6*density,w*split-4*density,h/2+6*density,paint);
                c.drawLine(w*split+4*density,h/2-6*density,w*split+4*density,h/2+6*density,paint);
            }
            label(c,"原图",12*density,12*density);
            if(processed != null) label(c,mode==0?"MobileIE":"Gamma 0.5",w-105*density,12*density);
        }
        c.restore();
    }
    @Override protected void onSizeChanged(int w,int h,int oldw,int oldh) {
        bounds.set(0,0,w,h); clip.reset(); clip.addRoundRect(bounds,16*density,16*density,Path.Direction.CW);
        placeholder=new LinearGradient(0,0,w,h,new int[]{0xff1e3b32,0xff53715e},null,Shader.TileMode.CLAMP);
    }
    private void label(Canvas c,String s,float x,float y) {
        paint.setColor(0xb322352e); c.drawRoundRect(x,y,x+92*density,y+28*density,8*density,8*density,paint);
        paint.setColor(Color.WHITE); paint.setTextSize(11*density); c.drawText(s,x+10*density,y+19*density,paint);
    }
    @Override public boolean onTouchEvent(android.view.MotionEvent event) {
        if(mode==2 || (mode==0?mobileie:gamma)==null) return super.onTouchEvent(event);
        if(event.getAction()==MotionEvent.ACTION_DOWN) getParent().requestDisallowInterceptTouchEvent(true);
        position(event.getX()/getWidth());
        if(event.getAction()==MotionEvent.ACTION_UP || event.getAction()==MotionEvent.ACTION_CANCEL) {
            getParent().requestDisallowInterceptTouchEvent(false); performClick();
        }
        return true;
    }
    @Override public boolean performClick() { super.performClick(); return true; }
    @Override public boolean onKeyDown(int key,KeyEvent event) {
        if(mode!=2 && (mode==0?mobileie:gamma)!=null && (key==KeyEvent.KEYCODE_DPAD_LEFT||key==KeyEvent.KEYCODE_DPAD_RIGHT)) {
            position(split+(key==KeyEvent.KEYCODE_DPAD_RIGHT?.05f:-.05f)); return true;
        }
        return super.onKeyDown(key,event);
    }
    @Override public void onInitializeAccessibilityNodeInfo(AccessibilityNodeInfo info) {
        super.onInitializeAccessibilityNodeInfo(info); info.setClassName("android.widget.SeekBar");
        info.setRangeInfo(AccessibilityNodeInfo.RangeInfo.obtain(AccessibilityNodeInfo.RangeInfo.RANGE_TYPE_FLOAT,0,1,split));
        info.addAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_FORWARD);
        info.addAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_BACKWARD);
    }
    @Override public boolean performAccessibilityAction(int action, Bundle args) {
        if(action==AccessibilityNodeInfo.ACTION_SCROLL_FORWARD || action==AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD) {
            position(split+(action==AccessibilityNodeInfo.ACTION_SCROLL_FORWARD?.1f:-.1f)); return true;
        }
        return super.performAccessibilityAction(action,args);
    }
}
