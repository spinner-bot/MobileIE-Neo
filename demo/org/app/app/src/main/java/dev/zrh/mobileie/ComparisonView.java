package dev.zrh.mobileie;

import android.content.Context;
import android.graphics.*;
import android.os.Bundle;
import android.view.*;
import android.view.accessibility.*;

/** Two simultaneous split panes share one position; static overview is separate. */
public final class ComparisonView extends View {
    public static final int DYNAMIC=0,STATIC=1;
    private final Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG|Paint.FILTER_BITMAP_FLAG);
    private final RectF[] cells={new RectF(),new RectF(),new RectF()};
    private final Bitmap[] overview=new Bitmap[3];
    private Bitmap original,mobileie,baseline;
    private TaskSpec task;
    private int mode,activePane=-1;
    private float split=.5f;
    private final float density;
    public ComparisonView(Context context){super(context);density=getResources().getDisplayMetrics().density;setFocusable(true);setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_YES);}
    public void images(Bitmap input,Bitmap model,Bitmap traditional,int selected,TaskSpec selectedTask){
        boolean layout=mode!=selected||original==null||input==null||original.getWidth()!=input.getWidth()||original.getHeight()!=input.getHeight();
        original=input;mobileie=model;baseline=traditional;mode=selected;task=selectedTask;overview[0]=input;overview[1]=model;overview[2]=traditional;
        setContentDescription((task==null?"":task.name+"。")+(mode==STATIC?"静态三图总览：原图、MobileIE、baseline":"动态并列比较：左侧原图与baseline，右侧原图与MobileIE，两条分界线同步滑动")+(model==null?"，MobileIE 未生成":""));
        if(layout)requestLayout();invalidate();
    }
    public float position(){return split;}
    public void position(float value){split=Math.max(0,Math.min(1,value));invalidate();}
    public int displayMode(){return mode;}
    public RectF paneBounds(int index){return new RectF(cells[index]);}
    @Override protected void onMeasure(int widthSpec,int heightSpec){
        int width=MeasureSpec.getSize(widthSpec),n=mode==STATIC?3:2;float ratio=original==null?2f/3:original.getHeight()/(float)original.getWidth();
        float cell=(width-(n-1)*8*density)/n;setMeasuredDimension(width,Math.round(cell*ratio+40*density));
    }
    @Override protected void onDraw(Canvas canvas){
        super.onDraw(canvas);int n=mode==STATIC?3:2;float width=(getWidth()-(n-1)*8*density)/n;
        float ratio=original==null?2f/3:original.getHeight()/(float)original.getWidth();float imageHeight=width*ratio;
        for(int i=0;i<n;i++){
            RectF r=cells[i];float left=i*(width+8*density);r.set(left,32*density,left+width,32*density+imageHeight);
            paint.setColor(0xffeef0e7);canvas.drawRoundRect(left,0,left+width,getHeight(),8*density,8*density,paint);
            String origin=task!=null?task.originalLabel():"原图";String method=i==0?(task!=null?task.baselineName:"baseline"):"MobileIE";
            String title=mode==DYNAMIC?origin+" ↔ "+method:i==0?origin:i==1?"MobileIE":task!=null?task.baselineName:"baseline";
            paint.setColor(0xff243d35);paint.setTypeface(Typeface.DEFAULT_BOLD);paint.setTextAlign(Paint.Align.CENTER);paint.setTextSize(11*density);
            if(paint.measureText(title)>width-8*density)paint.setTextSize(Math.max(8*density,paint.getTextSize()*(width-8*density)/paint.measureText(title)));
            canvas.drawText(title,left+width/2,21*density,paint);paint.setTypeface(Typeface.DEFAULT);paint.setTextAlign(Paint.Align.LEFT);
            if(original==null){placeholder(canvas,r,"待选择输入");continue;}
            if(mode==STATIC){if(overview[i]!=null)canvas.drawBitmap(overview[i],null,r,paint);else placeholder(canvas,r,"未生成");continue;}
            canvas.drawBitmap(original,null,r,paint);Bitmap result=i==0?baseline:mobileie;
            canvas.save();canvas.clipRect(r.left+r.width()*split,r.top,r.right,r.bottom);
            if(result!=null)canvas.drawBitmap(result,null,r,paint);else placeholder(canvas,r,"未生成");canvas.restore();
            float x=r.left+r.width()*split,y=r.centerY();paint.setColor(Color.WHITE);paint.setStrokeWidth(2*density);canvas.drawLine(x,r.top,x,r.bottom,paint);canvas.drawCircle(x,y,10*density,paint);
            paint.setColor(0xff243d35);paint.setStrokeWidth(density);canvas.drawLine(x-3*density,y-4*density,x-3*density,y+4*density,paint);canvas.drawLine(x+3*density,y-4*density,x+3*density,y+4*density,paint);
        }
    }
    private void placeholder(Canvas c,RectF r,String label){paint.setColor(0xffe8ecdf);c.drawRect(r,paint);paint.setColor(0xff68766c);paint.setTextSize(11*density);paint.setTextAlign(Paint.Align.CENTER);c.drawText(label,r.centerX(),r.centerY(),paint);paint.setTextAlign(Paint.Align.LEFT);}
    @Override public void setEnabled(boolean enabled){super.setEnabled(enabled);if(!enabled){activePane=-1;if(getParent()!=null)getParent().requestDisallowInterceptTouchEvent(false);}}
    @Override public boolean onTouchEvent(MotionEvent event){
        if(!isEnabled()||mode!=DYNAMIC||original==null)return super.onTouchEvent(event);
        if(event.getAction()==MotionEvent.ACTION_DOWN){activePane=-1;for(int i=0;i<2;i++)if(cells[i].contains(event.getX(),event.getY()))activePane=i;if(activePane<0)return super.onTouchEvent(event);getParent().requestDisallowInterceptTouchEvent(true);}
        if(activePane<0)return super.onTouchEvent(event);
        RectF r=cells[activePane];position((event.getX()-r.left)/r.width());
        if(event.getAction()==MotionEvent.ACTION_UP||event.getAction()==MotionEvent.ACTION_CANCEL){activePane=-1;getParent().requestDisallowInterceptTouchEvent(false);performClick();}return true;
    }
    @Override public boolean performClick(){super.performClick();return true;}
    @Override public boolean onKeyDown(int key,KeyEvent event){
        if(isEnabled()&&mode==DYNAMIC&&original!=null&&(key==KeyEvent.KEYCODE_DPAD_LEFT||key==KeyEvent.KEYCODE_DPAD_RIGHT)){position(split+(key==KeyEvent.KEYCODE_DPAD_RIGHT?.05f:-.05f));return true;}return super.onKeyDown(key,event);
    }
    @Override public void onInitializeAccessibilityNodeInfo(AccessibilityNodeInfo info){super.onInitializeAccessibilityNodeInfo(info);info.setClassName("android.widget.SeekBar");info.setRangeInfo(AccessibilityNodeInfo.RangeInfo.obtain(AccessibilityNodeInfo.RangeInfo.RANGE_TYPE_FLOAT,0,1,split));if(mode==DYNAMIC){info.addAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_FORWARD);info.addAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_BACKWARD);}}
    @Override public boolean performAccessibilityAction(int action,Bundle args){if(!isEnabled()||mode!=DYNAMIC)return false;if(action==AccessibilityNodeInfo.ACTION_SCROLL_FORWARD||action==AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD){position(split+(action==AccessibilityNodeInfo.ACTION_SCROLL_FORWARD?.1f:-.1f));return true;}return super.performAccessibilityAction(action,args);}
}
