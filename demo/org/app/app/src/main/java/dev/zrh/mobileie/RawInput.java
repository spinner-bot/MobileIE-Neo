package dev.zrh.mobileie;

import android.graphics.*;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import java.util.Arrays;

/** Lossless real Huawei RGGB input; ordinary RGB photographs are never RAW. */
public final class RawInput {
    public static final int SIZE=448, HEADER=28;
    public final int[] mosaic;
    public final float[] tensor;
    public final Bitmap preview;
    private RawInput(int[] raw) {
        mosaic=raw;tensor=new float[SIZE*SIZE];int[] pixels=new int[raw.length];int k=0;
        for(int y=0;y<SIZE/2;y++)for(int x=0;x<SIZE/2;x++)for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)tensor[k++]=raw[(2*y+dy)*SIZE+2*x+dx]/4095f;
        for(int i=0;i<raw.length;i++){int v=(int)(raw[i]/1023.0*255);pixels[i]=Color.rgb(v,v,v);}
        preview=Bitmap.createBitmap(pixels,SIZE,SIZE,Bitmap.Config.ARGB_8888);
    }
    public static RawInput read(File file) throws IOException {
        if(file.length()!=HEADER+SIZE*SIZE*2)throw new IOException("ISP 需要配套 448×448 .rggb RAW 文件，不能使用普通照片");
        byte[] data=Files.readAllBytes(file.toPath());ByteBuffer b=ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);byte[] magic=new byte[8];b.get(magic);
        if(!Arrays.equals(magic,new byte[]{'M','I','E','_','R','A','W','1'})||b.getInt()!=SIZE||b.getInt()!=SIZE||b.getInt()!=4095||b.getInt()!=64||b.getInt()!=1023)throw new IOException("RAW 格式或传感器配置不匹配");
        int[] raw=new int[SIZE*SIZE];for(int i=0;i<raw.length;i++){raw[i]=b.getShort()&65535;if(raw[i]>1023)throw new IOException("RAW 数值超过声明的 10-bit 范围");}
        return new RawInput(raw);
    }
    public Bitmap baseline() {
        double[][] channels=new double[3][mosaic.length];double[] means=new double[3];
        for(int c=0;c<3;c++)for(int y=0;y<SIZE;y++)for(int x=0;x<SIZE;x++){
            double sum=0,weight=0;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                int yy=y+dy,xx=x+dx;if(yy<0||xx<0||yy>=SIZE||xx>=SIZE)continue;
                int known=(yy%2==0?(xx%2==0?0:1):(xx%2==0?1:2));if(known!=c)continue;
                int w=c==1?(dx==0&&dy==0?4:dx==0||dy==0?1:0):(dx==0?2:1)*(dy==0?2:1);
                sum+=mosaic[yy*SIZE+xx]*w;weight+=w;
            }
            double value=sum/Math.max(1,weight)/4095;channels[c][y*SIZE+x]=value;means[c]+=value/mosaic.length;
        }
        double average=(means[0]+means[1]+means[2])/3;int[] pixels=new int[mosaic.length];
        for(int i=0;i<pixels.length;i++){
            int pixel=0xff000000;for(int c=0;c<3;c++){
                double v=Math.max(0,Math.min(1,channels[c][i]*Math.min(4,average/Math.max(means[c],1e-8))));
                double encoded=v<=.0031308?12.92*v:1.055*Math.pow(v,1/2.4)-.055;pixel|=((int)Math.max(0,Math.min(255,encoded*255)))<<(16-c*8);
            }pixels[i]=pixel;
        }
        return Bitmap.createBitmap(pixels,SIZE,SIZE,Bitmap.Config.ARGB_8888);
    }
}
