package dev.zrh.mobileie;

import android.app.Activity;
import android.content.Intent;
import android.os.SystemClock;
import android.widget.TextView;
import androidx.lifecycle.Lifecycle;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry;
import androidx.test.runner.lifecycle.Stage;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiDevice;
import androidx.test.uiautomator.Until;
import org.junit.*;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

/** Actual Activity back-stack reproduction of the phone's stale sample screen. */
@RunWith(AndroidJUnit4.class)
public class LifecycleRegressionTest {
    private final LightLabApplication app=(LightLabApplication)InstrumentationRegistry.getInstrumentation().getTargetContext().getApplicationContext();
    private final UiDevice device=UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());
    private ActivityScenario<MainActivity> scenario;
    private MainActivity first;
    private void main(Runnable action) { InstrumentationRegistry.getInstrumentation().runOnMainSync(action); }
    private void idle() {
        long end=SystemClock.uptimeMillis()+30000;
        do { SystemClock.sleep(50); } while(app.busy&&SystemClock.uptimeMillis()<end);
        assertFalse("Worker did not finish: "+app.status,app.busy);device.waitForIdle();
    }
    @Before public void launch() {
        scenario=ActivityScenario.launch(MainActivity.class);idle();
        scenario.onActivity(a->{first=a;app.session=null;app.message("选一张暗光照片，开始本机对比");});
    }
    @After public void close() {
        main(()->{
            for(Stage stage:Stage.values()) if(stage!=Stage.DESTROYED)
                for(Activity activity:new java.util.ArrayList<>(ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(stage)))
                    if(activity instanceof MainActivity)activity.finish();
        });
        if(scenario!=null) scenario.close();
    }
    private MainActivity resumedOtherThan(MainActivity previous) {
        MainActivity[] top={null};
        long end=SystemClock.uptimeMillis()+5000;
        do {
            main(()->{
                for(Activity activity:ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED))
                    if(activity instanceof MainActivity&&activity!=previous)top[0]=(MainActivity)activity;
            });
            if(top[0]!=null)break;
            SystemClock.sleep(50);
        } while(SystemClock.uptimeMillis()<end);
        assertNotNull("A real second MainActivity must be resumed",top[0]);
        return top[0];
    }
    private void openAndCloseSecondActivity() {
        main(()->first.startActivity(new Intent(first,MainActivity.class)));
        MainActivity top=resumedOtherThan(first);
        main(top::finish);
        device.waitForIdle();
        main(()->assertEquals("The original page must really resume",Stage.RESUMED,ActivityLifecycleMonitorRegistry.getInstance().getLifecycleStageOf(first)));
    }
    private void chooseFirstSampleAndAssertVisible() {
        main(()->assertTrue(first.findViewById(R.id.sample).performClick()));
        assertNotNull(device.wait(Until.findObject(By.text("01 · 书柜")),3000));
        device.findObject(By.text("01 · 书柜")).click();idle();
        assertNotNull("Sample task really produced a session",app.session);
        main(()->{
            assertTrue("Sample loaded, but visible Activity did not refresh",((TextView)first.findViewById(R.id.status)).getText().toString().contains("图片已准备"));
            assertTrue("Process button must become usable",first.findViewById(R.id.process).isEnabled());
        });
    }
    @Test public void aReturningFromSecondActivityKeepsSampleScreenLive() {
        openAndCloseSecondActivity();chooseFirstSampleAndAssertVisible();
    }
    @Test public void bCompletionWhileStoppedIsReplayedOnResume() {
        scenario.moveToState(Lifecycle.State.CREATED);
        main(()->app.loadSample(2));idle();String id=app.session.id;
        scenario.moveToState(Lifecycle.State.RESUMED);
        scenario.onActivity(a->{assertEquals(id,app.session.id);assertTrue(a.findViewById(R.id.process).isEnabled());assertTrue(((TextView)a.findViewById(R.id.status)).getText().toString().contains("图片已准备"));});
    }
    @Test public void cRepeatedBackStackAndRecreationKeepActionsLive() {
        for(int cycle=0;cycle<6;cycle++) {
            openAndCloseSecondActivity();chooseFirstSampleAndAssertVisible();
            MainActivity previous=first;main(previous::recreate);first=resumedOtherThan(previous);
        }
        main(()->{
            try { app.session.metadata.put("app_version","0.1.1"); }
            catch(org.json.JSONException e) { throw new AssertionError(e); }
            first.findViewById(R.id.process).performClick();
        });idle();
        assertNotNull(app.session.mobileie);assertNotNull(app.session.gamma);
        assertEquals("Reprocessing an upgraded session must record the running version",BuildConfig.VERSION_NAME,app.session.metadata.optString("app_version"));
        main(()->assertTrue(((TextView)first.findViewById(R.id.status)).getText().toString().contains("处理完成")));
    }
}
