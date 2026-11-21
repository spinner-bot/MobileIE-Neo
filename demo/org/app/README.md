# 微光实验室 · Android 0.2.0

原生 Java 离线App，三个任务使用三套对应官方模型：低光增强、水下增强、RAW→RGB ISP。每个任务有三个匹配样例和一种固定传统baseline。所有模型都是单图像输入，没有原生可调参数，因此没有参数面板。

动态对比把“原图↔baseline”和“原图↔MobileIE”两个窗口并排显示，分界线同步；静态总览独立保留三图。处理后，展示上方、导出组合图和记录均标注本次任务。切换任务清除旧结果并准备新样例。

低光/水下支持系统拍照与选图，以600×400共同取景输入处理。ISP只接收配套448×448 Huawei P20 RGGB `.rggb`，输入打包224×224×4，输出448×448 RGB；普通照片及任意DNG不支持。RAW面板明确标记强度可视化；不会用RGB图片伪造传感器输入。

从仓库根目录构建与安装：

```bash
source /root/android-dev/activate.sh
bash demo/org/app/build.sh assembleDebug
adb -s emulator-5580 install -r demo/org/app/app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5580 shell am start -n dev.zrh.mobileie/.MainActivity
```

配置见[个人安卓环境报告](../docs/06-android-environment.md)。运行采用LiteRT内置CPU4线程，关闭默认delegate；没有联网权限。最低声明Android10，本版验收平台为Android15 x86_64模拟器。包沿用开发调试签名，可覆盖升级旧版。

操作、输入限制、保存位置和现场彩排见[手机操作说明](../demo-package/微光实验室操作说明.txt)。保存到Pictures/MobileIE的PNG和Download/MobileIE的JSON均记录任务；ISP额外保留源RAW。取消选图保留旧结果，模型失败明确提示，baseline仍可独立查看。

AI Benchmark按钮仅打开外部App，模型选择和测速需手动完成，不自动读取跑分。旧低光模型字节未变；新增水下/ISP模型由官方冻结权重转换，三套模型与出处见[任务注册表](app/src/main/assets/tasks.json)、[来源记录](task-asset-sources.json)和[许可说明](app/src/main/assets/credits.txt)。新任务尚无真机AI Benchmark时延结论。

模拟验收入口（仅个人测试AVD）：

```bash
bash demo/org/scripts/start-android-emulator.sh
bash demo/org/scripts/run-android-app-gates.sh
source /root/android-dev/activate.sh
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I demo/org/scripts/verify-android-lifecycle.py --task lowlight
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I demo/org/scripts/verify-android-lifecycle.py --task underwater
```

主门控含26项仪器测试、原低光6组与三任务9组独立数值核对、Lint、签名和16KB对齐检查。另验相机前台进程死亡、RAW冷恢复、导出和界面。结果见[0.2.0验收报告](../docs/14-three-task-0.2.0-delivery.md)。0.1.2一加9R结果是历史记录；本版0.2.0真机彩排仍待执行。
