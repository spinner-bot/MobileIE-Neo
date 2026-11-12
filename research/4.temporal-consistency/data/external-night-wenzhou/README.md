# 作者部分Night Wenzhou备份的来源与用途

记录：2026-11-12（+08:00）；作者：Codex  (sqs_A100)。公开来源为[作者issue6回复](https://github.com/ShenZheng2000/LLIE_Survey/issues/6#issuecomment-2597100240)给出的部分帧/视频备份，稳定file ID 1bIaPaFeOWDwuhWgLXraY1FrSZBAqJF2T。原完整约26GB链接已失效，这份1170848021字节RAR不能当完整标准集。

完整下载见process/external-night-wenzhou/acquisition003/manifest.json；原001提前EOF与本地002技术失败均完整保留，003仅单次公开Range/1MiB重叠核验完成。完整归档与MP4原件都只在zrh自有data，Git只存来源、清单、计划、少量说明文本及过程资料，不重新分发大视频。

author-readmes/两个文件来自归档Frame/Video下原作者文本；Git中的阅读副本规范化换行和行尾空格，原字节/大小/摘要以自有cache及readme001/manifest.json为准。说明描述WKU校园和夜间骑行，没有算法增强视频列表。目录与文本共同支持将其作为作者提供的无配对夜景开发输入，不能保证未知ISP无处理或与论文全数据逐像素相同。Video说明的DJI_0020与实际DJI_0021不一致，保留原件名字；不伪造修订。

六文件：DJI_0019/0021/0023.MP4、Gopro1/2/3.mp4。容器实际DJI29.97003FPS、2720×1530；GoPro30FPS、1920×1080，不照抄论文概述的GoPro1440。三航拍文件有相似建筑，三骑行文件可能同一设备/区域，不称六独立地点或设备，也不以文件名证明具体相机型号。

R60固定全部六文件各中部128连续RGB，原亮度、全FOV长边640，主目标local4..123，共768观察/720主目标。六源位置、时间、形状、SHA见results/exploration-r60-wenzhou-data-attempt001/index.json。只为开发阶段无GT检查，不是完整视频、独立最终、配对PSNR/SSIM或手机速度评价。后续任何训练/输入改变须另协议，不能依赖本批结果挑seed、checkpoint或漂亮片段。

详见[输入身份协议](../../docs/exploration-r60-wenzhou-input-identity-protocol.md)及[冻结参考协议](../../docs/exploration-r60-wenzhou-frozen-reference-protocol.md)。原权重、训练历史/上下文、未来输入、后处理成本和代理局限均按协议披露；移动演示支线不使用或修改这些数据。
