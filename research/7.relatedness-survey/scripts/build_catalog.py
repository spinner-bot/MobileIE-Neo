"""One-time post-Gate1 preliminary grading and permanent ID allocation.
Grades are explicit metadata-based priorities, not Gate2 conclusions.
"""
import collections,datetime,hashlib,json
from pathlib import Path
from zoneinfo import ZoneInfo
B=Path(__file__).resolve().parents[1];C=B/'collection';D=B/'catalog'
# These positions refer to the frozen Gate1 candidate array, NOT permanent IDs.
P={
0:((3,3,3),'lightweight-image','024-liteie','同作者极轻量低光增强架构。优先核查其效率和训练设定，作为移动端预算与差异化的直接约束。'),
1:((2,1,2),'underwater-transfer','040-stereo','水下场景立体深度适配。与增强任务不同，可用于核查增强作为前处理的下游联系及信息条件。'),
2:((3,2,2),'same-author-image','025-igdnet','同作者零样本欠曝增强与去噪。可核查噪声处理、监督方式和图像到视频适配边界。'),
3:((2,2,2),'exposure-control','041-exposure','两阶段可控曝光图像增强。曝光控制与帧间亮度稳定存在潜在联系，但时域机制尚未核实。'),
4:((2,2,2),'imaging-restoration','042-minimalist','空间像差均匀化与重参数化复原。可核查训练/部署转换与成像任务差异。'),
5:((3,3,3),'same-group-reparameterization','022-drnet','同组先验引导动态重参数化复原。与轻量部署及结构贡献边界直接相关。'),
6:((1,0,1),'detection-transfer','080-detection','轻量检测网络中的多尺度融合/注意力。先核实MobileIE引用语境，不预设能迁移为视频增强机制。'),
7:((1,0,1),'detection-transfer','080-detection','海洋场景多类型舰船检测。主要价值在MobileIE组件或基线的实际引用边界。'),
8:((1,0,0),'medical-classification','090-medical','多源医学图像分类框架。任务远离视频增强，仍逐项核查引用和模型规模比较是否准确。'),
9:((2,1,1),'infrared-fusion','043-fusion','无监督低光红外/可见光融合。多传感器信息条件与普通RGB因果视频不同，需明确可比边界。'),
10:((3,3,3),'industry-lightweight-image','020-teleai','TeleAI署名的实时水下增强。已有明确MobileIE关联，是工业接续、轻量结构和重复风险的重要核查项。'),
11:((3,3,3),'same-group-reparameterization','021-fgdpa','同组频率引导双路径注意力水下增强。重点核查原MobileIE组件如何接续以及部署成本口径。'),
12:((1,0,1),'detection-transfer','080-detection','轻量PCB缺陷检测。属于异任务引用，分析其组件复用而不将其计作视频增强先例。'),
13:((3,2,3),'perceptual-training','031-jnd','JND引导的轻量感知编码预滤波。可为可见闪烁/感知目标提供线索，但图像编码与时序评价不能混同。'),
14:((1,0,1),'detection-transfer','080-detection','扶梯跌倒行为检测。视频应用标题不自动代表增强或闪烁处理，重点核实引用的实际用途。'),
15:((3,2,3),'efficient-denoising','030-fastshade','移动端高效图像去噪。噪声和完整手机成本是照片到视频扩展的关键约束。'),
16:((3,2,3),'quantized-deployment','032-qatie','量化感知增强与训练部署差距。可核查低精度实现、输入编码与精炼结构的端侧影响。'),
17:((1,0,1),'detection-transfer','080-detection','籽棉杂质检测。保留直接引用，主要检查轻量/注意力组件的引用来源和任务边界。'),
18:((1,0,1),'detection-transfer','080-detection','焊接缺陷检测。轻量网络模块来源与增强任务不同，方法关联需要正文证据。'),
19:((2,1,1),'industrial-enhancement','044-weld','焊缝DR图像的频域/金字塔增强。可作空间增强先例，不先验认为涉及自然视频或移动端部署。'),
20:((4,3,3),'video-enhancement','010-anymod','明确的低光视频增强及模态无关推理。直接影响视频信息条件、对照选择与创新边界，列最高优先级。'),
21:((2,1,1),'diffusion-image','045-diffusion','协同多域/多尺度扩散低光增强。主要核查与轻量因果视频目标的条件及成本差异。'),
22:((3,2,3),'edge-lightweight-image','033-iot','面向边缘IoT的零样本照明/噪声估计。训练可得性、噪声假设和部署成本具有直接参考价值。'),
23:((2,1,1),'infrared-restoration','046-infrared','外部频率提示与内部统计解耦的红外复原。侧重核实其MobileIE引用，避免把任务迁移当视频机制。'),
24:((1,0,1),'detection-transfer','080-detection','异构安全感知目标检测。需核实引用与尺度/形变建模的具体关系，暂未见直接增强用途。'),
25:((1,0,0),'industrial-imaging','091-ultrasound','复合材料三维超声成像与图像/深度融合。信息传感条件离题较远，仍核查引用范围。'),
26:((1,0,1),'detection-transfer','080-detection','光学遥感目标检测与部分通道注意力。属于跨任务模块引用，不能凭标题认定增强算法贡献。'),
27:((1,0,1),'detection-transfer','080-detection','可见/紫外融合微裂纹检测。多模态缺陷任务需与低光RGB视频的信息条件分开。'),
28:((3,2,3),'lightweight-image-transform','034-bilateral','双边网格的轻量无监督图像增强。变换表示和计算预算可能影响时序设计，需核实具体信息路径。'),
29:((1,0,0),'medical-reconstruction','090-medical','双域稀疏视角CT复原。与手机RGB视频差异大，主要核查为何引用MobileIE。'),
30:((2,1,2),'computational-imaging','047-roi','单镜头红外计算成像的检测门控ROI重建。局部计算与门控预算有参考线索，但不是现成的视频增强对照。'),
31:((2,1,2),'image-isp','048-isp-flow','奖励引导flow的美学ISP。需区分生成/概率flow与运动光流，核查质量目标与部署条件。'),
32:((2,2,2),'underwater-image','049-dual-branch','颜色/结构双分支水下图像复原。与表示分工、颜色稳定性相关，但时域效果尚待核验。'),
33:((3,2,3),'measured-edge-deployment','035-edge-energy','边缘NPU量化水下增强及能耗部署。完整硬件/输入/能量口径可能直接约束效率结论。'),
34:((1,0,0),'segmentation-transfer','092-segmentation','苹果叶病分割的全局局部建模。主要核查轻量模块引用，不把分割收益当增强收益。'),
35:((3,2,2),'interactive-mobile-image','036-ar','移动AR游戏中的照明自适应增强。交互场景与稳定性相关，但是否存在跨帧机制需正文核实。'),
36:((2,1,1),'subject-driven-generation','050-morphsig','同作者主题驱动图像生成与特征传输。标题中的传输不自动代表真实视频一致性；先查引用与机制边界。'),
37:((3,2,3),'mobile-challenge-evaluation','037-mobile-denoise','Mobile AI 2026手机GPU去噪挑战报告。主要价值为部署、计时与质量评价协议的现实参考。'),
38:((3,2,3),'mobile-challenge-evaluation','038-mobile-isp','Mobile AI 2026手机GPU ISP挑战报告。需核硬件、RAW/RGB条件及不同模型的比较范围。'),
39:((1,1,1),'image-fusion-transfer','081-attention-transfer','可见/红外融合的层次注意力。重点核实HDPA等组件来源，不直接当视频增强新机制。'),
40:((1,1,1),'detection-transfer','081-attention-transfer','单目身高估计检测中的HDPA。说明基础模块可能被复用，实际增益与二期任务不等同。'),
41:((3,2,3),'mobile-challenge-evaluation','039-ntire2026','NTIRE 2026高效低光增强挑战。可核MobileIE参照、变体、预算及端侧质量效率协议。'),
42:((3,3,3),'same-author-lightweight-image','023-yuv','同作者YUV轻量低光增强。是结构和效率差异化的重要接续工作，需核是否含时域成分。'),
43:((4,3,3),'original-anchor','000-anchor','MobileIE原始论文与补充材料。先校准架构、训练及GPU/手机效率口径，为所有引用与派生实现建立事实基准。'),
44:((3,2,3),'challenge-original-association','060-ntire2025','NTIRE 2025报告包含MobileIE方案。用于追踪版本、任务和竞赛训练条件，不能直接当视频证据。'),
45:((3,2,3),'mobile-challenge-evaluation','061-rgb2025','Mobile AI 2025手机GPU照片增强报告。为原始端侧部署、计时条件和真实比较提供线索。'),
46:((3,2,2),'review-source-reliability','062-review','明确提及MobileIE的系统综述/时延比较预印本。需优先核引用、模型身份和实测口径，不能把综述说法当独立证实。')}
# repo -> scores, category, family, role, metadata summary
CODE={
'AVC2-UESTC/MobileIE':((4,3,3),'original-anchor','000-anchor',1,'官方固定提交实现。校准训练/重参数化/损失与实际推理路径，并与论文逐点对照。'),
'lhfgghc/LLVE-AMNet':((4,3,3),'video-enhancement','010-anymod',1,'AnyMod-LLVE官方实现名与论文名不同。核查真实输入、历史/未来帧、状态及发布配置。'),
'lhfgghc/LLVE-AMNet-NPU':((4,3,3),'video-deployment','010-anymod',2,'官方README链接的NPU版本。重点核部署路径及其与论文模型的信息/计算条件是否一致。'),
'clarkren/LLVE-AMNet':((3,2,2),'video-implementation-variant','010-anymod',3,'AnyMod代码变体。先辨与官方固定版本的差异，不把fork自动计为独立方法。'),
'nlsidf/vnrit':((4,3,3),'explicit-temporal-claim','011-rust',1,'公开Rust代码同时提及MobileIE启发和时域/运动/闪烁声明。直接核声明是否对应神经模型及实际状态路径。'),
'nlsidf/ndk-vnrit':((4,3,3),'explicit-temporal-claim','011-rust',2,'相关NDK版本。核查与Rust上游、Android封装及所谓抗闪烁机制的实质关系。'),
'KrakeCZ/AIchat-Android-Local-AI-Agent-':((4,3,3),'android-temporal-integration','012-android',1,'Android插件菜单中出现MobileIE/TFLite与时域去噪。需分清注册/菜单、可运行模型及真正组合路径。'),
'LethyZhang/FGDPA':((3,3,3),'same-group-reparameterization','021-fgdpa',1,'同组水下增强代码。核结构重参数化与频率注意力的实现/部署边界。'),
'AVC2-UESTC/DRNet-AiO':((3,3,3),'same-group-reparameterization','022-drnet',1,'同组动态重参数化官方仓库。核发布源码的可得性与论文结构/条件。'),
'LukasYan30/L3IE-YUV':((3,3,3),'same-author-lightweight-image','023-yuv',1,'YUV接续实现。核训练/推理、颜色空间和轻量预算的实际代码条件。'),
'mubaisam/LiteIE':((3,3,3),'lightweight-image','024-liteie',1,'极轻量低光增强实现。核参数统计、迭代步骤和手机速度对应的实现。'),
'LukasYan30/IGDNet':((3,2,2),'same-author-image','025-igdnet',1,'零样本增强/去噪官方实现。核优化过程、单图条件和潜在视频适配开销。'),
'Cloudyu1215/UIE':((3,3,3),'industry-lightweight-image','020-teleai',1,'TeleAI相关水下增强的作者实现。核MobileIE联系、结构与测试/部署条件。'),
'viplab-fudan/FJNDF-Pytorch':((3,2,3),'perceptual-training','031-jnd',1,'感知编码预滤波实现。核JND目标、训练数据与实际部署算子，明确与视频感知评价的距离。'),
'GenAI4E/QATIE':((3,2,3),'quantized-deployment','032-qatie',1,'量化增强实现。核定点/浮点、编码及推理导出路径，而非只看宣称时延。'),
'w-xb/MobileIE-6Ch':((3,3,3),'explicit-mobileie-variant','026-mobileie6ch',1,'以MobileIE命名的挑战赛六通道变体。核与原4K模型、输入和完整推理量的实质区别。'),
'tediyankov/reef-track':((4,2,3),'video-application-integration','013-reeftrack',1,'水下跟踪代码含MobileIE增强实现。核是否逐帧、是否有增强状态，以及跟踪效果能否说明闪烁。'),
'gugugu531/Hispark_SocChina_2026':((3,2,3),'hardware-deployment','063-hispark',1,'HiSilicon相关项目明示MobileIE监督/参考。核算子、模型与端侧成本条件。'),
'Foadsf/neural-enhance-bench':((3,2,3),'mobile-benchmark-implementation','064-bench',1,'移动增强工具含MobileIE命名的处理类。优先核是否真实加载官方模型及其比较是否公平。'),
'Kazuhito00/Multimodal-Node-Editor':((3,1,3),'deployment-integration','065-node',1,'节点编辑器有MobileIE ONNX入口。核尺寸、前后处理及图像/视频调用条件。'),
'SKKUAutoLab/ETSS-07-ImageEnhance':((3,1,3),'framework-integration','066-framework',1,'综合增强框架收录MobileIE代码和配置。核上游版本、封装差异与评价条件。'),
'khaihoang004/dewarp':((3,2,2),'restoration-implementation-variant','067-dewarp',1,'去畸变项目中的MobileIE命名精炼模块。核原模型忠实度与具体应用路径。'),
'38260/MobileIE-repro':((3,2,3),'reproduction-and-deployment','068-repro',1,'独立复现项目。核其实际复现范围、差异和材料可靠性，不能将说明等同独立实验。'),
'Barry1749/MobileIE':((3,2,2),'mobileie-implementation-variant','069-forks',1,'含明显源码/配置差异的fork。大归档未完整取得，固定树文本已另取；实质改动待核。'),
'wenqi-deng/Computational-Photography-26Fall-MobileIE':((2,1,2),'course-reproduction','070-course',1,'计算摄影课程相关复现/变体。核其独立改动、训练条件和原MobileIE版本关系。'),
'horizon921/Computational-Photography-26Fall-MobileIE':((2,1,2),'course-reproduction','070-course',1,'同主题课程项目的不同快照。需逐项比较，不能按名称预设与其他复现完全相同。'),
'LukasYan30/MobileIE':((2,1,1),'mobileie-version-provenance','071-version',1,'原作者个人fork的不同树。主要核版本历史，而非预设新方法。'),
'songyang86/MobileIE':((2,1,1),'mobileie-version-provenance','071-version',1,'不同树的MobileIE fork。核其是否仅早期版本/文档变化。'),
'strategist922/MobileIE':((2,1,1),'mobileie-version-provenance','071-version',1,'不同树的MobileIE fork。辨别版本差异和独立技术内容。'),
'miaory/test':((2,1,2),'conversion-implementation','072-export',1,'ONNX导出文件中出现MobileIE类/路径。核有效代码与注释/样例的区别。'),
'AVC2-UESTC/NTIRE2025-LLIE':((2,1,1),'challenge-repository','073-challenge-code',1,'官方挑战仓库。需核是否实际发布方法源码，不能以官方仓库存在代替代码可得。'),
'AIGeeksGroup/StereoAdapter':((2,1,2),'underwater-transfer','040-stereo',1,'水下立体适配作者代码。核增强前处理关系和信息条件，非直接时域增强结论。'),
'LukasYan30/MorphSIG':((2,1,1),'subject-driven-generation','050-morphsig',1,'主题生成作者代码。核所谓特征传输是否为视频机制，避免名称相近造成误判。'),
'W-ayivi/ML-ConvNet':((1,0,0),'medical-classification','090-medical',1,'既有被引医学论文原文明确给出的作者代码。核引用/规模比较语境，不预设复用了MobileIE模型。')}
OTHER={
'other-mobileie-explanations':((2,1,1),'secondary-source-reliability','074-explain','中英文解读/转载族。主要核来源链与效率口径误传，不把多次转述当独立证据。'),
'other-mobileie-reading-notes':((2,1,2),'training-notes','075-notes','公开阅读笔记将MobileIE与蒸馏/训练部署联系。核笔记实际支持范围，不当作已有实验。'),
'other-mobileie-rtdetr-port':((1,1,1),'module-port-provenance','082-ports','RT-DETR公开变更说明声称移植MBRConv3。核公开说明、可得代码与商业材料边界。'),
'other-heniath--underwater-image-enhancement':((3,1,2),'application-survey','076-underwater-notes','水下增强项目中的文献/预算比较资料。可核设备、分辨率与来源口径，不把汇编表当同轮实测。'),
'other-TheComplianceAide--Drone_EnhancedVision':((3,2,2),'video-application-notes','077-drone','夜视/无人机项目的研究综合资料将MobileIE纳入实时图像/视频路线。核论据与实际实现范围。'),
'other-Synapse-iiserk--beautiful_ppt':((2,1,1),'presentation-material','078-present','水下AI演示材料涉及MobileIE。主要用于核对解释与技术来源，不预设有独立方法贡献。'),
'other-dnguyennk--portfolio':((2,1,1),'application-portfolio','079-portfolio','公开项目展示涉及MobileIE。核展示指向、实现可得性及可审查结果。'),
'other-38260--topconf-4-pillars':((2,1,1),'structured-paper-notes','079-paper-notes','结构化论文分析/核验项目收录MobileIE。核原文引用和自动生成解释的证据可靠性。')}
def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def main():
 gate=json.loads((C/'gate1-review001.json').read_text());assert gate['gate1_passed']
 source=C/'candidate-resources.json';assert digest(source)==gate['candidate_sha256'];raw=json.loads(source.read_text());assert len(raw)==113
 assert not (D/'index.json').exists(),'Permanent IDs already exist; do not regenerate/reassign'
 avail={r['temporary_key']:r for r in json.loads((C/'resource-acquisitions.json').read_text())};rows=[]
 for i,r in enumerate(raw):
  if r['type']=='paper':scores,category,family,summary=P[i];role=0
  elif r['type']=='other':scores,category,family,summary=OTHER[r['temporary_key']];role=3
  elif r['repo'] in CODE:scores,category,family,role,summary=CODE[r['repo']]
  else:
   # Remaining entries were explicitly admitted as module/HDPA/MBRConv ports or source dumps.
   assert r['repo'] in {'369pro/DEIMv8','Alleluia1/RTDETR-10.20','Alleluia1/RTDETR-main-distill1','FangruLinghu/PaperCode','FifthGentleness/RTDETRU','Wzhi997/YOLO','XuYuanDaWang/Yolo-Project','benderick/DEIM-DEIM','ccz308476-ai/General_yu','cui123000/rtdetr_indoor','daidaidaidaidaidai-cat/duomotai-yolo','gogoworker/FSOD','helloshuaige520/RT-DETR','lightxiu/yolo26','linchuan232/myrtdetr','lrbob/deim','mahongwei-bot/RTDETR','mahongwei-bot/RTDETR20260307-main','pbobip/wd_bishe','pls-Kaguya/LSO-RT-DETR','tpiece/RTDETR-main','wbzuo/dl-modules-toolkit','wujunye1212/wjy_vault','xuhangc/Plumbing'}
   scores=(1,1,1);category='module-transfer-or-source-dump';family='082-ports';role=2
   summary=r['repo']+' 的公开命中涉及MobileIE的MBRConv/HDPA/模型定义或配置。逐项核查具体复用/抄录与实现条件，不将它当独立视频方法。'
  ar=avail[r['temporary_key']];e=1;why_e='已有身份/检索资料，材料充分性未深读核验'
  for x in ar['receipts']:
   f=B/x['archive'] if x.get('archive') else None
   if x['status']=='success' and x.get('http')==200 and f and f.exists():
    b=f.read_bytes()[:10]
    if b.startswith((b'%PDF',b'PK\x03\x04')):e=3;why_e='已有PDF或完整固定提交包；不等于已读或实现有效';break
    e=max(e,2);why_e='已有可读HTML/文本/局部固定源码；内容充分性留待Gate2'
  score=10*sum(scores);tier='A' if score>=70 else 'B' if score>=40 else 'C'
  rows.append(dict(**r,preliminary=dict(R=scores[0],N=scores[1],U=scores[2],E=e,score=score,tier=tier,category=category,family=family,role_order=role,summary=summary,reason_R='任务联系：'+summary,reason_N='重复/混淆风险为初级待证判断；依据当前明确关联与基础/接续/移植角色，非已确立创新结论',reason_U='可操作性：按所列机制、对照或来源核查用途评分；未运行外部代码',reason_E=why_e,evidence=r['sources'],not_gate2=True)))
 rows.sort(key=lambda r:(-r['preliminary']['score'],r['preliminary']['family'],r['preliminary']['role_order'],r['temporary_key']))
 now=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
 for i,r in enumerate(rows,1):r.update(resource_id=f'MIE-{i:04d}',analysis_order=i,analysis_status='pending',indexed_at=now)
 D.mkdir(exist_ok=True)
 (D/'index.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 (D/'order-v1.json').write_text(json.dumps({'created_at':now,'policy':'docs/0004-初级排序与静态索引规则-26112012.md','gate1':'collection/gate1-review001.json','order':[r['resource_id'] for r in rows],'no_resource_analysis_started':True},ensure_ascii=False,indent=2)+'\n')
 freeze={'created_at':now,'files':[dict(path=str(f.relative_to(B)),sha256=digest(f)) for f in [D/'index.json',D/'order-v1.json',B/'docs/0004-初级排序与静态索引规则-26112012.md']]}
 (D/'freeze-v1.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
 text='# 永久资源索引与初级价值排序\n\n'+now+'；Codex  (sqs_A100)；Gate1通过后初筛，不是Gate2结果。E不参与排序，未取得全文不自动低价值。\n\n|顺序/永久ID|级别/分|类型|资源|初级内容与价值依据|\n|---|---|---|---|---|\n'
 for r in rows:text+=f"|{r['resource_id']}|{r['preliminary']['tier']} / {r['preliminary']['score']}|{r['type']}|{r['title'].replace('|','/')}|{r['preliminary']['summary'].replace('|','/')}|\n"
 (D/'README.md').write_text(text)
 print('Permanent IDs allocated once:',len(rows),'tiers',dict(collections.Counter(r['preliminary']['tier'] for r in rows)))
 for r in rows[:14]:print(r['resource_id'],r['preliminary']['score'],r['title'])
if __name__=='__main__':main()
