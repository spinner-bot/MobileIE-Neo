"""把 citations_s2.json 规范化成 CSV，并附上主题分类与相关性。

用法：
    python research/5.prior-art-survey/scripts/normalize_citations.py

分类为人工标注（2026-11-05 版本），新增引用时需要手动补充 CATEGORY 表；
未命中的条目会标为 other，便于发现新工作。
"""
import csv
import json
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# idx（S2 引用列表顺序）-> (category, relevance, note)
CATEGORY = {
    1: ('contest', '中', 'Mobile AI 2026 ISP 竞赛报告，MobileIE 作为基线'),
    2: ('contest', '中', 'Mobile AI 2026 去噪竞赛报告，MobileIE 作为基线'),
    3: ('image_generation', '低', '同作者组，图像生成；提到 pseudo-video，但与闪烁无关'),
    4: ('detection', '低', 'YOLO 检测，借用 HDPA 注意力'),
    5: ('segmentation', '低', '分割，无关'),
    6: ('isp', '中', '审美 ISP，图像任务'),
    7: ('llie', '中', 'IoT 端侧零样本低光增强'),
    8: ('medical', '低', 'CT 重建，无关'),
    9: ('detection', '低', '多模态缺陷检测，无关'),
    10: ('application', '中', '移动 AR/vSLAM 中用 MobileIE 做逐帧增强 + 自适应触发，未处理时序一致性'),
    11: ('infrared', '低', '红外复原，无关'),
    12: ('deployment', '中', '边缘 NPU 量化部署测量，强调延迟/能耗而非 params/FLOPs'),
    13: ('direct_followup', '高', '同组接续：实时水下增强，频域双路径，图像任务'),
    14: ('video', '高', '视频低光增强（多模态、重模型），唯一明确的视频 LLVE 引用'),
    15: ('enhancement', '低', '焊接 DR 图像增强'),
    16: ('llie', '中', '低光扩散增强，图像'),
    17: ('detection', '低', '焊接缺陷检测'),
    18: ('direct_followup', '高', '同组接续：动态重参数化 all-in-one 图像复原'),
    19: ('contest', '中', 'NTIRE 2026 高效低光增强竞赛，MobileIE 基线'),
    20: ('medical', '低', '医学分类，无关'),
    21: ('detection', '低', '棉花杂质检测'),
    22: ('detection', '低', '船舶检测'),
    23: ('deployment', '中', '量化感知增强，训练-部署差距'),
    24: ('deployment', '中', '移动端实时去噪网络'),
    25: ('uie', '中', '水下增强，实时'),
    26: ('direct_followup', '高', '同第一作者接续：YUV 空间轻量低光增强（单图）'),
    27: ('detection', '低', '电梯跌倒检测'),
    28: ('detection', '低', '风机叶片检测'),
    29: ('perceptual', '中', 'JND 引导的轻量预滤波（编码），与可见性阈值相关'),
    30: ('underwater', '低', '水下立体深度'),
    31: ('direct_followup', '高', '同作者接续：LiteIE 超轻无监督低光增强（单图）'),
    32: ('direct_followup', '高', '同第一作者：IGDNet 零样本低光增强（单图）'),
    33: ('llie', '中', '两阶段可控曝光低光增强'),
}


def main():
    papers = json.load(open(os.path.join(BASE, 'citations_s2.json'), encoding='utf-8'))
    rows = []
    for i, p in enumerate(papers, 1):
        ids = p.get('externalIds') or {}
        cat, rel, note = CATEGORY.get(i, ('other', '待定', ''))
        rows.append({
            'idx': i,
            'title': p.get('title') or '',
            'year': p.get('year') or '',
            'venue': p.get('venue') or '',
            'doi_or_arxiv': ids.get('DOI') or ids.get('ArXiv') or '',
            'authors': '; '.join(a['name'] for a in (p.get('authors') or [])),
            'category': cat,
            'relevance': rel,
            'note': note,
        })
    out = os.path.join(BASE, 'citations.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    unknown = [r['idx'] for r in rows if r['category'] == 'other']
    print('wrote %s (%d rows)' % (out, len(rows)))
    if unknown:
        print('unclassified idx (please update CATEGORY):', unknown)


if __name__ == '__main__':
    main()
