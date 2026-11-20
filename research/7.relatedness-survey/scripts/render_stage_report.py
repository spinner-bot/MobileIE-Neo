"""Render the requested stage report from local Markdown; never advance research gates.

Small renderer for this report's headings, paragraphs, tables, quotes and inline links.
No external package/network dependency. The Markdown remains the content authority.
"""
from pathlib import Path
import hashlib
import html
import json
import re

B = Path(__file__).resolve().parents[1]
MD = B / 'report/0001-阶段工作汇报-20261120.md'
OUT = MD.with_suffix('.html')


def inline(value):
    value = html.escape(value)
    value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', value)


def main():
    source = MD.read_text()
    snapshot = json.loads((B / 'report/0001-stage-snapshot.json').read_text())
    lines = source.splitlines()
    pieces, toc = [], []
    i, section = 0, 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('# '):
            i += 1
            continue
        if line.startswith('## '):
            if section:
                pieces.append('</section>')
            section += 1
            title = line[3:]
            sid = f'section-{section}'
            toc.append(f'<a href="#{sid}">{inline(title)}</a>')
            pieces.append(f'<section id="{sid}"><h2>{inline(title)}</h2>')
            i += 1
            continue
        if line.startswith('### '):
            pieces.append(f'<h3>{inline(line[4:])}</h3>')
            i += 1
            continue
        if line.startswith('> '):
            pieces.append(f'<aside class="notice">{inline(line[2:])}</aside>')
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in row):
                    rows.append(row)
                i += 1
            catalog = rows[0][0] == '顺序'
            attrs = ' id="catalog-table"' if catalog else ''
            if catalog:
                pieces.append('<label class="filter">查找目录 <input id="catalog-search" type="search" placeholder="输入ID、标题、等级或状态" aria-label="筛选资源目录"></label><p id="catalog-count" class="muted" aria-live="polite">124 个资源</p>')
            pieces.append(f'<div class="scroll-hint">窄屏可左右滑动查看完整表格</div><div class="table-wrap" tabindex="0" role="region" aria-label="数据表，可横向滚动"><table{attrs}><thead><tr>')
            pieces.extend('<th scope="col">' + inline(c) + '</th>' for c in rows[0])
            pieces.append('</tr></thead><tbody>')
            for row in rows[1:]:
                pieces.append('<tr>' + ''.join('<td>' + inline(c) + '</td>' for c in row) + '</tr>')
            pieces.append('</tbody></table></div>')
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '> ')):
            paragraph.append(lines[i])
            i += 1
        pieces.append('<p>' + inline(' '.join(paragraph)) + '</p>')
    if section:
        pieces.append('</section>')
    digest = hashlib.sha256(source.encode()).hexdigest()
    style = '''
:root{--ink:#152a3b;--muted:#506478;--accent:#067a72;--line:#d6e1e7;--paper:#fff;--back:#eff4f6;--warn:#7a4809}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:20px}body{margin:0;background:var(--back);color:var(--ink);font:16px/1.85 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}a{color:#006c73;text-underline-offset:3px}a:hover{color:#aa5400}a:focus-visible,input:focus-visible{outline:3px solid #d58c23;outline-offset:3px}code{font: .84em/1.6 ui-monospace,monospace;background:#eaf1f3;padding:.12em .32em;border-radius:4px;overflow-wrap:anywhere}strong{font-weight:730}.hero{color:#f2f8fa;background:linear-gradient(120deg,#102538,#173d50);padding:48px max(24px,calc((100vw - 1300px)/2)) 37px}.eyebrow{letter-spacing:.15em;font-size:12px;color:#8fd4cc;font-weight:700}.hero h1{font-size:clamp(27px,4.2vw,44px);line-height:1.3;margin:13px 0}.hero p{max-width:880px;color:#d1e1e9;margin:8px 0}.status{display:inline-block;color:#ffe1a3;border:1px solid #7c714e;border-radius:20px;padding:3px 12px;font-size:13px;margin-top:13px}.metrics{max-width:1300px;margin:24px auto 0;display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:0 24px}.metric{background:white;border:1px solid var(--line);border-top:3px solid var(--accent);padding:17px 20px;border-radius:9px}.metric b{display:block;font-size:30px;line-height:1.3;font-variant-numeric:tabular-nums}.metric span{font-size:13px;color:var(--muted)}.progress{max-width:1252px;margin:14px auto 0;height:5px;background:#d8e5e6;border-radius:5px;overflow:hidden}.progress div{height:100%;width:10.4838709677%;background:var(--accent)}.layout{display:grid;grid-template-columns:235px minmax(0,1fr);gap:30px;max-width:1300px;padding:28px 24px 60px;margin:auto}.side{position:sticky;top:18px;align-self:start;max-height:calc(100vh - 36px);overflow:auto;font-size:13px}.side summary{font-weight:750;margin-bottom:12px;cursor:pointer}.side nav{display:flex;flex-direction:column;gap:6px}.side nav a{text-decoration:none;padding:6px 10px;border-left:2px solid #cad9df;color:#345264}.side nav a:hover{background:#dceceb;border-color:var(--accent)}.side .note{margin-top:23px;color:var(--muted);font-size:12px}main{min-width:0;background:var(--paper);border:1px solid var(--line);border-radius:11px;padding:34px 40px;box-shadow:0 8px 25px #17304407}main>p:first-child{font-size:13px;color:var(--muted)}section{padding-top:18px;margin-top:26px;border-top:1px solid var(--line)}h2{font-size:24px;line-height:1.5;margin:5px 0 21px;letter-spacing:-.02em}h3{font-size:18px;margin:27px 0 9px;color:#086960}p{margin:12px 0;overflow-wrap:anywhere}.notice{background:#fff7e8;border-left:4px solid #c98922;border-radius:4px;padding:19px 21px;color:#67400f;margin:26px 0}.table-wrap{overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.7}th{background:#eaf3f3;color:#174c51;text-align:left;padding:12px;font-weight:730}td{padding:13px 12px;border-top:1px solid #e4ebee;vertical-align:top;overflow-wrap:anywhere}tbody tr:nth-child(even){background:#f8fafb}td:first-child{font-weight:650;min-width:75px}#catalog-table{font-size:12px}#catalog-table td:nth-child(2){white-space:nowrap}.filter{display:block;font-size:14px;color:var(--muted);margin-top:20px}.filter input{font:inherit;margin:6px 0 0;padding:10px 12px;border:1px solid #9cb5bf;border-radius:5px;width:100%;max-width:450px;display:block}.scroll-hint{display:none;font-size:12px;color:var(--muted);margin-top:16px}.muted{font-size:12px;color:var(--muted)}footer{border-top:1px solid var(--line);padding-top:23px;margin-top:32px;font-size:12px;color:var(--muted);overflow-wrap:anywhere}.pill{padding:2px 6px;background:#e7f3ef;border-radius:4px}button{font:inherit}@media(max-width:850px){.layout{grid-template-columns:1fr;gap:16px}.side{position:static;max-height:none}.side nav{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}.side .note{display:none}main{padding:24px}.metrics{gap:9px}.metric{padding:13px}.metric b{font-size:26px}.progress{margin-left:24px;margin-right:24px}}@media(max-width:480px){body{font-size:15px;line-height:1.85}.hero{padding:30px 18px}.metrics{grid-template-columns:repeat(2,1fr);padding:0 14px}.layout{padding:19px 12px 35px}main{padding:21px 17px}.side nav{grid-template-columns:1fr}h2{font-size:21px}.notice{padding:16px}.scroll-hint{display:block}.table-wrap{margin-left:0;margin-right:0;margin-top:6px}.table-wrap table{min-width:660px}#catalog-table{min-width:760px}th,td{padding:10px 12px}td{min-width:90px}td:last-child{min-width:170px}.progress{margin-left:14px;margin-right:14px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{body{background:#fff;font-size:10pt}.hero{padding:20px;background:#123345!important;print-color-adjust:exact}.hero h1{font-size:25pt}.metrics{padding:0}.metric b{font-size:20pt}.side,.filter,#catalog-count,.progress{display:none}.layout{display:block;padding:0;max-width:none}main{border:0;box-shadow:none;padding:10px}h2,h3{break-after:avoid}tr{break-inside:avoid}.table-wrap{overflow:visible}table{font-size:8pt}a{color:#174c51}section{margin-top:18px}}
'''
    script = '''
const input=document.querySelector('#catalog-search');
input.addEventListener('input',()=>{const q=input.value.toLowerCase().trim();let shown=0;document.querySelectorAll('#catalog-table tbody tr').forEach(row=>{row.hidden=!row.textContent.toLowerCase().includes(q);if(!row.hidden)shown++});document.querySelector('#catalog-count').textContent=shown+' / 124 个资源'});
if(window.innerWidth<=850)document.querySelector('.side details').open=false;
'''
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="MobileIE相关性调研阶段工作汇报：13/124项结项、证据、Gate限制、完整目录与未完成工作。"><title>MobileIE相关性调研 · 阶段工作完整汇报</title><style>{style}</style></head>
<body><header class="hero"><div class="eyebrow">MOBILEIE-NEO / SURVEY · STAGE 01</div><h1>相关性调研<br>阶段工作完整汇报</h1><p>从单帧照片走向视频：已取得的证据、尚存的缺口与可审查的推进记录。</p><span class="status">阶段报告 · 调研目标已暂停并清除 · G1-A0002待审</span></header>
<div class="metrics" aria-label="阶段统计"><div class="metric"><b>13 / 124</b><span>Gate2已结项 · 10.48%</span></div><div class="metric"><b>141</b><span>主要结论证据记录</span></div><div class="metric"><b>1,355</b><span>统计快照内提交 · 不含报告提交</span></div><div class="metric"><b>0</b><span>模型 / 训练 / GPU / Gate3运行</span></div></div><div class="progress" aria-hidden="true"><div></div></div>
<div class="layout"><aside class="side"><details open><summary>报告导航 / CONTENTS</summary><nav aria-label="章节导航">{''.join(toc)}</nav></details><p class="note">截止快照 2026-11-17<br>资料核验 2026-11-20<br>报告复核 2026-11-21<br>50 论文 / 65 代码 / 9 其他<br><br><a href="{MD.name}">阅读Markdown原文</a><br><a href="0001-stage-snapshot.json">统计快照</a></p></aside><main>{''.join(pieces)}<footer>Codex  (sqs_A100) &lt;langxibielangle@qq.com&gt;<br>内容源SHA-256：{digest}<br>统计快照提交：{snapshot['snapshot_commit']}。HTML样式与筛选不改变正文；完整原件依赖工作站资产目录。此报告不解除Gate限制或启动后续实验。</footer></main></div><script>{script}</script></body></html>'''
    OUT.write_text(document)
    print('Rendered', OUT.relative_to(B), '; content SHA256', digest)


if __name__ == '__main__':
    main()
