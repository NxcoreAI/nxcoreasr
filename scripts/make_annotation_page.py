"""生成说话人 truth 标注页 v2（自包含 HTML：内嵌压缩音频 + 逐句列表）。

用法：
  .venv/bin/python scripts/make_annotation_page.py <case_dir> [--run NAME] [--truth FILE]

--truth 预载已有人工标注（标签按原样填好，只补改不重打），文本按 start_ms
从 runs 逐句对齐。页面支持：点行跳播、下拉改说话人、批量设标签、
"在播放处拆分"（句内换人：听到切换点暂停后点它，句子变两半分别设标签）、
"并入上一行"（撤销拆分）、导出 truth.jsonl。
"""
import argparse
import base64
import html
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--run", default="wt0.65_st0.65_me4000")
    ap.add_argument("--truth", default=None, help="预载已有人工标注文件")
    a = ap.parse_args()
    case = Path(a.case_dir)
    utts = [json.loads(l) for l in
            (case / "runs" / a.run / "utterances.jsonl").read_text().splitlines()]

    if a.truth:  # 预载人工标注：标签来自 truth，文本按 start_ms 对齐回逐句
        trows = [json.loads(l) for l in Path(a.truth).read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        by_start = {u["start_ms"]: u["text"] for u in utts}
        rows = [{"start": t["start_ms"], "end": t["end_ms"],
                 "spk": t["speaker"], "text": by_start.get(t["start_ms"], "")}
                for t in trows]
    else:
        from collections import Counter
        ms = Counter()
        for u in utts:
            ms[u["person"]] += u["end_ms"] - u["start_ms"]
        order = {p: chr(ord("A") + i) for i, (p, _) in enumerate(ms.most_common())}
        rows = [{"start": u["start_ms"], "end": u["end_ms"],
                 "spk": order[u["person"]], "text": u["text"]} for u in utts]

    labels = sorted({r["spk"] for r in rows} | set("ABCDEFGH"[:7]))

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
        tmp = f.name
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(case / "cache" / "full.wav"),
                    "-c:a", "aac", "-b:a", "48k", "-ac", "1", tmp], check=True)
    audio_b64 = base64.b64encode(Path(tmp).read_bytes()).decode()
    Path(tmp).unlink()

    title = case.name
    page = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>__TITLE__ 标注</title>
<style>
 body{font:14px/1.6 -apple-system,"PingFang SC",sans-serif;margin:24px;max-width:920px}
 audio{width:100%;position:sticky;top:0;background:#fff;z-index:2}
 .row{display:flex;gap:6px;padding:4px 6px;border-bottom:1px solid #eee;cursor:pointer;align-items:center}
 .row:hover{background:#f6f8ff}.row.cur{background:#e8efff}
 .row.split{border-left:3px solid #e67e22}
 .t{color:#888;font-variant-numeric:tabular-nums;white-space:nowrap;width:110px}
 select{width:56px}
 .stats{position:sticky;top:48px;background:#fff;padding:6px 0;font-size:12px;color:#555;z-index:1}
 button{margin:0;font-size:12px;padding:1px 6px}
 .txt{color:#333}
</style></head><body>
<h3>__TITLE__ 说话人标注（v2 支持句内拆分）</h3>
<audio id="au" controls preload="auto" src="data:audio/mp4;base64,__AUDIO__"></audio>
<div class="stats" id="stats"></div>
<p><b>用法</b>：点行跳播；下拉改标签；"批量"= 该行起到下一手动改过的行都设为此标签。
<b>句内换人</b>：播到切换点暂停 → 点该行"拆分@播放"（句子变两半，各自改标签，
左边框变橙色的行是拆出来的）；拆错点"并入上一行"。全改完导出。</p>
<div id="rows"></div>
<button onclick="exportTruth()">导出 truth.jsonl</button>
<script>
const ROWS = __ROWS__;
const LABELS = __LABELS__;
const au = document.getElementById('au');
const box = document.getElementById('rows');
let cur = -1;
function fmt(ms){const s=ms/1000;return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(Math.floor(s%60)).padStart(2,'0')}.${String(Math.floor(ms%1000)).padStart(3,'0')}`}
function render(){
  box.innerHTML='';
  ROWS.forEach((r,i)=>{
    const d=document.createElement('div');
    d.className='row'+(i===cur?' cur':'')+(r.split?' split':'');
    d.innerHTML=`<span class="t">${fmt(r.start)}</span>
      <select data-i="${i}">${LABELS.map(l=>`<option ${l===r.spk?'selected':''}>${l}</option>`).join('')}</select>
      <button data-a="batch" data-i="${i}">批量</button>
      <button data-a="split" data-i="${i}">拆分@播放</button>
      <button data-a="merge" data-i="${i}" ${i?'':'disabled'}>并入上一行</button>
      <span class="txt">${r.text}</span>`;
    d.onclick=e=>{if(e.target.tagName==='SELECT'||e.target.tagName==='BUTTON')return;jump(i)};
    d.querySelector('select').onchange=e=>{ROWS[i].spk=e.target.value;stats()};
    d.querySelectorAll('button').forEach(b=>b.onclick=e=>{
      const i=+b.dataset.i, a=b.dataset.a;
      if(a==='batch'){const l=ROWS[i].spk;let j=i+1;
        while(j<ROWS.length&&ROWS[j].spk===ROWS[i].spk){ROWS[j].spk=l;j++}render()}
      else if(a==='split'){
        const r=ROWS[i];
        if(r.end-r.start<800){alert('句子太短（<0.8s）不能拆');return}
        const t=Math.round(au.currentTime*1000);
        const cut=Math.min(Math.max(t,r.start+300),r.end-300);
        ROWS.splice(i,1,{start:r.start,end:cut,spk:r.spk,text:'',split:1},
                          {start:cut,end:r.end,spk:r.spk,text:r.text});
        jump(i+1);render()}
      else if(a==='merge'&&i>0){ROWS[i-1].end=ROWS[i].end;ROWS.splice(i,1);render()}
      stats()});
    box.appendChild(d)});
}
function jump(i){cur=i;au.currentTime=ROWS[i].start/1000;au.play();
  [...box.children].forEach((c,k)=>c.classList.toggle('cur',k===i))}
function stats(){const s={};ROWS.forEach(r=>s[r.spk]=(s[r.spk]||0)+r.end-r.start);
  document.getElementById('stats').textContent=
    '共 '+ROWS.length+' 行（拆分会计入）| 各标签秒数: '+
    Object.entries(s).sort().map(([k,v])=>k+':'+(v/1000).toFixed(0)+'s').join('  ')}
function exportTruth(){
  const lines=ROWS.map(r=>JSON.stringify({start_ms:r.start,end_ms:r.end,speaker:r.spk}));
  const b=new Blob([lines.join('\\n')+'\\n'],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='truth.jsonl';a.click()}
render();stats();
</script></body></html>"""
    page = (page.replace("__TITLE__", html.escape(title))
               .replace("__AUDIO__", audio_b64)
               .replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
               .replace("__LABELS__", json.dumps(labels)))
    out = case / "annotate.html"
    out.write_text(page)
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(rows)} 句, 标签 {labels})")


if __name__ == "__main__":
    main()
