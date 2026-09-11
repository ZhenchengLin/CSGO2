"""Render the controlled V0 Markdown chapter into the existing static site shell.

No third-party dependencies. Supports the paragraph, heading, list, fenced code,
link, emphasis, tables, and named math fences used in docs/v0_explained.md.
Math fences use native MathML definitions in research_equations.py.
"""
from pathlib import Path
import html
import json
import re
import shutil
from research_equations import render_equations, format_inline_equations

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
DOC = ROOT / 'docs/v0_explained.md'

def inline(value):
    value = html.escape(value, quote=False)
    codes = []
    def save_code(match):
        codes.append('<code>' + match[1] + '</code>')
        return f'@@CODE{len(codes)-1}@@'
    value = re.sub(r'`([^`]+)`', save_code, value)
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: '<a href="' + html.escape(html.unescape(m[2]), quote=True) + '">' + m[1] + '</a>', value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
    for i,code in enumerate(codes): value = value.replace(f'@@CODE{i}@@',code)
    return format_inline_equations(value)

def anchor(heading):
    special = {'03':'timing','06':'features','10':'evaluation','13':'xgboost','16':'calibration','17':'errors','20':'uncertainty','21':'limitations','22':'v1','23':'sources'}
    return special.get(heading[:2], 'chapter-' + heading[:2])

def render(markdown):
    lines = markdown.splitlines(); out=[]; sections=[]; i=0; section_open=False
    while i < len(lines):
        line=lines[i]
        if not line.strip(): i+=1;continue
        if line.startswith('# '): i+=1;continue
        if line.startswith('## '):
            if section_open: out.append('</section>')
            heading=line[3:]; ident=anchor(heading); number,title=heading.split(' · ',1)
            sections.append((ident,number,title))
            out.append(f'<section id="{ident}"><span class="chapter-number">CHAPTER {number}</span><h2>{inline(title)}</h2>');section_open=True;i+=1;continue
        if line.startswith('```'):
            language=line[3:].strip()
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            if language.startswith('math '):
                out.append(render_equations(language.split(maxsplit=1)[1]))
            else:
                out.append('<pre><code>'+html.escape('\n'.join(block))+'</code></pre>')
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                i+=1
            out.append('<div class="table-wrap" role="region" aria-label="'+html.escape(sections[-1][2] if sections else 'Research data',quote=True)+' data table" tabindex="0"><table><thead><tr>'+''.join('<th scope="col">'+inline(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>')
            for row in rows[1:]:out.append('<tr'+(' class="best"' if row[0]=='XGB-A5' else '')+'>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>')
            out.append('</tbody></table></div>');continue
        if line.startswith('- '):
            out.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>');continue
        para=[]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('## ','```','|','- ')):para.append(lines[i]);i+=1
        out.append('<p>'+inline(' '.join(para))+'</p>')
    if section_open:out.append('</section>')
    return '\n'.join(out),sections

markdown=DOC.read_text()
article,sections=render(markdown.replace("(../site/pages/evidence.html)", "(evidence.html)"))
title='Understanding V0'
lead='The complete design story: what we predicted, how we investigated the data, why each model decision followed, and what the evidence actually supports.'
nav='<aside class="chapter-nav" aria-label="On this page"><strong>Inside this chapter</strong><div class="read-progress" aria-hidden="true"><i></i></div>'+''.join(f'<a href="#{ident}">{num} · {inline(name)}</a>' for ident,num,name in sections)+'</aside>'
body=f'<header class="hero"><div class="kicker">Version 0 / The complete explanation</div><h1>Understanding V0</h1><p class="lead">{lead}</p><div class="badges"><span class="badge ok">Development evaluation frozen</span><span class="badge">{len(sections)} chapters</span><span class="badge ok">Live path: synthetic E2E passed</span></div></header><div class="page-note">Read in order for the complete reasoning, or jump to a chapter. <a href="../assets/sources/v0_explained.md" download>Download the full guide ↓</a> · <a href="evidence.html">Open the evidence tables ↗</a></div><div class="reading-layout"><article class="article">{article}</article>{nav}</div>'
template=(SITE/'pages/models.html').read_text()
template=re.sub(r'<title>.*?</title>',f'<title>{title} — CS2 Tactical Intelligence Lab</title>',template)
template=re.sub(r'<meta name="description" content="[^"]*">','<meta name="description" content="'+lead+'">',template)
template=template.replace('data-page="models"><a class="skip-link"','data-page="v0"><a class="skip-link"')
template=template.replace(' class="active" aria-current="page"','')
template=template.replace('data-page="v0" href="../pages/v0.html"','data-page="v0" href="../pages/v0.html" class="active" aria-current="page"')
template=template.replace('The notebook <span>/</span> Models','The notebook <span>/</span> Understanding V0')
start=template.index('<div class="content">')+len('<div class="content">')
end=template.index('<footer class="footer">',start)
(SITE/'pages/v0.html').write_text(template[:start]+body+template[end:])
(SITE/'assets/sources').mkdir(parents=True,exist_ok=True)
for name in ['v0_explained.md','v0_evaluation_summary.md','v0_feature_candidate_matrix.md','v0_feature_evidence.md']:
    shutil.copyfile(ROOT/'docs'/name,SITE/'assets/sources'/name)
(SITE/'assets/data').mkdir(parents=True,exist_ok=True)
for source in sorted((ROOT/'artifacts').glob('v0_*.csv')):
    shutil.copyfile(source,SITE/'assets/data'/source.name)
manifest=json.loads((SITE/'site-manifest.json').read_text())
manifest.update(stage='V0 evaluation frozen; live engineering validated synthetically',v0_status='EVALUATION FROZEN · REAL GSI VALIDATION NEXT',selected_model='XGB-A5',observation_scope='Mirage, full observer, 10/20/30/40 seconds after freeze_end',pages=['index.html']+['pages/'+p+'.html' for p in ['v0','evidence','models','data','roadmap','decisions','research','build-log','journey']],explanation_chapters=len(sections))
(SITE/'site-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Rendered {len(sections)} V0 chapters ({len(markdown.split()):,} words) and copied frozen evidence.')
