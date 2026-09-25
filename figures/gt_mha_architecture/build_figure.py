from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
S = []
def add(s): S.append(s)
def text(x,y,s,size=24,color='#182c44',anchor='start',weight='normal',math=False):
    add(f'<text x="{x}" y="{y}" font-family="{"STIX Two Text, Times New Roman, serif" if math else "Arial, Helvetica, sans-serif"}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')
def box(x,y,w,h,fill='#ffffff',stroke='#cad5e0',rx=12):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def path(d,color='#526579',width=2.5,arrow=False,dash=False):
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'+(' marker-end="url(#orange)"' if arrow and color=='#d77522' else ' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="7 6"' if dash else '')+'/>')
def dot(x,y,c='#24609b',r=6): add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" stroke="white" stroke-width="2"/>')

add('''<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="960" viewBox="0 0 1680 960" role="img" aria-labelledby="title desc">
<title id="title">Group-transformed multi-head attention architecture</title>
<desc id="desc">C learned base projection triples supply H attention heads. Shared query-key and value generator banks construct head-specific transformations. The exact exponential is illustrated with a schematic tangent plane and group manifold; residual transformations are identified separately.</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#526579"/></marker>
<marker id="orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#d77522"/></marker>
<linearGradient id="surface" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#ecf4fc"/><stop offset="1" stop-color="#bdd8ef"/></linearGradient>
<clipPath id="surfaceClip"><path d="M 1140 753 C 1215 641 1440 660 1600 747 C 1488 766 1420 826 1390 879 C 1270 853 1160 894 1117 844 C 1098 820 1112 777 1140 753 Z"/></clipPath>
</defs>
<rect width="1680" height="960" fill="white"/>
<g id="attention-data-flow">''')
text(35,42,'(a)  Shared bases, head-specific attention',28,weight='bold')
box(28,64,1624,447,'#fbfcfe')
box(54,236,105,90,'#eff3f8','#9dafc0')
text(106,268,'Input',22,anchor='middle');text(106,304,'X',32,anchor='middle',math=True)
path('M 159 280 H 213',arrow=True)
text(335,106,'C learned base triples',25,anchor='middle',weight='bold')
for y,label in [(140,'Wᴽ⁽¹⁾, Wᴷ⁽¹⁾, Wⱽ⁽¹⁾'),(191,'Wᴽ⁽²⁾, Wᴷ⁽²⁾, Wⱽ⁽²⁾'),(286,'Wᴽ⁽ᶜ⁾, Wᴷ⁽ᶜ⁾, Wⱽ⁽ᶜ⁾')]:
    box(215,y,240,43,'#edf4fb','#91b7d9',7); text(335,y+29,label,25,anchor='middle',math=True)
text(335,267,'⋮',31,anchor='middle')
text(335,378,'Q⁽ᶜ⁾ = XWᴽ⁽ᶜ⁾',25,anchor='middle',math=True)
text(335,412,'K⁽ᶜ⁾ = XWᴷ⁽ᶜ⁾',25,anchor='middle',math=True)
text(335,446,'V⁽ᶜ⁾ = XWⱽ⁽ᶜ⁾',25,anchor='middle',math=True)
text(335,483,'c = 1, …, C',22,anchor='middle',math=True)
path('M 455 280 H 511',arrow=True)
box(515,134,582,333,'#ffffff','#8ea9c5')
text(541,170,'Head h uses base cₕ',26,weight='bold')
text(541,200,'Fixed assignment • H/C heads per base',21,color='#596c80')
box(540,226,532,87,'#edf4fb','#c3d7e9',8)
text(806,259,'Attention weights',22,anchor='middle',weight='bold')
text(806,295,'Pₕ = softmax((Q⁽ᶜʰ⁾ Mₕ) K⁽ᶜʰ⁾ᵀ / √dₕ + mask)',25,anchor='middle',math=True)
path('M 806 313 V 340',arrow=True)
box(540,345,532,60,'#edf8f5','#a6cec4',8)
text(806,383,'Zₕ = (Pₕ V⁽ᶜʰ⁾) Rₕⱽ',29,anchor='middle',math=True)
text(806,444,'H attention maps • C base projection families',22,anchor='middle')
text(1202,161,'Repeat for',23,anchor='middle');text(1202,193,'h = 1, …, H',25,anchor='middle',math=True)
path('M 1097 280 H 1142',arrow=True)
box(1147,218,108,124,'#f4f6fa','#a9b9c9',8)
text(1201,254,'Z₁',28,anchor='middle',math=True);text(1201,286,'⋮',27,anchor='middle');text(1201,320,'Zᴴ',28,anchor='middle',math=True)
path('M 1255 280 H 1295',arrow=True)
box(1300,239,133,82,'#eff3f8','#9dafc0',8);text(1366,273,'Concat',23,anchor='middle');text(1366,304,'× Wᴼ',27,anchor='middle',math=True)
path('M 1433 280 H 1480',arrow=True)
box(1485,239,132,82,'#edf8f5','#a6cec4',8);text(1551,273,'Output',22,anchor='middle');text(1551,304,'Y',30,anchor='middle',math=True)
text(1381,396,'Y = Concat(Z₁, …, Zᴴ) Wᴼ',25,anchor='middle',math=True)
text(1381,435,'Standard multi-head output interface',21,anchor='middle',color='#596c80')
add('</g><g id="transformation-construction">')
text(35,555,'(b)  Construct the head transformations',28,weight='bold')
box(28,577,992,347,'#fbfcfe')
text(58,616,'Shared generator banks',23,weight='bold')
box(54,638,230,70,'#edf4fb','#91b7d9',8)
text(169,667,'Query–key bank',22,anchor='middle');text(169,696,'E₁, …, Eₚ',27,anchor='middle',math=True)
box(54,740,230,70,'#edf8f5','#8cbcaf',8)
text(169,769,'Value bank',22,anchor='middle');text(169,798,'F₁, …, Fₚᵥ',27,anchor='middle',math=True)
path('M 284 673 H 326',arrow=True);path('M 284 775 H 326',arrow=True)
text(519,616,'Head-specific mixtures',23,anchor='middle',weight='bold')
box(331,638,376,70,'#ffffff','#c3d7e9',8);text(519,681,'Aₕ = β Σℓ πₕ,ℓ Eℓ',29,anchor='middle',math=True)
box(331,740,376,70,'#ffffff','#a6cec4',8);text(519,783,'Bₕ = βᵥ Σℓ ρₕ,ℓ Fℓ',29,anchor='middle',math=True)
text(519,849,'πₕ = softmax(θₕ)',25,anchor='middle',math=True)
text(519,881,'ρₕ = softmax(φₕ)',25,anchor='middle',math=True)
path('M 707 673 H 750',arrow=True);path('M 707 775 H 750',arrow=True)
box(755,638,230,172,'#fff6ed','#e2b086',8)
text(870,674,'Mₕ = f(Aₕ)',28,anchor='middle',math=True)
text(870,715,'Rₕⱽ = f(Bₕ)',28,anchor='middle',math=True)
text(870,758,'Residual: f(A) = I + A',21,anchor='middle')
text(870,792,'Exact: f(A) = exp(A)',21,anchor='middle')
path('M 870 638 V 488 H 1046 V 467',color='#d77522',arrow=True,dash=True)
text(62,892,'Shared across all heads and bases',21,color='#596c80')
add('</g><g id="geometric-inset">')
text(1055,555,'(c)  Exact map: geometric view',28,weight='bold')
box(1040,577,612,347,'#ffffff')
add('<path d="M 1140 753 C 1215 641 1440 660 1600 747 C 1488 766 1420 826 1390 879 C 1270 853 1160 894 1117 844 C 1098 820 1112 777 1140 753 Z" fill="url(#surface)" stroke="#6b9ac2" stroke-width="2"/>')
add('<g clip-path="url(#surfaceClip)" opacity="0.55">')
for j in range(7):
    x=1100+j*76
    path(f'M {x} 665 Q {x+72} 758 {x-20} 910','#78a8d2',1.3)
for j in range(6):
    y=690+j*35
    path(f'M 1070 {y} Q 1340 {y-70} 1650 {y+50}','#78a8d2',1.3)
add('</g>')
add('<path d="M 1090 656 L 1463 600 L 1530 699 L 1157 755 Z" fill="#f2f7fd" fill-opacity="0.95" stroke="#7c9fc4" stroke-width="2"/>')
for j in range(1,7):
    x=1090+j*373/7; y=656-j*56/7
    path(f'M {x} {y} l 67 99','#c3d7eb',1)
for j in range(1,4):
    path(f'M {1090+j*67/4} {656+j*99/4} l 373 -56','#c3d7eb',1)
text(1090,618,'TᵢG ≅ gl(dₕ, ℝ)',24,math=True)
dot(1260,710,'#182c44',5); text(1240,733,'I',23,math=True)
path('M 1260 710 L 1238 644','#526579',2,True);text(1211,637,'E₁',24,math=True)
path('M 1260 710 L 1360 646','#526579',2,True);text(1365,644,'E₂',24,math=True)
path('M 1260 710 L 1420 703','#526579',2,True);text(1426,711,'Eₚ',24,math=True)
path('M 1260 710 L 1367 676','#d77522',3,True);text(1372,677,'Aₕ',25,color='#b65b0d',math=True)
path('M 1367 676 C 1460 698 1420 755 1366 790','#d77522',3,True)
text(1415,772,'exp',24,color='#b65b0d',math=True)
dot(1359,801,'#d77522',7);text(1380,833,'Mₕ = exp(Aₕ)',25,color='#b65b0d',math=True)
dot(1179,822);text(1164,851,'M₁',22,math=True);dot(1505,746);text(1518,747,'Mᴴ',22,math=True)
text(1060,903,'G = GL⁺(dₕ, ℝ) • schematic, exact map only',23,math=True)
add('</g></svg>')
svg='\n'.join(S)
def sub(s): return f'<tspan baseline-shift="sub" font-size="70%">{s}</tspan>'
def sup(s): return f'<tspan baseline-shift="super" font-size="70%">{s}</tspan>'
for old,new in [('Wᴽ','W'+sub('Q')),('Wᴷ','W'+sub('K')),('Wⱽ','W'+sub('V')),('Wᴼ','W'+sub('O')),('Zᴴ','Z'+sub('H')),('Mᴴ','M'+sub('H')),('TᵢG','T'+sub('I')+'G'),('⁽ᶜʰ⁾',sup('(cₕ)')),('⁽ᶜ⁾',sup('(C)')),('⁽¹⁾',sup('(1)')),('⁽²⁾',sup('(2)'))]:
    svg=svg.replace(old,new)
# Base-stream definitions use a general base c rather than the last base C.
for prefix in ['Q','K','V']:
    svg=svg.replace(prefix+sup('(C)'),prefix+sup('(c)'))
for suffix in ['Q','K','V']:
    svg=svg.replace('XW'+sub(suffix)+sup('(C)'),'XW'+sub(suffix)+sup('(c)'))
for old,new in [('ₕ,ℓ',sub('h,ℓ')),('ₚᵥ',sub('p'+sub('V'))),('Eℓ','E'+sub('ℓ')),('Fℓ','F'+sub('ℓ')),('Σℓ','Σ'+sub('ℓ')),('ₕ',sub('h')),('ₚ',sub('p')),('ᵥ',sub('V')),('θₕ','θ'+sub('h')),('φₕ','φ'+sub('h')),('ⱽ',sup('V')),('ᵀ',sup('T'))]:
    svg=svg.replace(old,new)
(ROOT/'gt_mha_architecture.svg').write_text(svg)
