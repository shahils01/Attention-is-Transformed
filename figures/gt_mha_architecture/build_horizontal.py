"""Compact conceptual architecture, with editable vector geometry."""
from pathlib import Path
from html import escape
import math
P=Path(__file__).resolve().parent
s=[]
def a(v):s.append(v)
def t(x,y,v,n=23,c='#24374b',weight='normal',anchor='middle'):
 a(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{n}" font-weight="{weight}" text-anchor="{anchor}" fill="{c}">{escape(v)}</text>')
def path(d,c='#52677d',w=2,arrow=False,dash=False,opacity=1):
 a(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"'+(' marker-end="url(#arr)"' if arrow else '')+(' stroke-dasharray="5 5"' if dash else '')+'/>')
def rect(x,y,w,h,c,stroke='none',r=3):a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{c}" stroke="{stroke}"/>')
colors=['#367bae','#238b80','#8264ae']
pales=['#e9f2f9','#e7f4f1','#f0ebf7']
def matrix(x,y,w,h,c,rows=5,cols=5,kind=0):
 rect(x,y,w,h,'white',c,1)
 cw=w/cols;ch=h/rows
 for i in range(rows):
  for j in range(cols):
   if kind==0: alpha=.14+.7*(i==j)+.1*((i+2*j)%3==0)
   elif kind==1: alpha=.12+.65*(j==0)+.24*(j==i)
   else:alpha=.14+.7*((i+j)%4==0)+.12*(i==j)
   a(f'<rect x="{x+j*cw+1}" y="{y+i*ch+1}" width="{cw-2}" height="{ch-2}" fill="{c}" opacity="{min(alpha,.92)}"/>')
def dot(x,y,c,r=5):a(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" stroke="white" stroke-width="2"/>')

a('''<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="450" viewBox="0 0 1560 450" role="img" aria-labelledby="title desc">
<title id="title">GT-MHA: shared projections with learned head transformations</title>
<desc id="desc">One horizontal architecture illustration. Token embeddings feed C base projection families. Shared generators produce head-specific transformations, illustrated with an integrated tangent plane and manifold. Base features and transformations meet at each of H attention heads, whose outputs are concatenated and projected.</desc>
<defs>
<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#52677d"/></marker>
<linearGradient id="surf" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#f4f8fc"/><stop offset="1" stop-color="#d3e5f4"/></linearGradient>
<clipPath id="clip"><path d="M 569 245 C 624 186 808 180 952 249 C 841 254 793 301 761 350 C 684 323 565 355 551 311 C 544 292 552 263 569 245 Z"/></clipPath>
</defs><rect width="1560" height="450" fill="white"/>''')
# Feature routes remain continuous under the geometric inset.
a('<g id="base-feature-routing">')
for k,c in enumerate(colors):
 y=157+k*84
 route_y=364+k*12
 path(f'M 338 {y} H 383 Q 402 {y} 402 {route_y} H {973+k*18}',c,2.7,opacity=.65)
 for h in range(2):
  yy=128+k*94+h*39
  xx=1040+h*99
  path(f'M {973+k*18} {route_y} V {yy} H {xx-8}',c,2,opacity=.65)
t(690,414,'C shared Q/K/V streams',22,c='#64788b')
a('</g>')
a('<g id="input-and-bases">')
t(98,42,'Input',25,weight='bold');t(290,42,'Shared bases',25,weight='bold')
for j in range(5):
 rect(56+j*4,137+j*31,80,24,'#e9eef4','#b3c1ce',4)
 for k in range(5):rect(63+j*4+k*13,143+j*31,8,12,'#849ab1',r=1)
t(97,331,'X',27)
path('M 151 227 H 190',arrow=True)
path('M 194 156 V 325','#b2c2d0',2)
for k,c in enumerate(colors):
 y=134+k*84
 path(f'M 194 {y+23} H 235',c,2)
 for j in range(3):
  matrix(242+j*24,y+j*7,48,44,c,4,4,2)
 t(288,y+70,['Base 1','Base 2','Base C'][k],20,c)
t(291,401,'Learn C projection triples',21,c='#64788b')
a('</g>')
a('<g id="integrated-transformation-geometry">')
t(711,42,'Learned head transformations',25,weight='bold')
# Generator bank tiles and head coordinate notation.
for j in range(3):matrix(448+j*20,78+j*7,39,35,'#b1743e',3,3,2)
t(482,152,'Shared generators',20,c='#855329')
path('M 538 111 C 553 111 563 123 583 127','#b1743e',2,True)
t(727,81,'Head-specific mixtures',22,c='#855329')
# Manifold.
a('<path d="M 569 245 C 624 186 808 180 952 249 C 841 254 793 301 761 350 C 684 323 565 355 551 311 C 544 292 552 263 569 245 Z" fill="url(#surf)" stroke="#8cabc7" stroke-width="1.8"/>')
a('<g clip-path="url(#clip)">')
for j in range(7):path(f'M {550+j*61} 190 Q {610+j*44} 268 {540+j*51} 365','#acc8df',1.1)
for j in range(5):path(f'M 535 {226+j*29} Q 761 {178+j*35} 971 {263+j*27}','#acc8df',1.1)
a('</g>')
# Tangent plane, generator arrows, and exponential map.
a('<path d="M 543 128 L 798 95 L 865 178 L 610 212 Z" fill="#f4f8fc" fill-opacity=".96" stroke="#86a6c5" stroke-width="1.7"/>')
for j in range(1,6):path(f'M {543+j*42.5} {128-j*5.5} l 67 84','#c9dbea',1)
for j in range(1,4):path(f'M {543+j*16.75} {128+j*21} l 255 -33','#c9dbea',1)
dot(657,178,'#263c53',4)
path('M 657 178 L 643 116','#52677d',2,True)
path('M 657 178 L 746 112','#52677d',2,True)
path('M 657 178 L 794 172','#52677d',2,True)
t(628,116,'E₁',21);t(754,111,'E₂',21);t(811,177,'Eₚ',21)
path('M 657 178 L 737 145','#d97d26',3)
dot(737,145,'#d97d26',4)
t(730,131,'Aₕ',22,c='#af601e')
path('M 737 145 C 830 170 820 232 772 266','#d97d26',3,True)
t(820,221,'exp',22,c='#af601e')
t(644,202,'I',19)
# Six transformations become the six head control inputs.
points=[(610,287),(652,270),(701,292),(772,275),(823,268),(868,250)]
for i,(x,y) in enumerate(points):
 c=colors[i//2]; yy=128+(i//2)*94+(i%2)*39;xx=1040+(i%2)*99
 path(f'M {x} {y} C {x+58} {y-20} {xx-74} {yy-29} {xx} {yy-15}',c,1.7,dash=True,opacity=.8)
 dot(x,y,c,5)
t(663,326,'Transformation space',22,c='#416384')
t(737,440,'Geometry illustrates exp; residual uses I + A.',19,c='#64788b')
t(732,357,'Separate QK and value banks',21,c='#64788b')
a('</g>')
a('<g id="attention-heads-and-output">')
t(1110,42,'H attention heads',25,weight='bold')
for k,c in enumerate(colors):
 for h in range(2):
  y=100+k*94+h*39;x=1040+h*99
  # Transformation-feature junction immediately before each attention head.
  dot(x,y+28,c,3.5)
  matrix(x+9,y,53,53,c,5,5,(k+h)%3)
  rect(x+69,y+7,11,39,pales[k],c,1)
  end_y=136+(k*2+h)*30
  path(f'M {x+80} {y+26} H 1228 C 1240 {y+26} 1242 {end_y} 1254 {end_y}',c,1.8)
  # Labels identify illustrative heads rather than numerical matrix values.
  t(x+35,y-8,['h₁','h₂','h₃','h₄','h₅','h₆'][k*2+h],18,c)
t(1120,399,'Attention + value aggregation',20,c='#64788b')
# Concat as one multicolored output stack.
for i in range(6):rect(1254,123+i*30,18,27,colors[i//2],r=1)
t(1263,340,'Concat',19)
path('M 1283 211 H 1320',arrow=True)
rect(1326,149,47,126,'#e9eef4','#9caebe',3)
t(1349,215,'Wᴼ',22)
path('M 1382 211 H 1420',arrow=True)
for j in range(5):
 rect(1434,137+j*30,76,24,'#edf4f5','#95b6bd',4)
 for k in range(5):rect(1441+k*13,143+j*30,8,12,'#7095a2',r=1)
t(1469,42,'Output',25,weight='bold');t(1472,331,'Y',27)
a('</g>')
t(35,440,'Solid: features   ·   Dashed: transformations',20,c='#64788b',anchor='start')
a('</svg>')
svg='\n'.join(s)
def sub(v):return f'<tspan baseline-shift="sub" font-size="72%">{v}</tspan>'
for old,new in [('ₕ',sub('h')),('ₚ',sub('p')),('Wᴼ','W'+sub('O'))]:svg=svg.replace(old,new)
(P/'gt_mha_horizontal.svg').write_text(svg)
