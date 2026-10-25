"""Author the four-page G5 report using the bundled PDF runtime."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/g5'

def build():
    OUT.mkdir(exist_ok=True,parents=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('Body',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=10,textColor=colors.HexColor('#263b4b')))
    styles.add(ParagraphStyle('Hero',fontName='Helvetica-Bold',fontSize=30,leading=35,textColor=colors.HexColor('#102d40'),spaceAfter=16))
    styles.add(ParagraphStyle('Sub',fontName='Helvetica-Bold',fontSize=15,leading=20,textColor=colors.HexColor('#137c70'),spaceBefore=12,spaceAfter=9))
    styles.add(ParagraphStyle('Small2',fontName='Helvetica',fontSize=8.5,leading=12,textColor=colors.HexColor('#526777'),spaceAfter=8))
    story=[]
    def para(t,style='Body'):story.append(Paragraph(t,styles[style]))
    def heading(t):para(t,'Sub')
    def table(rows,widths):
        t=Table([[Paragraph(str(x),styles['Small2']) for x in row] for row in rows],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3f2ef')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#b7cbd3'))]))
        story.append(t);story.append(Spacer(1,8))
    para('ROBORACE / PHYSICAL AI FINAL PROJECT','Small2')
    para('Three robots.<br/>Local AI control.<br/>Measured comparison.','Hero')
    para('A 3D integration demonstration with a preliminary held-out study. Prepared October 25, 2026.','Body')
    story.append(Image(str(ROOT/'artifacts/g4b/ppo/final.png'),width=475,height=356.25))
    heading('Task and purpose')
    para('Three robots start simultaneously and travel to individually marked goals, avoiding static obstacles and one another. The study compares a classical navigation reference, behavior cloning (BC) and parameter-shared independent PPO. Shared-space transport is the motivating application; this is simulation research, not a deployed warehouse system.')
    para('<b>Main result:</b> PPO completed 20/40 held-out 2D episodes, versus 12/40 for classical and 10/40 for BC. All three controllers completed one easy 3D integration race without contact. The 80% success target was not met.')
    story.append(PageBreak())
    para('01 / System and AI methods','Hero')
    heading('Common controller contract')
    para('Each robot receives 44 local values: goal distance/bearing, own speed/turn rate, 24 planar ranges and up to three nearby peers with relative pose/motion and presence masks. Outputs are forward speed [0, 1] m/s and angular speed [-2, 2] rad/s. The controller interval is 0.1 s. Each method shares an explicit 0.27 m arrival-stop rule, with goal tolerance 0.3 m and five stopped dwell steps. Finished robots remain in the world.')
    table([['Method','Training and inference'],['Classical','Goal steering, range avoidance and local safety/yielding heuristic. No learning.'],['BC','Bounded-action MLP with 128/128/64 hidden units. 4,057 fit samples; 1,041 internal holdout samples. Whole-episode split. Train-only normalization. 50 epochs; validation selected epoch 40.'],['Independent PPO','Shared 64/64 actor and critic; local observations. Bounded Beta policy trained from scratch with PPO/GAE. 32,768 world steps; 91,215 policy-agent transitions. Validation selected update 96. Deterministic mean-action evaluation.']],[90,385])
    heading('3D integration')
    para('PyBullet DIRECT advances 1 kg cylindrical chassis with radius 0.2 m and height 0.24 m. An ideal body-frame velocity servo runs at 0.01 s physics steps with gravity disabled. Wheels and heading markers are visual only: tire friction, wheel torque, suspension and actuator dynamics are not modeled. Contacts come from Bullet; measured poses feed the existing observation adapter. Planar sensor intersections use matching cylinder/box footprints, rather than Bullet ray casts. There is no classical obstacle-avoidance fallback in learned inference.')
    heading('Original implementation contributions')
    para('The project combines deterministic multi-robot scenarios, a common local observation/action interface, custom BC/PPO pipelines, a 3D physics adapter, contact-aware race logging, raw camera capture and paired evaluation. Three colored chassis, raised obstacles, start/goal markers, finish timing and explicit ties make the race traceable. External libraries provide numerical computing, learning, physics, image/video encoding and PDF layout; no external robot artwork was used.')
    para('Retained pilot compute: BC fitting 24.16 s plus expert collection 126.31 s; PPO training 392.12 s on CPU. These are unequal pilot budgets and do not constitute a complete historical tuning-cost audit. Training seed 0 only for each learned method.','Small2')
    story.append(PageBreak())
    para('02 / Evaluation and results','Hero')
    para('The frozen preliminary test uses 40 paired, previously held-out 2D layouts: ten each from open, slalom, bottleneck and crossing. The first ten of each family were selected from the existing 100-layout manifest before running methods. Checkpoints were selected on validation only. No additional training or test-driven retuning was performed.')
    table([['Controller','Success','Collisions','Timeouts','Capped s','Success-only s'],['Classical','12/40','0','28','45.195','10.650'],['BC','10/40','30','0','48.260','13.040'],['PPO','20/40','20','0','38.213','16.425']],[85,65,75,75,80,95])
    story.append(Image(str(ROOT/'artifacts/g4/comparison.png'),width=400,height=160))
    heading('Interpretation and uncertainty')
    para('PPO is the practical AI-demo recommendation because it has the highest team success and lowest failure-capped completion time. Failure contributes the full 60 s horizon, preventing early collisions from appearing fast. Successful makespan is conditioned on completion. All methods achieved 0/10 successes on both slalom and bottleneck. PPO completed all open and crossing episodes; BC completed only open; classical completed open plus two crossing episodes.')
    para('Paired scenario bootstrap: 5,000 resamples, seed 415. The 95% success-difference intervals are classical minus BC [0.000, 0.125], classical minus PPO [-0.325, -0.075], and BC minus PPO [-0.400, -0.125]. These intervals cover scenario sampling only. One training seed gives no training-variation estimate. CPU contention during dependency compilation limits inference-latency comparisons.')
    heading('3D transfer: a separate integration result')
    para('On predesignated scene seed 41000, all three robots finished without contact for each controller: classical 10.2 s, BC 10.4 s and PPO 15.9 s team makespan. PPO robot finish times were [15.7, 15.8, 15.9] s. The scene has two raised obstacles between separated lanes; it is an easy integration scene. Matching 2D reference makespans were 10.2/10.4/15.9 s. These results are not pooled with the held-out comparison and do not establish obstacle-avoidance reliability.')
    story.append(PageBreak())
    para('03 / Evidence and reproduction','Hero')
    heading('Submission artifacts')
    para('The package contains a captioned 120-second MP4, this report, source/configs, selected BC/PPO weights, training summaries and curves, scenario manifests, raw episode results, raw 3D videos and step logs. The film includes an uninterrupted 16-second PPO race at 1x, a synchronized three-method view with labeled terminal holds, comparison metrics and a clearly identified 2D failure still. No personal account identifier is embedded in the media.')
    heading('Run from the extracted code root')
    para('Use Windows x64 with Python 3.13. Create a virtual environment and install requirements-release.txt (CPU Torch index included). Then run:<br/><font name="Courier" size="9">python -m pytest -q<br/>python -m roborace.g4 demo<br/>python -m roborace.g4 compare</font>')
    para('The demo loads included checkpoints and writes three 3D MP4s plus logs. The compare command reproduces the fixed 40-layout study; do not tune policies against these consumed test layouts. Training commands and configurations are included. The release verification record states the actual clean-environment checks, without implying a full training rerun.')
    heading('Evidence index and attribution')
    table([['Evidence','Package path'],['Frozen comparison / raw records','artifacts/g4/frozen_manifest.json; episodes.jsonl; summary.json'],['3D model / logs / provenance','artifacts/g4b/backend.json; runtime.json; scene.json; */steps.jsonl; */episode.json'],['Training / selected checkpoints','artifacts/g2/bc_seed0/; artifacts/g3/ppo_seed0/'],['Source / dependencies / scenarios','roborace/; scripts/; configs/; requirements-*.txt; scenarios/'],['Integrity / third-party notices','SHA256SUMS.json; THIRD_PARTY_NOTICES.md; package license files']],[145,330])
    para('Libraries: NumPy, PyTorch, PyBullet, Pillow, Matplotlib, imageio/imageio-ffmpeg and ReportLab. Package notices preserve dependency license metadata; no course sample assignment code is bundled. Course-provided assignment PDFs and private planning documents are excluded from the public-ready code archive. The source was developed with AI assistance.','Small2')
    heading('Limits and release status')
    para('Full multi-seed G4 and wider G4b transfer validation remain incomplete. Minimum-clearance/path-efficiency analysis and realistic wheel dynamics are not implemented. Local G5 media and packaging do not complete course submission. Account-specific naming, public GitHub access, YouTube upload/settings, current portal confirmation and OmniCampus submission must be verified separately. No external publication or submission is claimed.')
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#b7cbd3'));c.line(48,42,547,42)
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526777'));c.drawString(48,28,'RoboRace | Preliminary single-seed study | October 25, 2026');c.drawRightString(547,28,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/'説明資料_PAI最終課題_83792.pdf'),pagesize=A4,rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=55,title='RoboRace - 3D AI robot race and preliminary comparison',author='RoboRace project',subject='Physical AI final project report')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
if __name__=='__main__':build()
