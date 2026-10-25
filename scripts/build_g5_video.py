"""Build the 120-second captioned G5 film from retained simulator recordings."""
from pathlib import Path
import json
import textwrap
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/g5'
FONT=Path('C:/Windows/Fonts/segoeui.ttf')

def font(size):return ImageFont.truetype(str(FONT),size)

def base(title,subtitle=''):
    im=Image.new('RGB',(1280,720),'#101923');d=ImageDraw.Draw(im)
    d.text((48,28),'ROBORACE / PHYSICAL AI',font=font(19),fill='#69d6b5')
    d.text((48,65),title,font=font(36),fill='white')
    if subtitle:d.text((48,120),subtitle,font=font(21),fill='#b6c9d8')
    return im

def card(title,lines,subtitle=''):
    im=base(title,subtitle);d=ImageDraw.Draw(im);y=195
    for line in lines:
        for part in textwrap.wrap(line,86):
            d.text((56,y),part,font=font(27),fill='#e4edf4');y+=40
        y+=16
    return im

def fit(im,box):
    im=im.copy();im.thumbnail((box[2],box[3]),Image.Resampling.LANCZOS)
    return im,(box[0]+(box[2]-im.width)//2,box[1]+(box[3]-im.height)//2)

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    timeline=[];cursor=0
    with imageio.get_writer(OUT/'RoboRace_submission.mp4',fps=10,codec='libx264',quality=8,macro_block_size=1) as writer:
        def section(label,seconds,frames):
            nonlocal cursor
            n=round(seconds*10);timeline.append({'start_s':cursor,'end_s':cursor+seconds,'label':label,'frames':n})
            for i in range(n):
                im=frames(i) if callable(frames) else frames.copy()
                d=ImageDraw.Draw(im)
                d.rectangle((0,678,1280,720),fill='#1a2b39')
                d.text((48,687),f'{label}   |   {cursor+i/10:05.1f}s / 120.0s',font=font(18),fill='white')
                writer.append_data(np.asarray(im))
            cursor+=seconds
        section('01 / Task and purpose',12,card('Three robots. One race. Local AI control.',[
            'Goal: reach three marked destinations without obstacle or peer contact.',
            'Compare classical navigation, behavior cloning (BC), and independent PPO.',
            'Application: a small simulation study of shared-space robot transport.'
        ],'A working 3D integration demo plus a preliminary held-out comparison'))
        section('02 / How AI drives the robots',18,card('Saved checkpoints produce speed and turn commands.',[
            '44 local inputs: goal direction/distance, own motion, 24 ranges and nearby peers.',
            'BC learns expert actions. PPO learns through trial-and-error rewards.',
            'Each robot uses the same learned network; no classical avoidance fallback.',
            'A disclosed 0.27 m arrival-stop rule is shared by all three methods.',
            '3D motion and contacts: PyBullet velocity-servo bodies, decorative wheels.'
        ]))
        raw=[Image.fromarray(a) for a in imageio.get_reader(ROOT/'artifacts/g4b/ppo/race.mp4')]
        def ppo(i):
            im=base('PPO: continuous three-robot 3D race','Saved seed-0 checkpoint | all 160 raw frames | 1x playback | no cut within this run')
            picture,pos=fit(raw[i],(140,158,1000,510));im.paste(picture,pos)
            return im
        section('03 / Actual learned-controller execution',16,ppo)
        section('04 / 3D integration outcome',10,card('All three robots finished without contact.',[
            'PPO finishes: R1 15.7 s, R2 15.8 s, R3 15.9 s. Team makespan: 15.9 s.',
            'BC: 10.4 s. Classical: 10.2 s. Each method ran its own simultaneous race.',
            'This single easy scene has raised obstacles between separated lanes.',
            'It demonstrates transfer and execution, not broad avoidance reliability.'
        ],'3D integration scene / seed 41000 / never pooled with held-out 2D metrics'))
        comparison=[Image.fromarray(a) for a in imageio.get_reader(ROOT/'artifacts/g4b/comparison.mp4')]
        def parallel(i):
            im=base('Same 3D scene, three controller methods','Synchronized 1x | independent runs | finished panels held and labeled')
            pic,pos=fit(comparison[i],(30,165,1220,460));im.paste(pic,pos);return im
        section('05 / Method comparison in 3D',16,parallel)
        chart=Image.open(ROOT/'artifacts/g4/comparison.png').convert('RGB')
        def benchmark(i):
            im=base('Held-out 2D benchmark: PPO leads this sample','40 paired layouts: 10 each open / slalom / bottleneck / crossing | one training seed')
            pic,pos=fit(chart,(65,170,1150,430));im.paste(pic,pos)
            d=ImageDraw.Draw(im);d.text((55,618),'Success: classical 12/40 | BC 10/40 | PPO 20/40. Target 80% was not met.',font=font(24),fill='#ffc478');return im
        section('06 / Measured performance',18,benchmark)
        fail=Image.open(ROOT/'artifacts/g3/demo_slalom/final.png').convert('RGB')
        def failure(i):
            im=base('Failure evidence matters.','Retained PPO slalom validation failure / 2D still / separate run / no playback claim')
            pic,pos=fit(fail,(35,165,620,490));im.paste(pic,pos)
            d=ImageDraw.Draw(im);y=205
            for line in ['All methods: 0/10 slalom successes.','All methods: 0/10 bottleneck successes.','PPO: 20 collisions in 40 test episodes.','BC: 30 collisions in 40 test episodes.','Classical: 28 timeouts, no collisions.','One successful 3D race is not reliability.']:
                d.text((680,y),line,font=font(23),fill='#e4edf4');y+=65
            return im
        section('07 / Failures and evidence limits',14,failure)
        section('08 / Contribution and reproduction',16,card('Reproducible evidence, with clear limits.',[
            'Contributions: shared observation adapter, learned checkpoints, physics contacts,',
            'three-robot race logs, paired comparison and synchronized 3D recordings.',
            'Reproduce: python -m roborace.g4 demo   |   python -m roborace.g4 compare',
            'Retained: raw clips, observations/actions, contacts, weights and SHA-256 hashes.',
            'Limits: one training seed, easy 3D layout, ideal velocity servo; no wheel friction.',
            'No final claim of broad obstacle-avoidance reliability or full multi-seed completion.'
        ]))
    assert cursor==120
    (OUT/'video_timeline.json').write_text(json.dumps({'fps':10,'duration_s':120,'audio':'none; captioned','speed':'all moving footage 1x; edits/stills/holds labeled','sections':timeline},indent=2))
    # Decode exported film end-to-end and retain frame samples for visual inspection.
    samples=OUT/'video_qa';samples.mkdir(exist_ok=True)
    wanted={0,119,120,299,300,379,459,460,559,560,639,719,720,899,900,1039,1040,1199}
    count=0
    with imageio.get_reader(OUT/'RoboRace_submission.mp4') as reader:
        meta=reader.get_meta_data()
        for count,a in enumerate(reader,1):
            if count-1 in wanted:Image.fromarray(a).save(samples/f'frame_{count-1:04d}.jpg')
    assert count==1200,(count,meta)
    (OUT/'video_checks.json').write_text(json.dumps({'decoded_frames':count,'duration_s':count/10,'dimensions':meta['size'],'fps':meta['fps'],'codec':meta['codec'],'audio':'no audio track requested','all_frames_decoded':True},indent=2))

if __name__=='__main__':build()
