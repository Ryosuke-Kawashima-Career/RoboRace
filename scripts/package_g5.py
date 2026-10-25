"""Stage an allowlisted source/evidence release; never copy environments or private docs."""
from pathlib import Path
import shutil,json,hashlib,zipfile
import importlib.metadata as md
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/g5'
STAGE=OUT/'release/RoboRace'

def stage():
    STAGE.mkdir(parents=True,exist_ok=True)
    for name in ['roborace','scripts','tests','configs','scenarios','artifacts/g1/source_snapshots','artifacts/g2','artifacts/g3','artifacts/g4','artifacts/g4b']:
        shutil.copytree(ROOT/name,STAGE/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    for p in list(ROOT.glob('requirements*.txt'))+[ROOT/'README.md',ROOT/'pyproject.toml',ROOT/'.gitignore']:
        shutil.copy2(p,STAGE/p.name)
    sub=STAGE/'submission';sub.mkdir(exist_ok=True)
    for p in list(OUT.glob('*.pdf'))+[OUT/'RoboRace_submission.mp4',OUT/'video_timeline.json',OUT/'video_checks.json']:
        shutil.copy2(p,sub/p.name)
    if (OUT/'clean_environment.json').exists():shutil.copy2(OUT/'clean_environment.json',sub/'clean_environment.json')
    text=(ROOT/'dev/plans/40_g4_report.md').read_text(encoding='utf-8')+'\n\n'+(ROOT/'dev/plans/50_g5_report.md').read_text(encoding='utf-8')
    (sub/'IMPLEMENTATION.md').write_text(text,encoding='utf-8')
    readme=(STAGE/'README.md').read_text(encoding='utf-8')
    for p in ['dev/plans/README.md','dev/plans/40_g4_report.md','dev/plans/50_g5_report.md']:readme=readme.replace(p,'submission/IMPLEMENTATION.md')
    (STAGE/'README.md').write_text(readme,encoding='utf-8')
    (sub/'UPLOAD_METADATA.md').write_text('''# Prepared upload metadata (not uploaded)

YouTube title: PAI最終課題_83792
PDF filename: 説明資料_PAI最終課題_83792.pdf

Proposed description: Three robots race in a 3D PyBullet world under a saved PPO checkpoint. This captioned 120-second film includes continuous 1x footage, a three-method comparison, preliminary 40-layout held-out results and limitations. BC/PPO each have one training seed. The 3D integration scene is easy and uses ideal velocity-servo bodies with decorative wheels. PPO success is 20/40, below the 80% target. Raw evidence and reproduction code accompany the package. All moving footage is 1x; cuts, stills and finished-panel holds are labeled.

External steps still unperformed:
- Publish the repository and verify signed-out source/checkpoint access.
- Upload the final MP4; choose at least unlisted visibility.
- Set not made for kids, no age restriction and instant premiere disabled.
- Verify signed-out playback; keep video available through grading.
- Check current course portal instructions and deadline.
- Submit PDF, public repository URL and YouTube URL to OmniCampus; retain private confirmation.

Public repository URL: pending publication.
YouTube URL: pending upload.
OmniCampus confirmation: pending submission.
Account name is deliberately omitted; permitted account ID 83792 is used.
''',encoding='utf-8')
    licenses=STAGE/'third_party_licenses';licenses.mkdir(exist_ok=True)
    shutil.copytree(OUT/'authoring_licenses',licenses/'authoring',dirs_exist_ok=True)
    notices=['# Third-party dependencies','', 'PDF authoring: ReportLab 4.4.9 and pypdf 6.10.0 (bundled authoring runtime); licenses under third_party_licenses/authoring. Simulation runtime and authoring dependencies retain their own licenses. Generated robot geometry and scene artwork use no external assets. This project was developed with AI assistance. No project-wide redistribution license has been selected.','']
    for dist in sorted(md.distributions(),key=lambda d:d.metadata.get('Name','')):
        name=dist.metadata.get('Name','unknown');version=dist.version
        notices.append(f"- {name} {version}: {dist.metadata.get('License-Expression') or dist.metadata.get('License') or 'see package license files'}")
        for file in dist.files or []:
            if any(word in str(file).lower() for word in ['license','copying','notice']) and ('.dist-info/' in str(file)):
                source=Path(dist.locate_file(file))
                if source.is_file():
                    target=licenses/name/str(file);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    (STAGE/'THIRD_PARTY_NOTICES.md').write_text('\n'.join(notices),encoding='utf-8')
    return STAGE

def archive():
    files=sorted(p for p in STAGE.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='SHA256SUMS.json')
    manifest={p.relative_to(STAGE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (STAGE/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    path=OUT/'RoboRace_submission_package.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[STAGE/'SHA256SUMS.json']:z.write(p,'RoboRace/'+p.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    info={'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':len(files)+1,'bytes':path.stat().st_size,'excluded':['virtual environments','private docs/dev directories','course assignment PDF','Git history'],'account_id':'83792'}
    (OUT/'package_manifest.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
if __name__=='__main__':stage();archive()
