"""Verify package hashes and reconcile displayed metrics against raw evidence."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def verify(root=ROOT):
    checks={}
    path=root/'SHA256SUMS.json'
    if path.exists():
        expected=json.loads(path.read_text(encoding='utf-8'))
        bad=[k for k,v in expected.items() if not (root/k).is_file() or hashlib.sha256((root/k).read_bytes()).hexdigest()!=v]
        assert not bad,f'Integrity failures: {bad}'
        checks['hashed_files_verified']=len(expected)
    records=[json.loads(s) for s in (root/'artifacts/g4/episodes.jsonl').read_text().splitlines()]
    summary=json.loads((root/'artifacts/g4/summary.json').read_text())['methods']
    methods={'classical':'classical','bc':'behavior_cloning_with_arrival_stop','ppo':'independent_ppo_with_arrival_stop'}
    for label,method in methods.items():
        rows=[r for r in records if r['method']==method];s=summary[label]
        assert len(rows)==s['episodes']==40
        assert sum(r['team_success'] for r in rows)==s['successes']
        assert sum(r['collision'] is not None for r in rows)==s['collisions']
        assert sum(r['timeout'] for r in rows)==s['timeouts']
        assert abs(sum(r['capped_completion_s'] for r in rows)/40-s['capped_completion_s'])<1e-8
    checks['paired_heldout_episodes_reconciled']=120
    for k,ckpt in [('bc','artifacts/g2/bc_seed0/policy.pt'),('ppo','artifacts/g3/ppo_seed0/policy.pt')]:
        r=json.loads((root/f'artifacts/g4b/{k}/episode.json').read_text())
        assert hashlib.sha256((root/ckpt).read_bytes()).hexdigest()==r['checkpoint_sha256']
        assert r['team_success'] and r['collision'] is None
    checks['learned_3d_checkpoint_hashes_verified']=2
    print(json.dumps(checks,indent=2));return checks
if __name__=='__main__':verify()
