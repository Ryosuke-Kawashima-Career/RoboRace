"""Generate a labeled validation-only comparison from existing raw run results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    locations = {
        'classical': 'artifacts/g1/validation_final/validation_summary.json',
        'behavior_cloning': 'artifacts/g2/validation/validation_summary.json',
        'ppo_initial_untrained': 'artifacts/g3/initial_validation/validation_summary.json',
        'ppo_selected_trained': 'artifacts/g3/validation/validation_summary.json'
    }
    methods = {}
    for name, relative in locations.items():
        data = json.loads((ROOT / relative).read_text())
        results = data['results']
        methods[name] = {'episodes': len(results), 'successes': sum(r['team_success'] for r in results),
                         'collision_episodes': sum(r['collision'] is not None for r in results),
                         'timeouts': sum(r['timeout'] for r in results),
                         'mean_capped_s': sum(r['capped_completion_s'] for r in results)/len(results),
                         'courses': {c: {'successes': sum(r['team_success'] for r in results if r['course']==c),
                                         'episodes': sum(r['course']==c for r in results)}
                                     for c in ('open', 'slalom', 'bottleneck', 'crossing')}}
    summary = {'scope': 'same eight validation scenarios; includes checkpoint-selection layouts; not held-out G4 evidence', 'methods': methods}
    (ROOT / 'artifacts/g3/validation_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
