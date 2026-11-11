"""Verify all completed CPU evaluation budgets and source snapshots against their Git revisions."""
import hashlib
import json
import subprocess

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json


def main():
    base=ROOT/'research/4.temporal-consistency';records=[];hashes={}
    specs=[('exploration-r21-eval-pre-zero',2944),('exploration-r21-eval-half-post',2944),
        ('exploration-r21-reference-plain-cpu',736),('exploration-r21-reference-post-seed0-cpu',736),
        ('exploration-r21-reference-post-seed1-cpu',736),('exploration-r21b',384)]
    for name,expected in specs:
        folder=base/'results'/name;cfg=json.loads((folder/'config.json').read_text());run=json.loads((folder/'runtime.json').read_text())
        assert cfg['device']=='cpu' and run['device']=='cpu' and not run['cuda_initialized'] and run['rows']==expected
        revision=cfg.get('git_revision_at_startup',cfg.get('revision'))
        sources=cfg.get('source_sha256_at_startup',cfg.get('source_sha256'));assert sources and revision
        for path,expected_sha in sources.items():
            key=(revision,path)
            if key not in hashes:hashes[key]=hashlib.sha256(subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT)).hexdigest()
            assert hashes[key]==expected_sha,(name,path,'uncommitted source at startup')
        records.append({'result_directory':name,'rows':expected,'device':'cpu','cuda_initialized':False,
            'revision':revision,'source_files_verified':len(sources),'evaluation_loop_wall_seconds':run['wall_seconds']})
    save_json(base/'process/exploration-r21/cpu-source-audit.json',{'completed_jobs':records,'source_hashes_verified':len(hashes),
        'source_mismatches':[],'main_candidate_rows':5888,'matched_reference_rows':2208,'static_followup_rows':384,
        'scope':'Runtime fields exclude some startup/import costs and are not CPU core-seconds or GPU/mobile timings.'})


if __name__=='__main__':main()
