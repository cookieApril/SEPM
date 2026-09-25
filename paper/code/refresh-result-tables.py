"""Refresh marked tables in main.tex; never rewrite manuscript prose.

Run with Python 3.10+ from any directory. No network or benchmark execution.
"""
import collections
import csv
import json
import re
from result_data import *

def additional_tables():
    body = line(['Benchmark / host', 'Memory', 'Mean (\\%)', '95\\% interval'])+'\\midrule\n'
    for r in e1['summary_rows']:
        names={'alfworld':'ALFWorld','webarena':'WebArena','officebench':'OfficeBench',
               'autogen':'AutoGen','dylan':'DyLAN','gmemory':'G-Memory','team-memory':'\\method{}','no-memory':'No added memory'}
        lo,hi=r['score_ci95']
        body+=line([names[r['benchmark']]+' / '+names[r['mas']],names[r['memory_method']],
                    f"{r['official_task_score_mean']*100:.1f}",f'[{lo*100:.1f}, {hi*100:.1f}]'])
    TABLES['E1CI']=table('E1 official-score means and supplied case-bootstrap intervals.',
                        'tab:e1_intervals','llrr',body,'Each interval concerns a case mean, not a paired treatment effect or variation across repeated executions.')
    body=line(['Benchmark','Cases','Improved','Tied','Regressed'])+'\\midrule\n'
    for name in ['ALFWorld','WebArena','MultiAgentBench Research','OfficeBench']:
        b=e3['benchmarks'][name]; counts=b['full_vs_no_extra']
        body+=line([name.replace('MultiAgentBench Research','Research'),str(b['case_count']),str(counts['improved']),str(counts['tie']),str(counts['regressed'])])
    TABLES['E3PAIRS']=table('Case-level outcomes of full versus no added components in E3.',
                           'tab:e3_pairs','lrrrr',body,'Counts refer to the sign of the official-score difference on matched cases.')
    body=line(['Benchmark','Condition','Events/case','Extra tokens','Latency'])+'\\midrule\n'
    condition_names={'no-extra-components':'No added','sop-only':'Memory only','divergence-only':'Alignment only','full':'Full'}
    for bi,name in enumerate(['ALFWorld','WebArena','MultiAgentBench Research','OfficeBench']):
        if bi: body+='\\midrule\n'
        for condition in condition_names:
            r=r3[condition,name]
            cost='n/a' if r['token_overhead'] is None else f"{r['token_overhead']:+,.1f}"
            latency='n/a' if r['latency'] is None else f"{r['latency']:.1f}"
            body+=line([name.replace('MultiAgentBench Research','Research'),condition_names[condition],f"{r['divergence_recovery']:.3f}",cost,latency])
    TABLES['E3DIAG']=table('E3 activity and cost diagnostics using their actual source definitions.',
                          'tab:e3_diagnostics','llrrr',body,'Events are mean divergence-event counts. Extra tokens are relative to no added components. Latency follows the source field. n/a marks a metric not recorded for that benchmark adapter. Procedural-memory retrieval and unsafe-acceptance fields are zero throughout.')

additional_tables()
TABLES['E1']=TABLES.pop('tables/e1_main.tex')
TABLES['E2']=TABLES.pop('tables/e2_models.tex')
TABLES['E3']=TABLES.pop('tables/e3_contributions.tex')

def marked(key):
    return f'% BEGIN AUTO TABLE {key}\n'+TABLES[key]+f'% END AUTO TABLE {key}'

def export_data():
    out=PAPER/'code/data'; out.mkdir(exist_ok=True)
    e1_sums=collections.defaultdict(list)
    for r in e1['per_cell_result_paths']:
        raw=json.loads((ROOT/r['result_path']).read_text(encoding='utf-8'))
        value=raw['metrics']['primary_score']
        assert abs(value-r['official_score'])<1e-10,r['result_path']
        e1_sums[r['benchmark'],r['mas'],r['memory_method']].append(value)
    assert sum(map(len,e1_sums.values()))==168
    for r in e1['summary_rows']:
        assert abs(statistics.mean(e1_sums[r['benchmark'],r['mas'],r['memory_method']])-r['official_task_score_mean'])<1e-10
    with (out/'E2-procedural-memory-model-scores.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['procedural_memory_model']+[k for k,_,_,_ in benchmarks]+['cases'])
        for m in models: writer.writerow([m]+[by_model[m][k] for k,_,_,_ in benchmarks]+[72])
    index=json.loads((SOURCE/'e2_available_sop_models_quality.json').read_text(encoding='utf-8'))['per_cell_result_paths']
    audit=collections.defaultdict(collections.Counter)
    sums=collections.defaultdict(list); ids=collections.defaultdict(set)
    map_key={'multiagentbench':'multiagentbench_research'}
    for r in index:
        b=r['benchmark']; m=r['sop_model']; f=ROOT/r['result_path']
        d=json.loads(f.read_text(encoding='utf-8')); metrics=d['metrics']
        assert abs(metrics['primary_score']-r['official_score'])<1e-10, f
        assert metrics['case_count']==1, f
        audit[b]['accepted_records']+=1
        sums[m,map_key.get(b,b)].append(metrics['primary_score']); ids[m,b].add(r['case_id'])
        for key in ['sop_model_calls','sop_retrieval_count','sop_reuse_success','sop_parse_success']:
            if key in metrics:
                audit[b][key+'_present']+=1; audit[b][key+'_sum']+=metrics[key] or 0
    assert sum(a['accepted_records'] for a in audit.values())==864
    for (m,b),values in sums.items(): assert abs(statistics.mean(values)-by_model[m][b])<1e-10,(m,b)
    for b in audit:
        case_sets=[ids[m,b] for m in models]
        assert all(s==case_sets[0] for s in case_sets), b
    (out/'result-audit.json').write_text(json.dumps({'counts':{'E1':168,'E2':864,'E3':240},
        'e2_statistics':stats,'e3_contrasts':contrasts,'accepted_e2_metric_audit':dict(audit),
        'e3_metric_definitions':e3['metric_notes'],'e1_means_recomputed_from_168_raw_records':True,
        'e2_means_recomputed_from_864_raw_records':True},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=PAPER/'main.tex'; raw=p.read_bytes(); marker=b'\\section{Experiments}'
    prefix,tail=raw.split(marker,1); text=marker.decode()+tail.decode('utf-8').replace('\r\n', '\n')
    for key in TABLES:
        pattern=rf'% BEGIN AUTO TABLE {key}\n.*?% END AUTO TABLE {key}'
        text,count=re.subn(pattern,lambda _:marked(key),text,flags=re.S)
        if count!=1: raise ValueError(f'Expected exactly one marked table {key}; found {count}')
    export_data()
    p.write_bytes(prefix+text.encode('utf-8'))
    print('Refreshed six marked tables; verified all 864 E2 scores and matched case sets.')
