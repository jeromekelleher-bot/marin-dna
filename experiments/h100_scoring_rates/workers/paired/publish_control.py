import json,os,sys
from pathlib import Path
from original_full import publish
segments=["01-bf16","02-fp8","03-fp8","04-bf16"]
if len(sys.argv)>1:
    sub=sys.argv[1]
    assert sub in segments
    directory=Path(sub)
    benchmark=json.loads((directory/'benchmark.json').read_text())
    assert benchmark['n_variants']==9490 and len(benchmark['repeat_seconds'])==3
    intervals=[json.loads(line) for line in (directory/'intervals.jsonl').read_text().splitlines()]
    assert len(intervals)==6
    output=[publish(directory/name,sub+'/'+name) for name in ('benchmark.json','intervals.jsonl','scores.parquet','benchmark.log')]
    marker=directory/'complete.json'
    marker.write_text(json.dumps({'job':os.environ['IRIS_TASK_ID'],'segment':sub,'rows':9490,'artifacts':output},indent=2)+'\n')
    publish(marker,sub+'/complete.json')
else:
    artifacts=[publish(Path(name),name) for name in ('gpu-telemetry.csv','host-telemetry.jsonl','environment.txt')]
    marker=Path('complete.json')
    marker.write_text(json.dumps({'job':os.environ['IRIS_TASK_ID'],'rows':9490,'segments':segments,'artifacts':artifacts},indent=2)+'\n')
    publish(marker,marker.name)
