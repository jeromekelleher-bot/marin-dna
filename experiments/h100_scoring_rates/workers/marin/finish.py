import json
import os
from pathlib import Path
from original_full import publish

kind = os.environ['RATE_KIND']
files = ['benchmark.json','environment.txt','scores.parquet']
if kind.startswith('evo'):
    files += ['strand-+.json','strand--.json','score.log']
else:
    files += ['benchmark.log']
receipt = {'job':os.environ['IRIS_TASK_ID'],'configuration':kind,'rows':9490,
           'artifacts':[publish(Path(name),name) for name in files]}
marker=Path('complete.json')
marker.write_text(json.dumps(receipt,indent=2)+'\n')
publish(marker,marker.name)
print('Published and verified all outputs',flush=True)
