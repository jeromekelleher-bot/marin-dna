"""Record wall-clock endpoints outside the official timed path for telemetry alignment."""
from pathlib import Path

p = Path("scripts/issue430/benchmark_embeddings.py")
s = p.read_text()
start = "        start = time.perf_counter()"
end = "        repeat_seconds.append(time.perf_counter() - start)"
assert s.count(start) == s.count(end) == 1
s = s.replace(start, "        wall_start = time.time()\n"+start)
s = s.replace(end, end+'''
        wall_end = time.time()
        interval = dict(strand=str(prepared.metadata.iloc[row_indices[0]]["strand"]),
                        repetition=len(repeat_seconds), start_epoch=wall_start, end_epoch=wall_end,
                        measured_seconds=repeat_seconds[-1])
        with (Path(os.environ["TIMED_SEGMENT"]) / "intervals.jsonl").open("a") as handle:
            handle.write(json.dumps(interval)+"\\n")''')
p.write_text(s)
