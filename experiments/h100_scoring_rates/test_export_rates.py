import unittest

from export_rates import ROWS, TEST_SHA, calculate, extract_shard, paired_seconds


class ShardedTimingTest(unittest.TestCase):
    def test_rounding_boundary_and_gpu_normalization(self):
        log = "\r".join(
            f"strand={strand}:  {pct}%| | {done}/2372 [{clock}<00:00, 2.9s/batch]"
            for strand in "+-" for pct, done, clock in [(0, 1, "00:03"), (100, 2372, "1:52:51")]
        )
        shard = extract_shard(log)
        self.assertEqual(shard["strand_elapsed_seconds"]["+"], {"first": 3, "last": 6771})
        parts = [dict(shard, rows=n) for n in (2372, 2373, 2372, 2373)]
        timed, seconds, gpus, workers, repeats = calculate("evo40b", {"partitions": parts}, 8192, 1)
        self.assertEqual((timed, seconds, gpus, workers, repeats), (ROWS - 4, 6768 * 8, 2, 4, 1))

    def test_incomplete_strand(self):
        with self.assertRaises(ValueError):
            extract_shard("strand=+: 0%| | 1/2372 [00:03<2:00:00, 3.0s/batch]")


class PairedTimingTest(unittest.TestCase):
    def test_pools_six_passes_before_taking_median(self):
        def benchmark(passes):
            return dict(n_variants=ROWS, n_strand_rows=2 * ROWS, context_tokens=256,
                        batch_size=128, gpu_count=1, variant_dataset_sha256=TEST_SHA,
                        quantization="none", torch_compile=True, repeat_seconds=passes,
                        forward_seconds=[x / 2 for x in passes], reverse_seconds=[x / 2 for x in passes],
                        variants_per_h100_hour=3600 * ROWS / passes[1])
        records = {"01-bf16": benchmark([1, 2, 100]), "04-bf16": benchmark([8, 9, 10])}
        self.assertEqual(paired_seconds("bf16", records), (8.5, 6))


if __name__ == "__main__":
    unittest.main()
