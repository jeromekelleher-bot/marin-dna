import unittest

from export_rates import ROWS, calculate, extract_shard


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


if __name__ == "__main__":
    unittest.main()
