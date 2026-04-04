# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline KV cache eviction policy simulator.

Simulates vLLM's prefix cache behavior without GPU or model weights.
Directly uses vLLM's BlockPool and FreeKVCacheBlockQueue to ensure
the simulation matches real engine behavior exactly.

vLLM cache parameters (--block-size, --num-gpu-blocks-override, etc.)
are the same as ``vllm serve``. Simulation parameters use ``--sim-``.

Examples:
    python benchmarks/benchmark_cache_policy.py \\
        --num-gpu-blocks-override 500

    python benchmarks/benchmark_cache_policy.py \\
        --num-gpu-blocks-override 500 \\
        --sim-num-requests 10000 \\
        --sim-num-prefixes 200 \\
        --sim-max-concurrent 14
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class SimRequest:
    """A simulated request with prefix block hashes and decode length."""
    prefix_hashes: list[BlockHash]
    decode_blocks: int


# ---------------------------------------------------------------------------
# Workload generator — simple Zipf-like distribution
# ---------------------------------------------------------------------------

class WorkloadGenerator:
    """
    Generates requests by sampling from a pool of prefixes with Zipf-like
    distribution. No hardcoded scenarios — the caller controls how the
    workload changes over time by adjusting parameters between runs.

    Args:
        num_prefixes: Total unique prefixes in the pool.
        prefix_blocks: Number of cache blocks per prefix.
        decode_blocks: Number of decode blocks per request.
        zipf_alpha: Zipf exponent. Higher = more skewed (fewer hot prefixes).
            1.0 is standard Zipf. 0.0 is uniform.
    """

    def __init__(
        self,
        num_prefixes: int,
        prefix_blocks: int,
        decode_blocks: int,
        zipf_alpha: float = 1.0,
    ):
        self.num_prefixes = num_prefixes
        self.prefix_blocks = prefix_blocks
        self.decode_blocks = decode_blocks

        # Precompute Zipf weights
        self.prefixes = [f"p{i}" for i in range(num_prefixes)]
        self.weights = [1.0 / (i + 1) ** zipf_alpha
                        for i in range(num_prefixes)]

    def generate(self) -> SimRequest:
        prefix_name = random.choices(
            self.prefixes, weights=self.weights, k=1,
        )[0]
        hashes = [BlockHash(f"{prefix_name}_b{i}".encode())
                  for i in range(self.prefix_blocks)]
        return SimRequest(prefix_hashes=hashes,
                          decode_blocks=self.decode_blocks)


# ---------------------------------------------------------------------------
# BlockPool driver
# ---------------------------------------------------------------------------

KV_CACHE_GROUP_ID = 0
KV_CACHE_GROUP_IDS = [KV_CACHE_GROUP_ID]


def process_request(
    pool: BlockPool, req: SimRequest,
) -> tuple[int, int, list[KVCacheBlock]]:
    """Process a request through the real vLLM BlockPool."""
    hits = 0
    misses = 0
    allocated: list[KVCacheBlock] = []

    for block_hash in req.prefix_hashes:
        cached = pool.get_cached_block(block_hash, KV_CACHE_GROUP_IDS)
        if cached is not None:
            pool.touch(cached)
            allocated.extend(cached)
            hits += 1
        else:
            blocks = pool.get_new_blocks(1)
            block = blocks[0]
            bhg = make_block_hash_with_group_id(
                block_hash, KV_CACHE_GROUP_ID,
            )
            block.block_hash = bhg
            pool.cached_block_hash_to_block.insert(bhg, block)
            allocated.append(block)
            misses += 1

    if req.decode_blocks > 0:
        allocated.extend(pool.get_new_blocks(req.decode_blocks))

    return hits, misses, allocated


def release_request(pool: BlockPool, blocks: list[KVCacheBlock]) -> None:
    """Release all blocks held by a request."""
    pool.free_blocks(reversed(blocks))


# ---------------------------------------------------------------------------
# Metrics — mirrors vLLM's PrefixCacheStats sliding window
# ---------------------------------------------------------------------------

class CacheMetrics:
    """Sliding window hit rate, same algorithm as vLLM PrefixCacheStats."""

    def __init__(self, max_recent: int = 1000):
        self._max_recent = max_recent
        self._recent: deque[tuple[int, int]] = deque()
        self._agg_hits = 0
        self._agg_total = 0

    def record(self, hits: int, total: int) -> None:
        self._recent.append((hits, total))
        self._agg_hits += hits
        self._agg_total += total
        while len(self._recent) > self._max_recent:
            old_h, old_t = self._recent.popleft()
            self._agg_hits -= old_h
            self._agg_total -= old_t

    @property
    def hit_rate(self) -> float:
        return (self._agg_hits / self._agg_total
                if self._agg_total > 0 else 0.0)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Drives BlockPool with a stream of requests and collects metrics."""

    def __init__(
        self,
        pool: BlockPool,
        gen: WorkloadGenerator,
        num_requests: int,
        max_concurrent: int,
        log_interval: int = 100,
    ):
        self.pool = pool
        self.gen = gen
        self.num_requests = num_requests
        self.max_concurrent = max_concurrent
        self.log_interval = log_interval

        self.metrics = CacheMetrics()
        self._active: deque[list[KVCacheBlock]] = deque()
        self.log_entries: list[dict] = []

    def run(self) -> list[dict]:
        for i in range(self.num_requests):
            self._step()

            if (i + 1) % self.log_interval == 0:
                self.log_entries.append({
                    "step": i + 1,
                    "running": len(self._active),
                    "kv_cache_usage": self.pool.get_usage(),
                    "prefix_cache_hit_rate": self.metrics.hit_rate,
                    "cached_blocks": len(
                        self.pool.cached_block_hash_to_block),
                    "free_blocks": self.pool.get_num_free_blocks(),
                })

        # Drain
        while self._active:
            release_request(self.pool, self._active.popleft())

        return self.log_entries

    def _step(self) -> None:
        pool = self.pool
        gen = self.gen

        # Retire oldest requests to stay within concurrency limit
        while len(self._active) >= self.max_concurrent:
            release_request(pool, self._active.popleft())

        # If still not enough free blocks, retire more
        blocks_needed = gen.prefix_blocks + gen.decode_blocks
        while pool.get_num_free_blocks() < blocks_needed and self._active:
            release_request(pool, self._active.popleft())

        req = gen.generate()
        hits, misses, blocks = process_request(pool, req)
        self._active.append(blocks)
        self.metrics.record(hits, hits + misses)


# ---------------------------------------------------------------------------
# Output — vLLM-style log format
# ---------------------------------------------------------------------------

def print_log(entries: list[dict]) -> None:
    """Print metrics in a format similar to vLLM engine logs."""
    for e in entries:
        print(
            f"Step {e['step']:>6}  "
            f"Running: {e['running']:>3} reqs, "
            f"GPU KV cache usage: {e['kv_cache_usage'] * 100:>5.1f}%, "
            f"Prefix cache hit rate: "
            f"{e['prefix_cache_hit_rate'] * 100:>5.1f}%"
        )


def print_summary(entries: list[dict]) -> None:
    """Print overall summary."""
    if not entries:
        return
    avg_hit = (sum(e["prefix_cache_hit_rate"] for e in entries)
               / len(entries))
    min_hit = min(e["prefix_cache_hit_rate"] for e in entries)
    max_hit = max(e["prefix_cache_hit_rate"] for e in entries)
    print(f"\nSummary ({len(entries)} data points):")
    print(f"  Prefix cache hit rate: "
          f"avg {avg_hit * 100:.1f}%, "
          f"min {min_hit * 100:.1f}%, "
          f"max {max_hit * 100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline KV cache eviction policy simulator. "
        "Uses the same cache parameters as vllm serve. "
        "Simulation parameters use the --sim- prefix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- vLLM cache parameters (same names as vllm serve) ---
    vllm_group = parser.add_argument_group("vLLM cache parameters")
    vllm_group.add_argument(
        "--num-gpu-blocks-override", type=int, default=500,
        help="Number of GPU KV cache blocks. "
        "Get this from vllm serve startup log: '# GPU blocks: xxx'.")
    vllm_group.add_argument(
        "--block-size", type=int, default=16,
        help="Tokens per cache block.")
    vllm_group.add_argument(
        "--enable-prefix-caching", action="store_true", default=True,
        help="Enable prefix caching.")
    vllm_group.add_argument(
        "--no-enable-prefix-caching", dest="enable_prefix_caching",
        action="store_false",
        help="Disable prefix caching.")

    # --- Simulation workload parameters ---
    sim_group = parser.add_argument_group("Simulation parameters")
    sim_group.add_argument(
        "--sim-num-requests", type=int, default=10000,
        help="Total number of requests to simulate.")
    sim_group.add_argument(
        "--sim-num-prefixes", type=int, default=200,
        help="Number of unique prefixes in the workload.")
    sim_group.add_argument(
        "--sim-prefix-blocks", type=int, default=8,
        help="Cache blocks per prefix.")
    sim_group.add_argument(
        "--sim-decode-blocks", type=int, default=28,
        help="Decode blocks per request.")
    sim_group.add_argument(
        "--sim-max-concurrent", type=int, default=14,
        help="Max concurrent requests.")
    sim_group.add_argument(
        "--sim-zipf-alpha", type=float, default=1.0,
        help="Zipf distribution exponent. "
        "Higher = more skewed. 0 = uniform.")
    sim_group.add_argument(
        "--sim-log-interval", type=int, default=100,
        help="Print metrics every N requests.")
    sim_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.")

    return parser


def main():
    args = make_parser().parse_args()
    random.seed(args.seed)

    num_blocks = args.num_gpu_blocks_override
    block_size = args.block_size
    pb = args.sim_prefix_blocks
    db = args.sim_decode_blocks

    pool = BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=args.enable_prefix_caching,
        hash_block_size=block_size,
    )

    gen = WorkloadGenerator(
        num_prefixes=args.sim_num_prefixes,
        prefix_blocks=pb,
        decode_blocks=db,
        zipf_alpha=args.sim_zipf_alpha,
    )

    print(
        f"Config: {num_blocks} blocks, "
        f"block_size={block_size}, "
        f"prefix_caching={args.enable_prefix_caching}"
    )
    print(
        f"Workload: {args.sim_num_requests} requests, "
        f"{args.sim_num_prefixes} prefixes (zipf α={args.sim_zipf_alpha}), "
        f"{pb}+{db}={pb + db} blocks/req, "
        f"max_concurrent={args.sim_max_concurrent}"
    )
    print()

    sim = Simulator(
        pool=pool,
        gen=gen,
        num_requests=args.sim_num_requests,
        max_concurrent=args.sim_max_concurrent,
        log_interval=args.sim_log_interval,
    )

    entries = sim.run()
    print_log(entries)
    print_summary(entries)


if __name__ == "__main__":
    main()
