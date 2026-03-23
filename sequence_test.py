"""
SAGE Sequence Awareness Test
==============================
Tests two lightweight mechanisms for adding sequence awareness
to SAGE without any learned weights or backpropagation.

BASELINE (current SAGE-only):
  Each observation queried independently.
  No knowledge of what came before.

TEST 1 - Sinusoidal Positional Encoding:
  Fixed sin/cos position vectors added to each embedding.
  Step 1 gets pos_1, step 2 gets pos_2, etc.
  Same technique as original Transformer (Vaswani et al. 2017)
  but here applied to SAGE cube queries — no learned weights.
  query = normalize(embed + scale * sinusoidal(step))

TEST 2 - Context Delta:
  Query encodes direction of change, not just current state.
  query = normalize(current + weight * (current - previous))
  Drone that was nominal and detects a tree produces a different
  delta query than one already in emergency mode.
  Zero new parameters. Two lines of code.

TEST 3 - Combined (Positional + Delta):
  Both mechanisms together.
  query = normalize(embed + pos_scale * pos(step) +
                    delta_weight * (current - previous))

All three are compared against the pure baseline.
Metric: meaningful decisions (real action text vs point_XXXX)

Usage:
  python sequence_test.py
  python sequence_test.py --no-llm
  python sequence_test.py --model phi3

Output:
  outputs/out_seq_baseline.txt
  outputs/out_seq_positional.txt
  outputs/out_seq_delta.txt
  outputs/out_seq_combined.txt
  outputs/out_seq_comparison.txt

Author: Ivelin Likov
"""

import argparse
import sys
import os
import time
import io
import math
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollama_adapter import OllamaAdapter
from sage_memory import SAGEMemory

OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Sequence encoding mechanisms
# ─────────────────────────────────────────────────────────────────

def sinusoidal_position(step: int, embed_dim: int = 768) -> list:
    """
    Generate a fixed sinusoidal position vector for step N.

    Same formula as Vaswani et al. (Attention Is All You Need, 2017)
    but used here without learned weights -- purely as a geometric
    offset to distinguish step 1 from step 2 in embedding space.

    pos[2i]   = sin(step / 10000^(2i/d))
    pos[2i+1] = cos(step / 10000^(2i/d))
    """
    pos = []
    for i in range(embed_dim):
        if i % 2 == 0:
            pos.append(math.sin(step / (10000 ** (i / embed_dim))))
        else:
            pos.append(math.cos(step / (10000 ** ((i-1) / embed_dim))))
    return pos


def add_positional(embedding: list, step: int,
                   embed_dim: int = 768, scale: float = 0.3) -> list:
    """
    Add scaled sinusoidal position to an embedding.
    scale=0.3 means position adds 30% of its norm to the query.
    Lower scale = weaker positional signal.
    """
    pos = sinusoidal_position(step, embed_dim)
    combined = [e + scale * p for e, p in zip(embedding, pos)]
    # Renormalise to unit length
    norm = math.sqrt(sum(x**2 for x in combined))
    return [x / norm for x in combined] if norm > 0 else combined


def add_delta(current: list, previous: list,
              weight: float = 0.3) -> list:
    """
    Add direction-of-change signal to the current embedding.
    delta = current - previous encodes what changed since last step.
    query = normalize(current + weight * delta)

    First step: previous is None, delta = zero, falls back to baseline.
    """
    if previous is None:
        return current
    delta = [c - p for c, p in zip(current, previous)]
    combined = [c + weight * d for c, d in zip(current, delta)]
    norm = math.sqrt(sum(x**2 for x in combined))
    return [x / norm for x in combined] if norm > 0 else combined


def add_combined(embedding: list, step: int, previous: list = None,
                 embed_dim: int = 768,
                 pos_scale: float = 0.2,
                 delta_weight: float = 0.2) -> list:
    """
    Positional encoding + context delta together.
    Smaller scales for each since both are additive.
    """
    pos = sinusoidal_position(step, embed_dim)
    if previous is not None:
        delta = [c - p for c, p in zip(embedding, previous)]
    else:
        delta = [0.0] * embed_dim

    combined = [
        e + pos_scale * p + delta_weight * d
        for e, p, d in zip(embedding, pos, delta)
    ]
    norm = math.sqrt(sum(x**2 for x in combined))
    return [x / norm for x in combined] if norm > 0 else combined


# ─────────────────────────────────────────────────────────────────
# Tee writer
# ─────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self):
        self.buf = io.StringIO()
        self._stdout = sys.stdout
    def write(self, data):
        self._stdout.write(data)
        self._stdout.flush()
        self.buf.write(data)
    def flush(self):
        self._stdout.flush()
    def getvalue(self):
        return self.buf.getvalue()


def save_output(filename, content, header=''):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
            f.write('='*60 + '\n\n')
        f.write(content)
    print(f"  [saved] {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────

DEMO_SCENARIOS = [
    'All systems nominal, beginning survey pattern',
    'Reached waypoint alpha, proceeding to waypoint beta',
    'Tree detected 15 metres ahead on current flight path',
    'Power lines detected crossing flight path at 25 metres',
    'Person detected directly below drone at 30 metres altitude',
    'Battery level at 20 percent',
    'GPS signal degraded, accuracy reduced to 10 metres',
    'Strong wind detected, drone drifting 2 metres per second east',
    'Connection to ground station lost',
    'Low visibility fog reducing camera range to 20 metres',
    'Obstacle detected on current flight path',
    'Mission area survey 80 percent complete',
]


# ─────────────────────────────────────────────────────────────────
# Agent with configurable sequence encoding
# ─────────────────────────────────────────────────────────────────

class SequenceAgent:
    """
    SAGE-only agent with configurable sequence encoding.
    LLM is disabled — tests pure geometric memory recall.
    """

    def __init__(self, ollama: OllamaAdapter, sage: SAGEMemory,
                 mode: str = 'baseline'):
        self.ollama   = ollama
        self.sage     = sage
        self.mode     = mode   # baseline | positional | delta | combined
        self.step     = 0
        self.previous_embedding = None

        self.sage_decisions   = 0
        self.default_decisions = 0
        self.meaningful_count  = 0

        print(f"\n  [SequenceAgent] mode={mode}")

    def process(self, observation: str, verbose: bool = True) -> dict:
        self.step += 1
        t0 = time.time()

        if verbose:
            print(f"\n{'-'*55}")
            print(f"  STEP {self.step} | {observation[:60]}")
            print(f"{'-'*55}")

        # Base embedding
        base_embedding = self.ollama.embed(observation)

        # Apply sequence encoding
        if self.mode == 'baseline':
            query_embedding = base_embedding

        elif self.mode == 'positional':
            query_embedding = add_positional(
                base_embedding, self.step,
                embed_dim=self.ollama.embed_dim,
                scale=0.3
            )

        elif self.mode == 'delta':
            query_embedding = add_delta(
                base_embedding,
                self.previous_embedding,
                weight=0.3
            )

        elif self.mode == 'combined':
            query_embedding = add_combined(
                base_embedding, self.step,
                self.previous_embedding,
                embed_dim=self.ollama.embed_dim,
                pos_scale=0.2,
                delta_weight=0.2
            )
        else:
            query_embedding = base_embedding

        # Recall from action_cube using modified query
        recalled = self.sage.recall_action(query_embedding, top_k=3)

        decision = None
        if recalled:
            best     = recalled[0]
            decision = best['action']
            mode_tag = f"sage_memory [{best['confidence']} confidence]"
            self.sage_decisions += 1

            meaningful = not decision.startswith('point_')
            if meaningful:
                self.meaningful_count += 1

            if verbose:
                tag = '[TEXT]' if meaningful else '[POINT]'
                print(f"  {tag} Confidence: {best['confidence']}")
                print(f"  Decision: {decision[:80]}")
                if len(recalled) > 1:
                    print(f"  Other candidates:")
                    for r in recalled[1:]:
                        t = '[TEXT]' if not r['action'].startswith('point') else '[POINT]'
                        print(f"    {t} [{r['confidence']}] {r['action'][:60]}")
        else:
            decision = 'No memory found. Holding position.'
            mode_tag = 'default_hold'
            self.default_decisions += 1
            if verbose:
                print(f"  [DEFAULT] {decision}")

        # Store current as previous for next step
        self.previous_embedding = base_embedding

        # Store observation in mission cube (keeps memory growing)
        self.sage.store(
            embedding=base_embedding,
            label=observation.replace(' ', '_')[:30],
            cube_name='mission',
            learn=True
        )

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  [mode: {self.mode}] [{elapsed:.2f}s]")

        return {
            'step':       self.step,
            'observation': observation,
            'decision':   decision,
            'meaningful': not (decision or '').startswith('point_'),
            'elapsed_s':  elapsed,
        }

    def summary(self) -> dict:
        total = self.sage_decisions + self.default_decisions
        print(f"\n{'='*55}")
        print(f"  SEQUENCE TEST SUMMARY — mode: {self.mode.upper()}")
        print(f"{'='*55}")
        print(f"  Total steps:         {self.step}")
        print(f"  SAGE decisions:      {self.sage_decisions}")
        print(f"  Default (hold):      {self.default_decisions}")
        print(f"  Meaningful:          {self.meaningful_count}/{self.step}")
        pct = int(100 * self.meaningful_count / self.step) if self.step > 0 else 0
        print(f"  Quality:             {pct}%")
        print(f"{'='*55}\n")
        return {
            'mode':        self.mode,
            'meaningful':  self.meaningful_count,
            'total':       self.step,
            'defaults':    self.default_decisions,
            'quality_pct': pct,
        }


# ─────────────────────────────────────────────────────────────────
# Run one full sequence test
# ─────────────────────────────────────────────────────────────────

def run_sequence_test(ollama, cube_size, mode, save_dir, label):
    """Load saved SAGE state and run SAGE-only with given encoding mode."""

    print(f"\n{'#'*55}")
    print(f"  SEQUENCE TEST: {mode.upper()}")
    print(f"  {label}")
    print(f"{'#'*55}")

    tee = Tee()
    sys.stdout = tee
    try:
        # Load pre-trained state (from run_all.py preload)
        sage = SAGEMemory(
            cube_size=cube_size,
            embed_dim=768,
            save_dir=save_dir
        )
        loaded = sage.load()
        if not loaded:
            print(f"  WARNING: No saved state found at {save_dir}")
            print(f"  Run run_all.py first to preload SAGE knowledge.")

        # Force LLM offline — pure SAGE
        ollama.llm_online = False

        agent = SequenceAgent(ollama, sage, mode=mode)

        results = []
        for scenario in DEMO_SCENARIOS:
            result = agent.process(scenario, verbose=True)
            results.append(result)
            time.sleep(0.05)

        stats = agent.summary()
        sage.save()

    finally:
        sys.stdout = tee._stdout

    content = tee.getvalue()
    save_output(
        f'out_seq_{mode}.txt',
        content,
        f'SEQUENCE TEST: {mode.upper()} | {datetime.now().isoformat()}'
    )
    return stats, results


# ─────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────

def build_seq_comparison(all_stats: list) -> str:
    lines = []
    lines.append('SAGE SEQUENCE AWARENESS - TEST COMPARISON')
    lines.append(f'Generated: {datetime.now().isoformat()}')
    lines.append('='*60)
    lines.append('')
    lines.append('MECHANISM DESCRIPTIONS')
    lines.append('-'*40)
    lines.append('baseline:   Raw embedding query — no sequence context')
    lines.append('positional: Fixed sinusoidal position added to query')
    lines.append('            query = normalize(embed + 0.3 * sin_pos(step))')
    lines.append('delta:      Direction-of-change added to query')
    lines.append('            query = normalize(embed + 0.3 * (curr - prev))')
    lines.append('combined:   Both positional + delta together')
    lines.append('            query = normalize(embed + 0.2*pos + 0.2*delta)')
    lines.append('')
    lines.append('RESULTS')
    lines.append('-'*40)
    lines.append(f"{'Mode':<12} {'Meaningful':>10} {'Total':>7} "
                 f"{'Quality':>8} {'Defaults':>9}")
    lines.append('-'*45)

    baseline_quality = None
    for s in all_stats:
        if s['mode'] == 'baseline':
            baseline_quality = s['quality_pct']
        delta = ''
        if baseline_quality is not None and s['mode'] != 'baseline':
            diff = s['quality_pct'] - baseline_quality
            delta = f"  ({'+' if diff >= 0 else ''}{diff}% vs baseline)"
        lines.append(
            f"  {s['mode']:<12} {s['meaningful']:>8}/{s['total']:<4}"
            f"  {s['quality_pct']:>6}%"
            f"  {s['defaults']:>7}"
            f"{delta}"
        )

    lines.append('')
    lines.append('INTERPRETATION')
    lines.append('-'*40)

    best = max(all_stats, key=lambda x: x['quality_pct'])
    baseline = next(s for s in all_stats if s['mode'] == 'baseline')

    if best['mode'] == 'baseline':
        lines.append('Baseline matched or beat all sequence encodings.')
        lines.append('Sequence context did not improve recall quality.')
        lines.append('Possible reasons:')
        lines.append('  - Action memory too sparse (only 12 seeded pairs)')
        lines.append('  - Positional signal disrupts semantic similarity')
        lines.append('  - More flights needed before sequence context helps')
    else:
        improvement = best['quality_pct'] - baseline['quality_pct']
        lines.append(f"Best mechanism: {best['mode'].upper()}")
        lines.append(f"Improvement over baseline: +{improvement}%")
        lines.append(f"  ({best['meaningful']}/{best['total']} meaningful"
                     f" vs {baseline['meaningful']}/{baseline['total']} baseline)")
        if improvement > 10:
            lines.append('SIGNIFICANT improvement — sequence awareness is helping.')
            lines.append('Recommend integrating this mechanism into v3.')
        elif improvement > 0:
            lines.append('Modest improvement — sequence context adds some benefit.')
            lines.append('May improve further with more flight data.')
        else:
            lines.append('Marginal improvement — may not justify added complexity.')

    lines.append('')
    lines.append('NEXT STEPS')
    lines.append('-'*40)
    lines.append('If positional or delta improved quality:')
    lines.append('  -> Integrate into sage_memory_v3.py')
    lines.append('  -> This becomes the foundation for GeoMemFormer')
    lines.append('  -> Sequence-aware SAGE = direct path to SAGE-Transformer')
    lines.append('')
    lines.append('If no improvement:')
    lines.append('  -> More flight data needed first (confidence too low)')
    lines.append('  -> Re-test after 50+ real flights')
    lines.append('  -> The mechanism is correct but memory is too sparse')
    lines.append('')
    lines.append('='*60)

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SAGE Sequence Awareness Test'
    )
    parser.add_argument('--model',     type=str, default='mistral')
    parser.add_argument('--cube-size', type=int, default=16)
    parser.add_argument('--save-dir',  type=str, default='./sage_state_v1',
                        help='Path to existing SAGE state from run_all.py')
    parser.add_argument('--no-llm',    action='store_true')
    args = parser.parse_args()

    run_start = datetime.now().isoformat()

    print(f"\n{'='*55}")
    print(f"  SAGE SEQUENCE AWARENESS TEST SUITE")
    print(f"  Testing: positional encoding + context delta")
    print(f"  Based on v1 SAGE state: {args.save_dir}")
    print(f"{'='*55}")
    print(f"  Started: {run_start}\n")

    print('Connecting to Ollama...')
    ollama = OllamaAdapter(llm_model=args.model, verbose=False)
    # All sequence tests run SAGE-only regardless of flag
    ollama.llm_online = False
    print('  LLM disabled for all sequence tests (SAGE-only).\n')

    all_stats = []

    # Test 1: Baseline
    stats, _ = run_sequence_test(
        ollama, args.cube_size, 'baseline',
        args.save_dir,
        'Pure SAGE-only, no sequence context (current system)'
    )
    all_stats.append(stats)

    # Test 2: Sinusoidal positional encoding
    stats, _ = run_sequence_test(
        ollama, args.cube_size, 'positional',
        args.save_dir,
        'Fixed sinusoidal position vectors added to query'
    )
    all_stats.append(stats)

    # Test 3: Context delta
    stats, _ = run_sequence_test(
        ollama, args.cube_size, 'delta',
        args.save_dir,
        'Direction-of-change (current - previous) added to query'
    )
    all_stats.append(stats)

    # Test 4: Combined
    stats, _ = run_sequence_test(
        ollama, args.cube_size, 'combined',
        args.save_dir,
        'Positional + delta combined'
    )
    all_stats.append(stats)

    # Comparison report
    print(f"\n{'#'*55}")
    print(f"  BUILDING SEQUENCE COMPARISON REPORT")
    print(f"{'#'*55}")
    report = build_seq_comparison(all_stats)
    print(report)
    save_output(
        'out_seq_comparison.txt',
        report,
        f'SEQUENCE COMPARISON | {run_start}'
    )

    print(f"\n{'='*55}")
    print(f"  SEQUENCE TESTS COMPLETE")
    print(f"{'='*55}")
    print(f"  Files saved to ./outputs/:")
    for fname in [
        'out_seq_baseline.txt',
        'out_seq_positional.txt',
        'out_seq_delta.txt',
        'out_seq_combined.txt',
        'out_seq_comparison.txt',
    ]:
        path = os.path.join(OUTPUT_DIR, fname)
        status = '[OK]' if os.path.exists(path) else '[MISSING]'
        print(f"  {status} {fname}")
    print(f"\n  Upload out_seq_comparison.txt for analysis.\n")


if __name__ == '__main__':
    main()
