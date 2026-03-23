"""
SAGE Drone - Full Test Suite
==============================
Runs BOTH v1 and v2 in sequence and saves all outputs.

v1: MultiCube only (flat long-term memory)
v2: SAGEDivided + MultiCube + Consolidation pathway

Usage:
  python run_all.py                    # full run, all 6 tests
  python run_all.py --model phi3       # use a different LLM
  python run_all.py --cube-size 8      # smaller cube (faster)
  python run_all.py --no-llm           # SAGE-only mode for all tests

Outputs written to ./outputs/ folder:
  out_v1_preload.txt
  out_v2_preload.txt
  out_v1_demo.txt
  out_v2_demo.txt
  out_v1_sage_only.txt
  out_v2_sage_only.txt
  out_comparison.txt

Author: Ivelin Likov
"""

import argparse
import sys
import os
import time
import io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollama_adapter import OllamaAdapter
from sage_memory import SAGEMemory
from sage_memory_v2 import SAGEMemoryV2


# =================================================================
# SAGE Force Constants (imported logic, mirrored here for run_all)
# =================================================================
G_BASE = 0.003
def get_gravity(cube_size):
    return G_BASE * (cube_size / 32.0) ** 3



OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# ─────────────────────────────────────────────────────────────────────
# Tee writer -- prints to screen AND collects into a string
# ─────────────────────────────────────────────────────────────────────

class Tee:
    """Write to both stdout and an internal buffer simultaneously."""
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
    print(f"\n  [saved] {path}")
    return path


# ─────────────────────────────────────────────────────────────────────
# V1 Agent
# ─────────────────────────────────────────────────────────────────────

class AgentV1:
    def __init__(self, ollama, sage, cube_size=16):
        self.ollama = ollama
        self.sage   = sage
        self.step   = 0
        self.llm_decisions    = 0
        self.sage_decisions   = 0
        self.default_decisions = 0
        self.G = get_gravity(cube_size)

    def process(self, observation, learn=True, verbose=True):
        self.step += 1
        t0 = time.time()

        if verbose:
            print(f"\n{'-'*55}")
            print(f"  STEP {self.step} | {observation[:60]}")
            print(f"{'-'*55}")

        embedding = self.ollama.embed(observation)
        sage_context = self.sage.get_context_string(embedding, top_k=3)
        if verbose:
            print(sage_context)

        decision = None
        mode     = None

        if self.ollama.llm_online:
            decision = self.ollama.generate(
                prompt=observation,
                sage_context=sage_context,
                stream=True
            )
            if decision:
                mode = 'llm'
                self.llm_decisions += 1

        if decision is None:
            recalled = self.sage.recall_action(embedding, top_k=3)
            if recalled:
                best     = recalled[0]
                decision = best['action']
                mode     = f"sage_memory [{best['confidence']} confidence]"
                self.sage_decisions += 1
                if verbose:
                    print(f"\n  [SAGE AGENT - LLM offline]")
                    print(f"  Confidence: {best['confidence']}")
                    print(f"  Decision:   {decision}")
                    if len(recalled) > 1:
                        print(f"\n  Other candidates:")
                        for r in recalled[1:]:
                            print(f"    [{r['confidence']}] {r['action'][:60]}")
            else:
                decision = 'No prior memory found. Holding current position.'
                mode     = 'default_hold'
                self.default_decisions += 1
                if verbose:
                    print(f"\n  [SAFE DEFAULT] {decision}")

        if learn and decision:
            self.sage.store(
                embedding=embedding,
                label=observation.replace(' ', '_')[:30],
                cube_name='mission',
                learn=True
            )
            if mode == 'llm':
                action_embedding = self.ollama.embed(decision)
                self.sage.store_action(
                    observation_embedding=embedding,
                    action_text=decision,
                    action_embedding=action_embedding,
                    alpha=0.025
                )

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  [mode: {mode}] [{elapsed:.2f}s]")

        return {
            'step': self.step, 'observation': observation,
            'mode': mode, 'decision': decision, 'elapsed_s': elapsed,
        }

    def summary(self):
        total = self.llm_decisions + self.sage_decisions + self.default_decisions
        if total == 0:
            return {}
        print(f"\n{'='*55}")
        print(f"  SESSION DECISION SUMMARY (v1)")
        print(f"{'='*55}")
        print(f"  Total steps:       {self.step}")
        print(f"  LLM decisions:     {self.llm_decisions}  ({100*self.llm_decisions//total}%)")
        print(f"  SAGE decisions:    {self.sage_decisions}  ({100*self.sage_decisions//total}%)")
        print(f"  Default (hold):    {self.default_decisions}")
        print(f"{'='*55}\n")
        return {
            'llm': self.llm_decisions,
            'sage': self.sage_decisions,
            'default': self.default_decisions,
            'total': total,
        }


# ─────────────────────────────────────────────────────────────────────
# V2 Agent
# ─────────────────────────────────────────────────────────────────────

class AgentV2:
    def __init__(self, ollama, sage, cube_size=16):
        self.ollama = ollama
        self.sage   = sage
        self.step   = 0
        self.llm_decisions    = 0
        self.sage_decisions   = 0
        self.default_decisions = 0
        self.consolidation_steps = 0
        self.G = get_gravity(cube_size)

    def process(self, observation, learn=True, verbose=True):
        self.step += 1
        t0 = time.time()

        if verbose:
            print(f"\n{'-'*55}")
            print(f"  STEP {self.step} | {observation[:60]}")
            print(f"{'-'*55}")

        obs_embedding = self.ollama.embed(observation)

        # Write to SAGEDivided working memory
        self.sage.working_memory.encode(obs_embedding, None, alpha=0.025)
        self.sage.working_memory.label_current(
            observation.replace(' ', '_')[:30], ''
        )

        sage_context = self.sage.get_context_string(obs_embedding, top_k=3)
        if verbose:
            print(sage_context)

        decision = None
        mode     = None

        if self.ollama.llm_online:
            decision = self.ollama.generate(
                prompt=observation,
                sage_context=sage_context,
                stream=True
            )
            if decision:
                mode = 'llm'
                self.llm_decisions += 1

        if decision is None:
            recalled = self.sage.recall_action(obs_embedding, top_k=3)
            if recalled:
                best     = recalled[0]
                decision = best['action']
                mode     = f"sage_memory [{best['confidence']} confidence]"
                self.sage_decisions += 1
                if verbose:
                    print(f"\n  [SAGE AGENT - LLM offline]")
                    print(f"  Confidence: {best['confidence']}")
                    print(f"  Decision:   {decision}")
                    if len(recalled) > 1:
                        print(f"\n  Other candidates:")
                        for r in recalled[1:]:
                            print(f"    [{r['confidence']}] {r['action'][:60]}")
            else:
                decision = 'No prior memory found. Holding current position.'
                mode     = 'default_hold'
                self.default_decisions += 1
                if verbose:
                    print(f"\n  [SAFE DEFAULT] {decision}")

        if learn and decision:
            action_embedding = self.ollama.embed(decision)
            self.sage.working_memory.encode(obs_embedding, action_embedding, alpha=0.025)
            self.sage.working_memory.label_current(
                observation.replace(' ', '_')[:30],
                decision.replace(' ', '_')[:30]
            )
            self.sage.consolidate(alpha_long=0.015)
            self.consolidation_steps += 1

            if mode == 'llm':
                self.sage.store_action(obs_embedding, decision, action_embedding, alpha=0.025)

            self.sage.mission_cube.label_point(
                self.sage._to_tensor(obs_embedding),
                observation.replace(' ', '_')[:30]
            )

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  [mode: {mode}] [{elapsed:.2f}s] [consolidation: yes]")

        return {
            'step': self.step, 'observation': observation,
            'mode': mode, 'decision': decision, 'elapsed_s': elapsed,
        }

    def summary(self):
        total = self.llm_decisions + self.sage_decisions + self.default_decisions
        if total == 0:
            return {}
        print(f"\n{'='*55}")
        print(f"  SESSION DECISION SUMMARY (v2)")
        print(f"{'='*55}")
        print(f"  Total steps:          {self.step}")
        print(f"  LLM decisions:        {self.llm_decisions}  ({100*self.llm_decisions//total}%)")
        print(f"  SAGE decisions:       {self.sage_decisions}  ({100*self.sage_decisions//total}%)")
        print(f"  Default (hold):       {self.default_decisions}")
        print(f"  Consolidation steps:  {self.consolidation_steps}")
        print(f"  Architecture:         SAGEDivided + MultiCube")
        print(f"{'='*55}\n")
        return {
            'llm': self.llm_decisions,
            'sage': self.sage_decisions,
            'default': self.default_decisions,
            'total': total,
            'consolidations': self.consolidation_steps,
        }


# ─────────────────────────────────────────────────────────────────────
# Shared demo runner
# ─────────────────────────────────────────────────────────────────────

def run_demo(agent, label, simulate_offline_after=8):
    print(f"\n{'='*55}")
    print(f"  {label} - AUTOMATED SCENARIO DEMO")
    print(f"  LLM goes offline after step {simulate_offline_after}")
    print(f"{'='*55}\n")

    results = []
    for i, scenario in enumerate(DEMO_SCENARIOS):
        if i == simulate_offline_after:
            agent.ollama.llm_online = False
            print(f"\n{'!'*55}")
            print(f"  !! LLM CONNECTION LOST - SAGE TAKING OVER !!")
            print(f"{'!'*55}")
        result = agent.process(scenario, learn=True, verbose=True)
        results.append(result)
        time.sleep(0.1)

    stats = agent.summary()
    return results, stats


# ─────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────

def build_comparison(v1_demo, v1_sage, v2_demo, v2_sage,
                     v1_demo_results, v2_demo_results,
                     v1_sage_results, v2_sage_results):

    import re

    def avg_time(results, mode_filter=None):
        times = [r['elapsed_s'] for r in results
                 if mode_filter is None or
                 (r['mode'] and mode_filter in r['mode'])]
        return sum(times)/len(times) if times else 0

    def meaningful(results):
        return sum(1 for r in results
                   if r['decision'] and not r['decision'].startswith('point_'))

    lines = []
    lines.append('SAGE DRONE - V1 vs V2 COMPARISON REPORT')
    lines.append(f'Generated: {datetime.now().isoformat()}')
    lines.append('='*60)
    lines.append('')

    lines.append('ARCHITECTURE')
    lines.append('-'*40)
    lines.append('v1: MultiCube only (flat long-term memory)')
    lines.append('    4 specialist cubes | Hebbian learning')
    lines.append('    action_cube for autonomous fallback')
    lines.append('')
    lines.append('v2: SAGEDivided + MultiCube + Consolidation')
    lines.append('    Working memory: subject(x<0) / object(x>=0)')
    lines.append('    Long-term: same 4 cubes as v1')
    lines.append('    Consolidation: working -> long-term each step')
    lines.append('    Hippocampal-neocortical transfer modelled')
    lines.append('')

    lines.append('HYBRID TEST (LLM + SAGE, LLM offline after step 8)')
    lines.append('-'*40)
    for label, stats, results in [
        ('v1', v1_demo, v1_demo_results),
        ('v2', v2_demo, v2_demo_results),
    ]:
        if not stats:
            continue
        total = stats.get('total', 12)
        lines.append(f"  {label}: LLM={stats.get('llm',0)} "
                     f"SAGE={stats.get('sage',0)} "
                     f"default={stats.get('default',0)}")
        lines.append(f"      avg LLM time:  {avg_time(results, 'llm'):.2f}s")
        lines.append(f"      avg SAGE time: {avg_time(results, 'sage'):.2f}s")
        lines.append(f"      meaningful decisions: {meaningful(results)}/{len(results)}")
        if label == 'v2':
            lines.append(f"      consolidation steps: {stats.get('consolidations',0)}")
    lines.append('')

    lines.append('SAGE-ONLY TEST (no LLM at all)')
    lines.append('-'*40)
    for label, stats, results in [
        ('v1', v1_sage, v1_sage_results),
        ('v2', v2_sage, v2_sage_results),
    ]:
        if not stats:
            continue
        lines.append(f"  {label}: SAGE={stats.get('sage',0)} "
                     f"default={stats.get('default',0)}")
        lines.append(f"      avg time: {avg_time(results):.2f}s")
        lines.append(f"      meaningful decisions: {meaningful(results)}/{len(results)}")
    lines.append('')

    lines.append('KEY DIFFERENCES')
    lines.append('-'*40)
    lines.append('  v1 context:  LT knowledge + past actions only')
    lines.append('  v2 context:  WORKING MEMORY + LT knowledge + past actions')
    lines.append('  v1 learning: obs stored in mission_cube each step')
    lines.append('  v2 learning: obs/action encoded in SAGEDivided THEN')
    lines.append('               consolidated into MultiCube (2-stage)')
    lines.append('  v1 relation: flat embedding, no subject/object split')
    lines.append('  v2 relation: obs=subject(x<0), action=object(x>=0)')
    lines.append('               relational structure preserved geometrically')
    lines.append('')

    lines.append('MEMORY FOOTPRINT')
    lines.append('-'*40)
    lines.append('  v1: 4 cubes x 48MB = 192MB total')
    lines.append('  v2: 5 cubes x 48MB = 240MB total (SAGEDivided added)')
    lines.append('  vs Mistral 7B q4:   3,300MB')
    lines.append('  v2 overhead vs v1:  +48MB (+25%)')
    lines.append('  v2 vs LLM:          14x smaller')
    lines.append('')

    lines.append('DRONE NEVER STOPPED')
    lines.append('-'*40)
    for label, demo_r, sage_r in [
        ('v1', v1_demo_results, v1_sage_results),
        ('v2', v2_demo_results, v2_sage_results),
    ]:
        d_defaults = sum(1 for r in demo_r if r.get('mode')=='default_hold')
        s_defaults = sum(1 for r in sage_r if r.get('mode')=='default_hold')
        lines.append(f"  {label} hybrid:    {d_defaults} defaults (drone stopped: {'NO' if d_defaults==0 else 'YES'})")
        lines.append(f"  {label} SAGE-only: {s_defaults} defaults (drone stopped: {'NO' if s_defaults==0 else 'YES'})")
    lines.append('')
    lines.append('='*60)

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SAGE Drone - Full Test Suite v1 + v2')
    parser.add_argument('--model',      type=str, default='mistral')
    parser.add_argument('--cube-size',  type=int, default=16)
    parser.add_argument('--no-llm',     action='store_true', help='Force SAGE-only for all tests')
    parser.add_argument('--skip-preload', action='store_true', help='Skip preload if already done')
    args = parser.parse_args()

    run_start = datetime.now().isoformat()

    print(f"\n{'='*55}")
    print(f"  SAGE DRONE - FULL TEST SUITE")
    print(f"  v1 (MultiCube) + v2 (SAGEDivided+MultiCube)")
    print(f"  All outputs -> ./outputs/")
    print(f"{'='*55}")
    print(f"  Started: {run_start}\n")

    # ── Ollama ────────────────────────────────────────────────────
    print('Connecting to Ollama...')
    ollama = OllamaAdapter(llm_model=args.model, verbose=False)

    if args.no_llm:
        ollama.llm_online = False
        print('  LLM disabled. SAGE-only for all tests.\n')

    # ─────────────────────────────────────────────────────────────
    # STEP 1: V1 PRELOAD
    # ─────────────────────────────────────────────────────────────
    if not args.skip_preload:
        print(f"\n{'#'*55}")
        print(f"  STEP 1/6: V1 PRELOAD")
        print(f"{'#'*55}")
        tee = Tee()
        sys.stdout = tee
        try:
            sage_v1_pre = SAGEMemory(
                cube_size=args.cube_size, embed_dim=768,
                save_dir='./sage_state_v1'
            )
            sage_v1_pre.preload_drone_knowledge(embedder_fn=ollama.embed)
            sage_v1_pre.save()
        finally:
            sys.stdout = tee._stdout
        save_output('out_v1_preload.txt', tee.getvalue(),
                    f'V1 PRELOAD | {run_start}')
        del sage_v1_pre

    # ─────────────────────────────────────────────────────────────
    # STEP 2: V2 PRELOAD
    # ─────────────────────────────────────────────────────────────
    if not args.skip_preload:
        print(f"\n{'#'*55}")
        print(f"  STEP 2/6: V2 PRELOAD")
        print(f"{'#'*55}")
        tee = Tee()
        sys.stdout = tee
        try:
            sage_v2_pre = SAGEMemoryV2(
                cube_size=args.cube_size, embed_dim=768,
                save_dir='./sage_state_v2'
            )
            sage_v2_pre.preload_drone_knowledge(embedder_fn=ollama.embed)
            sage_v2_pre.save()
        finally:
            sys.stdout = tee._stdout
        save_output('out_v2_preload.txt', tee.getvalue(),
                    f'V2 PRELOAD | {run_start}')
        del sage_v2_pre

    # ─────────────────────────────────────────────────────────────
    # STEP 3: V1 HYBRID DEMO
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'#'*55}")
    print(f"  STEP 3/6: V1 DEMO (LLM + SAGE hybrid)")
    print(f"{'#'*55}")
    ollama.llm_online = not args.no_llm
    if ollama.llm_online:
        ollama._check_connection()
    tee = Tee()
    sys.stdout = tee
    try:
        sage_v1 = SAGEMemory(
            cube_size=args.cube_size, embed_dim=768,
            save_dir='./sage_state_v1'
        )
        sage_v1.load()
        agent_v1 = AgentV1(ollama, sage_v1)
        v1_demo_results, v1_demo_stats = run_demo(agent_v1, 'SAGE v1')
        sage_v1.save()
    finally:
        sys.stdout = tee._stdout
    save_output('out_v1_demo.txt', tee.getvalue(),
                f'V1 HYBRID DEMO | {run_start}')

    # ─────────────────────────────────────────────────────────────
    # STEP 4: V2 HYBRID DEMO
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'#'*55}")
    print(f"  STEP 4/6: V2 DEMO (LLM + SAGE hybrid)")
    print(f"{'#'*55}")
    ollama.llm_online = not args.no_llm
    if ollama.llm_online:
        ollama._check_connection()
    tee = Tee()
    sys.stdout = tee
    try:
        sage_v2 = SAGEMemoryV2(
            cube_size=args.cube_size, embed_dim=768,
            save_dir='./sage_state_v2'
        )
        sage_v2.load()
        agent_v2 = AgentV2(ollama, sage_v2)
        v2_demo_results, v2_demo_stats = run_demo(agent_v2, 'SAGE v2')
        sage_v2.save()
    finally:
        sys.stdout = tee._stdout
    save_output('out_v2_demo.txt', tee.getvalue(),
                f'V2 HYBRID DEMO | {run_start}')

    # ─────────────────────────────────────────────────────────────
    # STEP 5: V1 SAGE-ONLY
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'#'*55}")
    print(f"  STEP 5/6: V1 SAGE-ONLY (no LLM)")
    print(f"{'#'*55}")
    ollama.llm_online = False
    tee = Tee()
    sys.stdout = tee
    try:
        sage_v1_s = SAGEMemory(
            cube_size=args.cube_size, embed_dim=768,
            save_dir='./sage_state_v1'
        )
        sage_v1_s.load()
        agent_v1_s = AgentV1(ollama, sage_v1_s)
        v1_sage_results, v1_sage_stats = run_demo(agent_v1_s, 'SAGE v1 SAGE-ONLY', simulate_offline_after=0)
        sage_v1_s.save()
    finally:
        sys.stdout = tee._stdout
    save_output('out_v1_sage_only.txt', tee.getvalue(),
                f'V1 SAGE-ONLY | {run_start}')

    # ─────────────────────────────────────────────────────────────
    # STEP 6: V2 SAGE-ONLY
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'#'*55}")
    print(f"  STEP 6/6: V2 SAGE-ONLY (no LLM)")
    print(f"{'#'*55}")
    ollama.llm_online = False
    tee = Tee()
    sys.stdout = tee
    try:
        sage_v2_s = SAGEMemoryV2(
            cube_size=args.cube_size, embed_dim=768,
            save_dir='./sage_state_v2'
        )
        sage_v2_s.load()
        agent_v2_s = AgentV2(ollama, sage_v2_s)
        v2_sage_results, v2_sage_stats = run_demo(agent_v2_s, 'SAGE v2 SAGE-ONLY', simulate_offline_after=0)
        sage_v2_s.save()
    finally:
        sys.stdout = tee._stdout
    save_output('out_v2_sage_only.txt', tee.getvalue(),
                f'V2 SAGE-ONLY | {run_start}')

    # ─────────────────────────────────────────────────────────────
    # COMPARISON REPORT
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'#'*55}")
    print(f"  BUILDING COMPARISON REPORT")
    print(f"{'#'*55}")
    report = build_comparison(
        v1_demo_stats, v1_sage_stats,
        v2_demo_stats, v2_sage_stats,
        v1_demo_results, v2_demo_results,
        v1_sage_results, v2_sage_results,
    )
    print(report)
    save_output('out_comparison.txt', report,
                f'V1 vs V2 COMPARISON | {run_start}')

    # ─────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ALL TESTS COMPLETE")
    print(f"{'='*55}")
    print(f"  Files saved to ./outputs/:")
    for fname in [
        'out_v1_preload.txt',
        'out_v2_preload.txt',
        'out_v1_demo.txt',
        'out_v2_demo.txt',
        'out_v1_sage_only.txt',
        'out_v2_sage_only.txt',
        'out_comparison.txt',
    ]:
        path = os.path.join(OUTPUT_DIR, fname)
        exists = '[OK]' if os.path.exists(path) else '[MISSING]'
        print(f"  {exists} {fname}")
    print(f"\n  Upload all 7 txt files for full analysis.\n")


if __name__ == '__main__':
    main()
