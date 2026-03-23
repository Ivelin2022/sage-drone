"""
SAGE Memory v2 - Full Architecture
=====================================
SAGEDivided (short-term) + MultiCube (long-term) + Consolidation pathway

Architecture:
  SAGEDivided  <-- each observation
       |
       | consolidation (hippocampal-neocortical transfer)
       v
  MultiCube:
    nav_cube     -- waypoints, routes, landmarks
    object_cube  -- obstacles, terrain, objects
    mission_cube -- tasks, goals, rules
    action_cube  -- observation->action pairs (autonomous fallback)

The consolidation pathway is the strongest novelty claim:
  Working memory (SAGEDivided) -> Long-term memory (MultiCube)
  Modelled on hippocampal-neocortical memory transfer.
  No backpropagation. No retraining. Hebbian updates only.

Author: Ivelin Likov
"""

import torch
import torch.nn.functional as F
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube_core_torch import SpatialCubeTorch
from sage_divided import SAGEDivided

# =================================================================
# SAGE PHYSICAL FORCES
# =================================================================
# Force 1 - Semantic Attraction (alpha): cube-size independent
# Force 2 - Spatial Cohesion / Gravity (beta): scales as n^3
#
#   G = G_BASE * (cube_size / 32)^3
#     16^3 -> G = 0.000375  (drone default)
#     32^3 -> G = 0.003000  (reference)
# =================================================================

ALPHA_BASE    = 0.01
ALPHA_STORE   = 0.01
ALPHA_ACTION  = 0.025
ALPHA_PRELOAD = 0.05
ALPHA_LONG    = 0.015   # consolidation (long-term write, gentler)
G_BASE        = 0.003

def get_gravity(cube_size: int) -> float:
    """Return scaled gravity constant for the given cube size."""
    return G_BASE * (cube_size / 32.0) ** 3




class SAGEMemoryV2:
    """
    Full SAGE memory system for drone agents.

    Two layers:
      1. SAGEDivided  -- short-term working memory
                         splits obs into subject (x<0) / object (x>=0)
                         updated every step

      2. MultiCube    -- long-term persistent memory
                         four specialist cubes, each 48MB
                         updated via consolidation after each step

    Decision fallback hierarchy:
      LLM online  -> SAGE context + LLM reasons -> store action
      LLM offline -> action_cube recall via 768D cosine -> replay
      No memory   -> hold position (safe default)
    """

    def __init__(self, cube_size=16, embed_dim=768, device=None,
                 save_dir='./sage_drone_state_v2'):

        self.embed_dim = embed_dim
        self.save_dir  = save_dir
        self.device    = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = get_gravity(cube_size)    # Force 2: gravity (cube-size scaled)
        self.A = ALPHA_BASE                # Force 1: semantic attraction

        print(f"\n{'='*55}")
        print(f"  SAGE DRONE MEMORY v2")
        print(f"  SAGEDivided + MultiCube + Consolidation")
        print(f"{'='*55}")
        print(f"  Device:    {self.device}")
        print(f"  Cube size: {cube_size}^3 = {cube_size**3:,} pts per cube")
        print(f"  Layer 1:   SAGEDivided (short-term working memory)")
        print(f"  Layer 2:   MultiCube x4 (long-term persistent memory)")
        print(f"{'='*55}")
        print_forces(cube_size)
        print(f"{'='*55}\n")

        # Layer 1: SAGEDivided -- short-term
        self.working_memory = SAGEDivided(
            cube_size=cube_size,
            embed_dim=embed_dim,
            device=self.device,
            seed=11
        )

        # Layer 2: MultiCube -- long-term
        self.nav_cube     = SpatialCubeTorch(cube_size=cube_size, embed_dim=embed_dim, seed=42,  device=self.device)
        self.object_cube  = SpatialCubeTorch(cube_size=cube_size, embed_dim=embed_dim, seed=123, device=self.device)
        self.mission_cube = SpatialCubeTorch(cube_size=cube_size, embed_dim=embed_dim, seed=999, device=self.device)
        self.action_cube  = SpatialCubeTorch(cube_size=cube_size, embed_dim=embed_dim, seed=777, device=self.device)

        self.cubes = {
            'nav':     self.nav_cube,
            'objects': self.object_cube,
            'mission': self.mission_cube,
            'actions': self.action_cube,
        }

        # Full action text index
        self.action_index = {}

        # Consolidation log -- tracks what moved from short to long term
        self.consolidation_log = []

        self.flight_log = []
        self.session_start = datetime.now().isoformat()
        self.observations_count = 0

        print("SAGE Memory v2 ready.\n")

    # ─────────────────────────────────────────
    # Main pipeline: observe -> consolidate
    # ─────────────────────────────────────────

    def observe(self, observation_vec, action_vec=None,
                obs_label='', action_label='', alpha=0.03):
        """
        Step 1: Write observation into SAGEDivided (working memory).

        observation_vec -> subject half
        action_vec      -> object half

        This is the SHORT-TERM write. Call this every step.
        """
        s_idx, o_idx = self.working_memory.encode(
            observation_vec, action_vec, alpha=alpha
        )
        self.working_memory.label_current(obs_label, action_label)
        self.observations_count += 1
        return s_idx, o_idx

    def consolidate(self, alpha_long=0.015):
        """
        Step 2: Transfer working memory into MultiCube (long-term).

        This is the CONSOLIDATION PATHWAY.
        Modelled on hippocampal-neocortical memory transfer.

        Called after each agent step -- continuous consolidation,
        not batch. No backpropagation. Pure Hebbian.
        """
        ctx = self.working_memory.get_working_context()
        if ctx['subject_vec'] is None:
            return

        obs_vec = ctx['subject_vec']
        act_vec = ctx['object_vec']

        # Push observation into mission cube
        self.mission_cube.learn_association(obs_vec, obs_vec, alpha=alpha_long, beta=self.G)

        # Push action into action cube (observation->action pair)
        if act_vec is not None:
            self.action_cube.learn_association(obs_vec, act_vec, alpha=alpha_long, beta=self.G)

        self.consolidation_log.append({
            'step':   ctx['step'],
            'subject': ctx['subject_label'],
            'object':  ctx['object_label'],
        })

    def full_step(self, observation_vec, action_vec=None,
                  obs_label='', action_label='', alpha=0.03):
        """
        Combined observe + consolidate in one call.
        Use this in the agent loop for convenience.
        """
        s_idx, o_idx = self.observe(
            observation_vec, action_vec,
            obs_label, action_label, alpha
        )
        self.consolidate()
        return s_idx, o_idx

    # ─────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────

    def store(self, embedding, label, cube_name='mission', learn=True, alpha=0.01):
        """Store a knowledge embedding in a named long-term cube."""
        cube = self.cubes.get(cube_name)
        if cube is None:
            raise ValueError(f"Unknown cube: {cube_name}")
        vec = self._to_tensor(embedding)
        idx = cube.label_point(vec, label)
        if learn:
            cube.learn_association(vec, vec, alpha=alpha, beta=self.G)
        self.flight_log.append({
            'time': datetime.now().isoformat(),
            'cube': cube_name, 'label': label,
        })
        return idx

    def store_action(self, obs_embedding, action_text,
                     action_embedding=None, alpha=0.03):
        """Store obs->action pair directly into action_cube."""
        obs_vec = self._to_tensor(obs_embedding)
        tgt_vec = self._to_tensor(action_embedding) if action_embedding is not None else obs_vec
        idx = self.action_cube.label_point(obs_vec, action_text[:40])
        self.action_index[idx] = action_text
        self.action_cube.learn_association(obs_vec, tgt_vec, alpha=alpha, beta=self.G)
        return idx

    def recall_action(self, observation_vec, top_k=3):
        """
        Recall best matching past action from action_cube.
        Pure SAGE agency -- 768D cosine, no LLM needed.
        """
        vec = self._to_tensor(observation_vec)
        result = self.action_cube.query(vec, top_k=top_k)
        recalled = []
        for i in range(top_k):
            idx   = result['indices'][i].item()
            score = result['scores'][i].item()
            text  = self.action_index.get(
                idx, result['labels'][i].replace('_', ' ')
            )
            if text.startswith('point_'):
                continue
            recalled.append({
                'action':     text,
                'score':      score,
                'confidence': 'high'   if score > 0.01  else
                              'medium' if score > 0.003 else 'low',
            })
        return recalled

    def query_working_memory(self, query_vec, top_k=3):
        """
        Query SAGEDivided working memory.
        Returns subject and object candidates separately.
        """
        vec = self._to_tensor(query_vec)
        subjects = self.working_memory.query_subject(vec, top_k=top_k)
        objects  = self.working_memory.query_object(vec,  top_k=top_k)
        return {'subjects': subjects, 'objects': objects}

    def query_all_cubes(self, embedding, top_k=3):
        """Query all three knowledge long-term cubes."""
        vec = self._to_tensor(embedding)
        combined = {}
        for name in ['nav', 'objects', 'mission']:
            result = self.cubes[name].query(vec, top_k=top_k)
            combined[name] = [{
                'label': result['labels'][i],
                'score': result['scores'][i].item(),
            } for i in range(top_k)]
        return combined

    def get_context_string(self, embedding, top_k=3):
        """
        Build full context string for LLM injection.
        Includes working memory + long-term knowledge + past actions.
        """
        vec = self._to_tensor(embedding)

        # Working memory context
        wm = self.query_working_memory(vec, top_k=2)
        # Long-term context
        lt = self.query_all_cubes(vec, top_k=top_k)
        # Action recall
        actions = self.recall_action(vec, top_k=2)

        lines = ['[SAGE MEMORY CONTEXT v2]']

        # Working memory
        wm_labels = []
        for r in wm['subjects']['labels'] + wm['objects']['labels']:
            if not r.startswith('point_'):
                wm_labels.append(r)
        if wm_labels:
            lines.append(f"  WORKING MEMORY: {', '.join(wm_labels[:3])}")

        # Long-term cubes
        for cube_name, results in lt.items():
            labels = [
                r['label'] for r in results
                if not r['label'].startswith('point_') and r['score'] > 0.001
            ]
            if labels:
                lines.append(f"  {cube_name.upper()}: {', '.join(labels)}")

        # Past actions
        if actions:
            lines.append('  PAST ACTIONS IN SIMILAR SITUATIONS:')
            for pa in actions:
                lines.append(f"    [{pa['confidence']}] {pa['action']}")

        if len(lines) == 1:
            lines.append('  No prior relevant memories found.')

        lines.append('[END CONTEXT]')
        return '\n'.join(lines)

    # ─────────────────────────────────────────
    # Pre-load
    # ─────────────────────────────────────────

    def preload_drone_knowledge(self, embedder_fn):
        """
        Pre-populate long-term cubes with baseline drone knowledge.
        Also pre-seeds action_cube with known good decisions.
        """
        print('\nPre-loading drone knowledge (v2)...')

        knowledge = {
            'nav': [
                'return to home base immediately',
                'fly to waypoint alpha at 50 metres altitude',
                'hold position and hover',
                'land at designated landing zone',
                'ascend to safe altitude above obstacles',
                'descend slowly to landing approach',
                'navigate around obstacle on the left',
                'emergency landing procedure activated',
            ],
            'objects': [
                'tree detected ahead',
                'building structure blocking path',
                'person detected below drone',
                'vehicle on ground',
                'power lines detected',
                'water body detected',
                'open field clear for landing',
                'bird in flight path',
                'drone battery low warning',
            ],
            'mission': [
                'primary mission objective is aerial survey',
                'do not fly over restricted airspace',
                'maintain visual line of sight',
                'wind speed exceeds safe operating threshold',
                'mission complete return to base',
                'begin search pattern grid alpha',
                'connection to ground station lost',
                'GPS signal degraded use visual navigation',
                'all systems nominal proceed with mission',
            ],
        }

        seeded_actions = [
            ('tree detected ahead',
             'Reduce speed, alter heading 30 degrees right to avoid obstacle. Resume course when clear.'),
            ('battery level at 20 percent',
             'Abort current task. Return to home base immediately on most direct route.'),
            ('GPS signal degraded accuracy reduced',
             'Reduce speed. Switch to visual navigation. Hold current altitude. Attempt GPS reacquisition.'),
            ('person detected below drone',
             'Increase altitude to 50 metres. Move laterally 20 metres away from person. Continue when clear.'),
            ('strong wind detected drone drifting',
             'Reduce altitude to 20 metres. Compensate heading into wind. Reduce speed to minimum stable.'),
            ('power lines detected crossing flight path',
             'Stop immediately. Ascend 25 metres above power lines before proceeding. Log hazard location.'),
            ('connection to ground station lost',
             'Continue mission autonomously. Use SAGE memory for decisions. Attempt reconnect every 30 seconds.'),
            ('low visibility fog reducing camera range',
             'Reduce speed to 2 metres per second. Activate proximity sensors. Descend below fog layer if safe.'),
            ('mission area survey complete',
             'Return to home base at cruise altitude. Log mission complete. Prepare for landing sequence.'),
            ('all systems nominal proceed with mission',
             'Continue current mission at planned speed and altitude. All systems green.'),
            ('obstacle detected on current path',
             'Stop. Assess obstacle size. Navigate around using 3D avoidance. Resume mission after clear.'),
            ('emergency landing procedure activated',
             'Find nearest flat open ground. Descend at safe rate. Cut motors at 0.5 metre height.'),
        ]

        counts = {k: 0 for k in ['nav', 'objects', 'mission', 'actions', 'working']}

        for cube_name, items in knowledge.items():
            for text in items:
                try:
                    vec = embedder_fn(text)
                    self.store(vec, text.replace(' ', '_')[:30], cube_name)
                    counts[cube_name] += 1
                    print(f"  [{cube_name:8s}] [OK] {text[:50]}")
                except Exception as e:
                    print(f"  [{cube_name:8s}] [FAIL] {e}")

        print(f'\n  [actions] Pre-seeding autonomous decision baseline...')
        for obs_text, action_text in seeded_actions:
            try:
                obs_vec = embedder_fn(obs_text)
                act_vec = embedder_fn(action_text)
                # Store in action_cube (long-term)
                self.store_action(obs_vec, action_text, act_vec, alpha=ALPHA_PRELOAD)
                # Also encode into SAGEDivided (working memory seed)
                self.working_memory.encode(obs_vec, act_vec, alpha=ALPHA_PRELOAD)
                counts['actions'] += 1
                counts['working'] += 1
                print(f"  [actions ] [OK] '{obs_text[:35]}'")
                print(f"              -> '{action_text[:55]}'")
            except Exception as e:
                print(f"  [actions ] [FAIL] {e}")

        print(f"\nPre-load complete: " + " | ".join(f"{k}={v}" for k, v in counts.items()))
        print()
        return counts

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)

        # Save long-term cubes
        for name, cube in self.cubes.items():
            cube.save(f'{self.save_dir}/{name}')

        # Save action index
        with open(f'{self.save_dir}/action_index.json', 'w') as f:
            json.dump({str(k): v for k, v in self.action_index.items()}, f, indent=2)

        # Save flight log
        with open(f'{self.save_dir}/flight_log.json', 'w') as f:
            json.dump({
                'session_start': self.session_start,
                'observations':  self.observations_count,
                'consolidations': len(self.consolidation_log),
                'log': self.flight_log[-500:],
            }, f, indent=2)

        print(f'\nSAGE v2 saved -> {self.save_dir}')
        print(f'  Action memories:  {len(self.action_index)}')
        print(f'  Consolidations:   {len(self.consolidation_log)}')
        print(f'  Observations:     {self.observations_count}')

    def load(self):
        if not os.path.exists(self.save_dir):
            print(f'No saved state at {self.save_dir}. Starting fresh.')
            return False

        for name in self.cubes:
            path = f'{self.save_dir}/{name}'
            if os.path.exists(path):
                self.cubes[name] = SpatialCubeTorch.load(path, device=self.device)
                print(f'  [OK] {name} cube loaded')

        action_path = f'{self.save_dir}/action_index.json'
        if os.path.exists(action_path):
            with open(action_path) as f:
                self.action_index = {int(k): v for k, v in json.load(f).items()}
            print(f'  [OK] action index: {len(self.action_index)} entries')

        log_path = f'{self.save_dir}/flight_log.json'
        if os.path.exists(log_path):
            with open(log_path) as f:
                data = json.load(f)
                self.flight_log         = data.get('log', [])
                self.observations_count = data.get('observations', 0)

        print(f'\nSAGE v2 state restored.\n')
        return True

    # ─────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────

    def _to_tensor(self, vec):
        if isinstance(vec, torch.Tensor):
            return vec.to(self.device).float()
        return torch.tensor(vec, device=self.device, dtype=torch.float32)

    def stats(self):
        print(f"\n{'='*55}")
        print(f"  SAGE DRONE MEMORY v2 - STATS")
        print(f"{'='*55}")
        print(f"  Session start:   {self.session_start}")
        print(f"  Observations:    {self.observations_count}")
        print(f"  Action memories: {len(self.action_index)}")
        print(f"  Consolidations:  {len(self.consolidation_log)}")
        print(f"  Working memory:")
        self.working_memory.stats()
        print(f"  Long-term cubes:")
        for name, cube in self.cubes.items():
            print(f"    {name:10s}: {len(cube.labels)} labelled, "
                  f"{cube.step_count} learn steps")
        print(f"{'='*55}\n")
