"""
SAGE Memory - Drone Edition
============================
SpatialCubeTorch wrapper for local drone AI agent.
Stores all mission knowledge, observations and adaptations
as geometric positions in a 3D cube.

No cloud. No connection needed. Runs fully on-device.

KEY FEATURE - ACTION MEMORY:
  A fourth specialist cube stores (observation -> action) pairs.
  When the LLM is offline, SAGE queries this cube and replays
  the action taken last time a similar situation was seen.
  The drone never stops. It degrades to geometric memory-based
  agency rather than failing.

Architecture:
  - nav_cube:     waypoints, routes, spatial landmarks
  - object_cube:  obstacle types, terrain, objects seen
  - mission_cube: tasks, goals, rules of engagement
  - action_cube:  observation -> action pairs  ← autonomous fallback

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

# =================================================================
# SAGE PHYSICAL FORCES
# =================================================================
#
# Two forces govern learning in every SpatialCubeTorch cube.
#
# Force 1 - Semantic Attraction (alpha):
#   Pulls activated point embeddings toward the target vector.
#   Applied every step to all top-k activated points.
#   Weighted by softmax attention score.
#   Does NOT need cube-size scaling -- softmax temperature
#   stabilises activation scores regardless of n_points.
#
# Force 2 - Spatial Cohesion / Gravity (beta):
#   Pulls 3D-nearby points together if semantically similar (cos>0.3).
#   Applied every 5 steps for efficiency.
#   MUST be scaled by cube size -- neighbourhood count grows as n^3.
#
#   Scaling formula:  G = G_BASE * (cube_size / 32)^3
#   Neighbourhood counts at radius=0.35:
#     8^3  cube: ~14   neighbours -> G = 0.000047
#     16^3 cube: ~144  neighbours -> G = 0.000375  (drone default)
#     32^3 cube: ~1277 neighbours -> G = 0.003000  (reference)
#     64^3 cube: ~10720 neighbours -> G = 0.024000
# =================================================================

ALPHA_BASE   = 0.01    # semantic attraction -- cube-size independent
ALPHA_STORE  = 0.01    # storing knowledge entries
ALPHA_ACTION = 0.025   # storing obs->action pairs (stronger signal)
ALPHA_PRELOAD = 0.05   # pre-seeding baseline knowledge
G_BASE       = 0.003   # gravity reference for 32^3 cube

def get_gravity(cube_size: int) -> float:
    """Return scaled gravity constant for the given cube size."""
    return G_BASE * (cube_size / 32.0) ** 3




class SAGEMemory:

    def __init__(self, cube_size=16, embed_dim=768, device=None,
                 save_dir="./sage_drone_state"):

        self.embed_dim = embed_dim
        self.save_dir = save_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Physical forces scaled for this cube size
        self.G = get_gravity(cube_size)    # gravity (Force 2, scaled)
        self.A = ALPHA_BASE                # semantic attraction (Force 1)

        print(f"\n{'='*55}")
        print(f"  SAGE DRONE MEMORY SYSTEM")
        print(f"{'='*55}")
        print(f"  Device:    {self.device}")
        print(f"  Cube size: {cube_size}^3 = {cube_size**3:,} points per cube")
        print(f"  Cubes:     nav | objects | mission | actions")
        print(f"{'='*55}")
        print_forces(cube_size)
        print(f"{'='*55}\n")

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

        # Full action text index - labels are truncated to 40 chars,
        # this stores the complete action string keyed by point index.
        self.action_index = {}

        self.flight_log = []
        self.session_start = datetime.now().isoformat()
        self.observations_count = 0

        print("SAGE Memory ready.\n")

    # -----------------------------------------
    # Knowledge storage
    # -----------------------------------------

    def store(self, embedding, label, cube_name='mission', learn=True, alpha=0.01):
        """Store a knowledge embedding in the named cube."""
        cube = self.cubes.get(cube_name)
        if cube is None:
            raise ValueError(f"Unknown cube: {cube_name}")

        vec = self._to_tensor(embedding)
        idx = cube.label_point(vec, label)
        if learn:
            cube.learn_association(vec, vec, alpha=alpha, beta=self.G)

        self.observations_count += 1
        self.flight_log.append({
            'time': datetime.now().isoformat(),
            'cube': cube_name,
            'label': label,
        })
        return idx

    # -----------------------------------------
    # Action memory - the autonomous fallback
    # -----------------------------------------

    def store_action(self, observation_embedding, action_text,
                     action_embedding=None, alpha=0.03):
        """
        Store an (observation -> action) pair in the action cube.

        This is what lets SAGE act as a standalone agent when the LLM
        is offline. Each time the LLM decides an action, we call this.
        Next time the drone sees a similar situation, recall_action()
        retrieves what was done before.

        Args:
            observation_embedding: 768-dim vector of what was perceived
            action_text:           the action that was decided (full text)
            action_embedding:      768-dim vector of action text (optional)
                                   If provided, learns obs->action mapping.
                                   If None, stores obs->obs (identity).
            alpha:                 learning rate (higher overwrites old faster)
        """
        obs_vec = self._to_tensor(observation_embedding)
        target_vec = self._to_tensor(action_embedding) if action_embedding is not None else obs_vec

        # Label the point with truncated action text
        idx = self.action_cube.label_point(obs_vec, action_text[:40])

        # Store full text in index
        self.action_index[idx] = action_text

        # Force 1 (attraction) + Force 2 (gravity) -- both scaled
        self.action_cube.learn_association(obs_vec, target_vec, alpha=alpha, beta=self.G)

        return idx

    def recall_action(self, observation_embedding, top_k=3):
        """
        Given a current observation, recall the most relevant past actions.

        Pure SAGE agency - no LLM, no connection needed.
        The cube geometry does the reasoning: nearest neighbour in
        observation space = most similar past situation = best action to replay.

        Returns:
            list of dicts: {action, score, confidence}
        """
        vec = self._to_tensor(observation_embedding)
        result = self.action_cube.query(vec, top_k=top_k)

        recalled = []
        for i in range(top_k):
            idx   = result['indices'][i].item()
            score = result['scores'][i].item()

            # Full text from index, or fall back to truncated label
            action_text = self.action_index.get(
                idx, result['labels'][i].replace('_', ' ')
            )

            if action_text.startswith('point_'):
                continue  # uninitialised point, skip

            recalled.append({
                'action':     action_text,
                'score':      score,
                'confidence': 'high'   if score > 0.01  else
                              'medium' if score > 0.003 else 'low',
            })

        return recalled

    # -----------------------------------------
    # Queries
    # -----------------------------------------

    def query(self, embedding, cube_name='mission', top_k=5):
        cube = self.cubes.get(cube_name)
        if cube is None:
            raise ValueError(f"Unknown cube: {cube_name}")
        vec = self._to_tensor(embedding)
        result = cube.query(vec, top_k=top_k)
        return [{
            'label':    result['labels'][i],
            'score':    result['scores'][i].item(),
            'position': result['positions'][i].cpu().tolist(),
            'index':    result['indices'][i].item(),
        } for i in range(top_k)]

    def query_all_cubes(self, embedding, top_k=3):
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
        Build a formatted context string for LLM prompt injection.
        Includes knowledge memories + past action memories.
        """
        all_results  = self.query_all_cubes(embedding, top_k=top_k)
        past_actions = self.recall_action(embedding, top_k=2)

        lines = ["[SAGE MEMORY CONTEXT]"]

        for cube_name, results in all_results.items():
            labels = [
                r['label'] for r in results
                if not r['label'].startswith('point_') and r['score'] > 0.001
            ]
            if labels:
                lines.append(f"  {cube_name.upper()}: {', '.join(labels)}")

        if past_actions:
            lines.append("  PAST ACTIONS IN SIMILAR SITUATIONS:")
            for pa in past_actions:
                lines.append(f"    [{pa['confidence']}] {pa['action']}")

        if len(lines) == 1:
            lines.append("  No prior relevant memories found.")

        lines.append("[END CONTEXT]")
        return "\n".join(lines)

    # -----------------------------------------
    # Pre-load baseline drone knowledge
    # -----------------------------------------

    def preload_drone_knowledge(self, embedder_fn):
        """
        Pre-populate all four cubes with baseline drone knowledge.
        Critically, pre-seeds the action_cube with known good decisions
        so SAGE can act autonomously from the very first flight.
        """
        print("\nPre-loading drone knowledge...")

        knowledge = {
            'nav': [
                "return to home base immediately",
                "fly to waypoint alpha at 50 metres altitude",
                "hold position and hover",
                "land at designated landing zone",
                "ascend to safe altitude above obstacles",
                "descend slowly to landing approach",
                "navigate around obstacle on the left",
                "emergency landing procedure activated",
            ],
            'objects': [
                "tree detected ahead",
                "building structure blocking path",
                "person detected below drone",
                "vehicle on ground",
                "power lines detected",
                "water body detected",
                "open field clear for landing",
                "bird in flight path",
                "drone battery low warning",
            ],
            'mission': [
                "primary mission objective is aerial survey",
                "do not fly over restricted airspace",
                "maintain visual line of sight",
                "wind speed exceeds safe operating threshold",
                "mission complete return to base",
                "begin search pattern grid alpha",
                "connection to ground station lost",
                "GPS signal degraded use visual navigation",
                "all systems nominal proceed with mission",
            ],
        }

        # Pre-seeded (observation -> action) pairs
        # These give SAGE an immediate autonomous capability
        seeded_actions = [
            ("tree detected ahead",
             "Reduce speed, alter heading 30 degrees right to avoid obstacle. Resume course when clear."),
            ("battery level at 20 percent",
             "Abort current task. Return to home base immediately on most direct route."),
            ("GPS signal degraded accuracy reduced",
             "Reduce speed. Switch to visual navigation. Hold current altitude. Attempt GPS reacquisition."),
            ("person detected below drone",
             "Increase altitude to 50 metres. Move laterally 20 metres away from person. Continue when clear."),
            ("strong wind detected drone drifting",
             "Reduce altitude to 20 metres. Compensate heading into wind. Reduce speed to minimum stable."),
            ("power lines detected crossing flight path",
             "Stop immediately. Ascend 25 metres above power lines before proceeding. Log hazard location."),
            ("connection to ground station lost",
             "Continue mission autonomously. Use SAGE memory for decisions. Attempt reconnect every 30 seconds."),
            ("low visibility fog reducing camera range",
             "Reduce speed to 2 metres per second. Activate proximity sensors. Descend below fog layer if safe."),
            ("mission area survey complete",
             "Return to home base at cruise altitude. Log mission complete. Prepare for landing sequence."),
            ("all systems nominal proceed with mission",
             "Continue current mission at planned speed and altitude. All systems green."),
            ("obstacle detected on current path",
             "Stop. Assess obstacle size. Navigate around using 3D avoidance. Resume mission after clear."),
            ("emergency landing procedure activated",
             "Find nearest flat open ground. Descend at safe rate. Cut motors at 0.5 metre height."),
        ]

        counts = {k: 0 for k in ['nav', 'objects', 'mission', 'actions']}

        for cube_name, items in knowledge.items():
            for text in items:
                try:
                    vec = embedder_fn(text)
                    self.store(vec, text.replace(' ', '_')[:30], cube_name)
                    counts[cube_name] += 1
                    print(f"  [{cube_name:8s}] [OK] {text[:50]}")
                except Exception as e:
                    print(f"  [{cube_name:8s}] [FAIL] {text[:40]}: {e}")

        print(f"\n  [actions] Pre-seeding autonomous decision baseline...")
        for obs_text, action_text in seeded_actions:
            try:
                obs_vec = embedder_fn(obs_text)
                act_vec = embedder_fn(action_text)
                self.store_action(obs_vec, action_text, act_vec, alpha=ALPHA_PRELOAD)
                counts['actions'] += 1
                print(f"  [actions ] [OK] '{obs_text[:35]}'")
                print(f"             -> '{action_text[:55]}'")
            except Exception as e:
                print(f"  [actions ] [FAIL] {e}")

        print(f"\nPre-load complete: " + " | ".join(f"{k}={v}" for k, v in counts.items()))
        print()
        return counts

    # -----------------------------------------
    # Persistence
    # -----------------------------------------

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        for name, cube in self.cubes.items():
            cube.save(f"{self.save_dir}/{name}")

        with open(f"{self.save_dir}/action_index.json", 'w') as f:
            json.dump({str(k): v for k, v in self.action_index.items()}, f, indent=2)

        with open(f"{self.save_dir}/flight_log.json", 'w') as f:
            json.dump({
                'session_start': self.session_start,
                'observations':  self.observations_count,
                'log':           self.flight_log[-500:],
            }, f, indent=2)

        print(f"\nSAGE saved -> {self.save_dir}")
        print(f"  Action memories: {len(self.action_index)}")
        print(f"  This session:    {self.observations_count} observations")

    def load(self):
        if not os.path.exists(self.save_dir):
            print(f"No saved state at {self.save_dir}. Starting fresh.")
            return False

        for name in self.cubes:
            path = f"{self.save_dir}/{name}"
            if os.path.exists(path):
                self.cubes[name] = SpatialCubeTorch.load(path, device=self.device)
                print(f"  [OK] {name} cube loaded")

        action_path = f"{self.save_dir}/action_index.json"
        if os.path.exists(action_path):
            with open(action_path, 'r') as f:
                self.action_index = {int(k): v for k, v in json.load(f).items()}
            print(f"  [OK] action index: {len(self.action_index)} entries")

        log_path = f"{self.save_dir}/flight_log.json"
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                data = json.load(f)
                self.flight_log          = data.get('log', [])
                self.observations_count  = data.get('observations', 0)

        print(f"\nSAGE state restored.\n")
        return True

    def _to_tensor(self, vec):
        if isinstance(vec, torch.Tensor):
            return vec.to(self.device).float()
        return torch.tensor(vec, device=self.device, dtype=torch.float32)

    def stats(self):
        print(f"\n{'='*55}")
        print(f"  SAGE DRONE MEMORY - STATS")
        print(f"{'='*55}")
        print(f"  Session start:   {self.session_start}")
        print(f"  Observations:    {self.observations_count}")
        print(f"  Action memories: {len(self.action_index)}")
        for name, cube in self.cubes.items():
            print(f"  {name:10s}: {len(cube.labels)} labelled pts, "
                  f"{cube.step_count} learn steps")
        print(f"{'='*55}\n")
