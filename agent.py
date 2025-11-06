import networkx as nx
from agent_baselines import Agent
import random
import time
import math
from game import copy_game
from collections import deque

#  IMPROVED CONVOYS

class StudentAgent(Agent):

    # @timeout_decorator.timeout(1)
    def __init__(self, agent_name='Deamon of Fate'):
        super().__init__(agent_name)

        # connection graphs not state space
        self.map_graph_army = None 
        self.map_graph_navy = None 
        self.army_paths = {}
        self.navy_paths = {}
        self.hostile_powers = set()
        self.hostile_targets = set()

        self.ROOT_MAX_ACTIONS = 12
        self.UCT_C = 1.1
        self.TIME_BUDGET_S = 0.95
        self.SAFETY_MARGIN = 0.05

        self.no_progress_moves = 0        # consecutive movement phases with no SC change
        self.PROGRESS_THRESHOLD = 5       #
        
    # @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        self.build_map_graphs()
        self.precompute_paths()

    # @timeout_decorator.timeout(1) 
    def update_game(self, all_power_orders):
        # record pre-state (centers and unit locations) so we can spot attacks/captures
        pre_my_centers = set(self.game.get_centers(self.power_name))
        pre_my_unit_locs = {u.split()[1].upper() for u in self.game.get_units(self.power_name)}
        was_movement = (self.game.phase_type == 'M')  # only count movement phases

        # inspect incoming orders to detect direct threats BEFORE applying them
        for p_name, orders in all_power_orders.items():
            if p_name == self.power_name:
                continue
            # check each order string for moves that target our SCs or unit locs
            for ord_str in orders:
                parts = ord_str.split()
                if len(parts) >= 4 and parts[2] == '-' and 'VIA' not in ord_str:
                    dest = parts[-1].upper()
                    if dest in pre_my_centers or dest in pre_my_unit_locs:
                        self.hostile_powers.add(p_name)

        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

        # after adjudication, check for centers we lost (captures)
        post_my_centers = set(self.game.get_centers(self.power_name))
        lost = pre_my_centers - post_my_centers
        gained = post_my_centers - pre_my_centers
        if lost:
            # for each lost SC find who owns it now and mark them hostile
            for sc in lost:
                for p in self.game.powers.keys():
                    if p == self.power_name:
                        continue
                    if sc in set(self.game.get_centers(p)):
                        self.hostile_powers.add(p)

        # progress meter: only count movement phases
        if was_movement:
            if lost or gained:
                self.no_progress_moves = 0
            else:
                self.no_progress_moves += 1

    class Node:
        __slots__ = ("action", "N", "Q") # faster than storing in dict; stores in array instead
        def __init__(self, action):
            self.action = action  # one root child = one order combinations list (list[str])
            self.N = 0  # visit count
            self.Q = 0.0  # total value (sum of rollout evaluations)
        
    # @timeout_decorator.timeout(1)
    def get_actions(self):
        start = time.perf_counter()

        # build small set of candidate order combinations for us only
        candidates = self.get_mcts_root_actions(self.ROOT_MAX_ACTIONS)
        if not candidates:
            return []
        
        # progressive widening 
        pending = list(candidates)  
        random.shuffle(pending)
        nodes = []
        totalN = 0  # total simulation runs

        widening_thresholds = {0, 1, 3, 7, 15}

        # best so far in case we get cut off mid loop
        best_action = pending[0]

        def uct_value(node, total_visits):
            if node.N == 0:  
                return float("inf")
            return (node.Q / node.N) + self.UCT_C * math.sqrt(math.log(max(1, total_visits)) / node.N)
        
        # MCTS loop under time budget
        while True:
            if time.perf_counter() - start > self.TIME_BUDGET_S - self.SAFETY_MARGIN:
                break

            # progressive widening: add a new child at certain visit counts, clamp to ROOT_MAX_ACTIONS
            if pending and (totalN in widening_thresholds or not nodes) and len(nodes) < self.ROOT_MAX_ACTIONS:
                act = pending.pop()
                nodes.append(self.Node(action=act))

            if not nodes:
                break

            # SELECTION for root only: pick child with max UCT 
            chosen = max(nodes, key=lambda n: uct_value(n, max(1, totalN)))

            # SIMULATION: run one rollout from this action 
            value = self.simulate_value(chosen.action)

            # BACKPROPAGATION 
            chosen.N += 1
            chosen.Q += value
            totalN += 1

            # Track best by mean value so we can return immediately if time expires
            best_mean = -1e18
            best = best_action
            for n in nodes:
                if n.N > 0:
                    mean_value = n.Q / n.N
                    if mean_value > best_mean:
                        best_mean = mean_value
                        best = n.action
            best_action = best

        return best_action

    # ---------------------------- root action construction ----------------------------------
    def get_mcts_root_actions(self, cap):
        if self.game.phase_type != 'M':
            # Non-movement phases (builds/retreats/disbands): bias to constructive choices
            poss = self.game.get_all_possible_orders()
            locs = self.game.get_orderable_locations(self.power_name)
            joint = []
            for loc in locs:
                opts = poss.get(loc, [])
                if not opts:
                    continue
                build = [o for o in opts if o.endswith(' B')]
                retreat = [o for o in opts if ' R ' in o]
                hold = [o for o in opts if o.endswith(' H')]
                choice_pool = (build or retreat or hold or opts)
                joint.append(random.choice(choice_pool))
            return [joint]  
        
        # movement phase
        possible_orders = self.game.get_all_possible_orders()
        locs = self.game.get_orderable_locations(self.power_name)
        my_scs = set(self.game.get_centers(self.power_name))
        all_scs = set(self.game.map.scs)
        self.is_eng = (self.power_name == "ENGLAND")

        other_scs = set().union(*(set(self.game.get_centers(p))
                                  for p in self.game.powers if p != self.power_name))
        targets = list((all_scs - my_scs) - other_scs)

        phase = self.game.get_current_phase()  
        year = int(''.join(filter(str.isdigit, phase)) or 0)

        # only attack powers that are hostile; if none, pick a neighbor/nearest
        if year >= 1903:
            # prune hostiles that have no SCs left
            for p in list(self.hostile_powers): 
                if len(self.game.get_centers(p)) == 0:
                    self.hostile_powers.remove(p)

            if self.hostile_powers:
                # if we're stuck against a single enemy, attack some one else
                if len(self.hostile_powers) == 1 and self.no_progress_moves >= self.PROGRESS_THRESHOLD:
                    exclude = next(iter(self.hostile_powers))
                    alt = self.bfs_nearest_enemy(exclude_power=exclude)
                    if alt:
                        self.hostile_powers.add(alt)
                for p in self.hostile_powers:
                    targets += self.game.get_centers(p)
            else:
                if self.power_name == "ENGLAND" and len(self.game.get_centers("FRANCE")) > 0:
                    attack = "FRANCE"
                elif self.power_name == "AUSTRIA" and len(self.game.get_centers("AUSTRIA")) > 0:
                    attack = "RUSSIA"
                else:
                    attack = self.bfs_nearest_enemy()
                    if attack is None:
                        # deterministic-ish fallback
                        attack = random.choice([p for p in self.game.powers if p != self.power_name])
                if attack:
                    self.hostile_powers.add(attack)
                    targets += self.game.get_centers(attack)

        desired_convoy = None   # tuple: (origin, dest, via_string)
        if self.power_name == "ENGLAND":
            # cheap early exit search for a single A LON/EDI convoy to BEL/HOL
            for origin in ("LON", "EDI"):
                opts = possible_orders.get(origin, ())
                if not opts:
                    continue
                for dest in ("BEL", "HOL"):
                    via = f"A {origin} - {dest} VIA"
                    if via in opts:
                        desired_convoy = (origin, dest, via)
                        break
                if desired_convoy:
                    break

        per_unit_choices = []
        for loc in locs:
            options = possible_orders.get(loc, [])
            if not options:
                continue

            # 1: allow holding (from 1902 onward)
            hold = None
            if year >= 1902:
                for i in options:
                    if i.endswith(" H"):
                        hold = i
                        break
            
            # 2: step towards a nearest target SC (or convoy setup for England)
            unit_is_army = options[0].startswith("A ")
            step = None

            # convoy preference (England)
            if desired_convoy:
                origin, dest, via_str = desired_convoy
                if unit_is_army and loc.upper() == origin and via_str in options:
                    step = via_str
                elif not unit_is_army:
                    locU = loc.upper()
                    if locU in ("ENG", "NTH"):
                        if hold and not step:
                            step = hold
                    else:
                        mv_to_eng = f"F {locU} - ENG"
                        mv_to_nth = f"F {locU} - NTH"
                        if mv_to_eng in options and not step:
                            step = mv_to_eng
                        elif mv_to_nth in options and not step:
                            step = mv_to_nth

            # compute/take the nearest-SC step if we still haven't chosen one
            if not step:
                next_loc = self.next_hop_nearest_sc(loc, unit_is_army, targets)
                if next_loc:
                    prefix = "A " if unit_is_army else "F "
                    candidate = f"{prefix}{loc} - {next_loc}"
                    if candidate in options:
                        step = candidate
                    else:
                        # Coast variants
                        locU = loc.upper()
                        nxtU = next_loc.upper()
                        for i in options:
                            parts = i.split()
                            if len(parts) >= 4 and parts[1].upper() == locU and parts[2] == "-" and parts[-1].upper() == nxtU:
                                step = i
                                break

            if not step and self.is_eng and unit_is_army:
                # as last resort, allow VIA if engine listed it
                step = next((i for i in options if " VIA" in i), None)

            # final per-unit choice list max size 2
            choices = []
            if hold:
                choices.append(hold)
            if step and step != hold:
                choices.append(step)

            if not choices:
                # take order that moves the unit but isn't a convoy (unless England allows it)
                moves = [i for i in options if " - " in i and (self.is_eng or " VIA" not in i)]
                if moves:
                    choices = [random.choice(moves)]
                else:
                    choices = [random.choice(options)]
            per_unit_choices.append(choices)

        if not per_unit_choices:
            return []
        
        # Sample up to the cap for possible order combinations
        order_combinations = []
        seen = set()
        attempts = 0
        MAX_ATTEMPTS = min(cap * 6, 200)

        # Some randomness to avoid subtle bias
        random.shuffle(per_unit_choices)

        while len(order_combinations) < cap and attempts < MAX_ATTEMPTS:
            attempts += 1
            oc = [random.choice(choices) for choices in per_unit_choices]
            oc = self.create_support(oc)  # transform conflicts into supports where legal
            key = tuple(sorted(oc))
            if key not in seen:
                seen.add(key)
                order_combinations.append(oc)

        # fallback in case something unexpected happened
        if not order_combinations:
            first = [choices[0] for choices in per_unit_choices]
            order_combinations.append(self.create_support(first))

        return order_combinations
    
    def bfs_nearest_enemy(self, exclude_power=None):
        sc_owner = {}
        for p in self.game.powers:
            for sc in self.game.get_centers(p):
                sc_owner[sc.upper()] = p

        q = deque()
        visited = set()
        for s in self.game.get_centers(self.power_name):
            if s in self.map_graph_army:
                q.append(s)
                visited.add(s)

        while q:
            v = q.popleft()
            if v in sc_owner:
                owner = sc_owner[v]
                if owner != self.power_name and owner != exclude_power:
                    return owner
            for nbr in self.map_graph_army[v]:
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)
        return None
    
    def create_support(self, orders):
        # modified to group by destination so it allows 2+ units to support
        # using a dict to make sure the same unit doesn't have 2+ orders
        unit_orders = {}
        move_info = []
        for idx, order in enumerate(orders):
            parts = order.split()
            if len(parts) >= 2:
                key = (parts[0], parts[1])
                unit_orders[key] = order
            # pure move format: ["A"/"F", ORIGIN, "-", DEST] no " VIA" for no convoys
            if len(parts) >= 4 and parts[2] == "-" and " VIA" not in order:
                utype = parts[0]  # "A" or "F"
                origin = parts[1]  # loc
                dest = parts[-1]  # dest loc
                move_info.append((idx, utype, origin, dest))

        # Build index: origin -> (idx, utype, origin, dest) for our movers
        by_origin = {orig: tup for tup in move_info for _, _, orig, _ in [tup]}

        for idx, utype, origin, dest in move_info:
            # Is this move trying to enter the origin of another of OUR units that is moving?
            leader = by_origin.get(dest)
            if not leader:
                continue
            # leader is the unit currently at dest (its origin), moving to leaderdest
            _, leader_utype, leader_origin, leader_dest = leader
            # Support-to-move requires supporter adjacent to the DESTINATION being supported
            if self.game.map.abuts(utype, origin, "-", leader_dest):
                supporter_key = (utype, origin)
                unit_orders[supporter_key] = f"{utype} {origin} S {leader_utype} {leader_origin} - {leader_dest}"

        # group movers by their destination province
        by_dest = {}
        for tup in move_info:
            dest = tup[3]
            if dest not in by_dest:
                by_dest[dest] = []
            by_dest[dest].append(tup)

        # for each destination with >1 attacks pick a leader and convert others to support
        for dest, movers in by_dest.items():
            if len(movers) < 2:
                continue

            # leader is just the first one
            leader_idx, leader_utype, leader_origin, _ = movers[0]
            leader_unit = f"{leader_utype} {leader_origin}"

            # convert remaining movers into supports when adjacency allows
            for idx, utype, origin, _ in movers[1:]:
                if self.game.map.abuts(utype, origin, "-", dest):
                    supporter_key = (utype, origin)
                    unit_orders[supporter_key] = f"{utype} {origin} S {leader_unit} - {dest}"

        # England: force convoy orders only if they appear in the engine's possible orders;
        # ensure we don't overwrite multiple times per sea
        if self.power_name == 'ENGLAND':
            used_seas = set()
            possible_orders = self.game.get_all_possible_orders()
            for o in list(unit_orders.values()):
                p = o.split()
                # look for an army convoy move like "A LON - PIC VIA"
                if len(p) >= 5 and p[0] == 'A' and p[2] == '-' and p[-1] == 'VIA':
                    orig, dest = p[1].upper(), p[-2].upper()
                    # for each of our fleets, check if the engine lists a matching convoy order
                    for u in self.game.get_units(self.power_name):
                        if u.startswith('F '):
                            sea = u.split()[1].upper()
                            if sea in used_seas:
                                continue
                            legal = possible_orders.get(sea, [])
                            convoy_str = f"F {sea} C A {orig} - {dest}"
                            if convoy_str in legal:
                                unit_orders[('F', sea)] = convoy_str
                                used_seas.add(sea)

        # convert dict back to list
        return list(unit_orders.values())
    
    # -------------------------------- Simulation and value ---------------------------------

    def simulate_value(self, my_orders):
        game_copy = copy_game(self.game)

        # PRE locations for progress signal
        for p in game_copy.powers.keys():
            if p == self.power_name:
                game_copy.set_orders(p, my_orders)
            else:
                game_copy.set_orders(p, self.opponent_orders_random_lite(game_copy, p))
        game_copy.process()

        return self.heuristic(game_copy, my_orders)
    
    def opponent_orders_random_lite(self, game_copy, power_name):
        poss = game_copy.get_all_possible_orders()
        locs = game_copy.get_orderable_locations(power_name)
        scs = set(game_copy.get_centers(power_name))   
        out = []
        for loc in locs:
            options = poss.get(loc, [])
            if not options:
                continue
            # if unit is sitting on an owned SC bias to hold
            if loc in scs:
                holds = [i for i in options if i.endswith(" H")]
                if holds:
                    out.append(random.choice(holds))
                    continue
            # otherwise do move, ignore convoys
            moves = [i for i in options if " - " in i and " VIA" not in i]
            if moves:
                out.append(random.choice(moves))
            else:
                out.append(random.choice(options))
        return out
        
    def heuristic(self, game_copy, my_orders=None):
        # Weights 
        W_DIFF = 3.0          # long-term pressure
        W_FOREIGN_SC = 4.0    # reward occupying neutral/enemy SCs after adjudication
        HOLD_OFF_SC = 4.0     # discourage idling off objective
        HOLD_ON_SC  = 3.0     # smaller nudge against camping on SCs
        BOUNCE      = 3.0     # punish failed moves

        # --- SC differential vs best opponent 
        my_scs = set(game_copy.get_centers(self.power_name))
        best_opp = max((len(game_copy.get_centers(p)) for p in game_copy.powers if p != self.power_name), default=0)
        score = W_DIFF * (len(my_scs) - best_opp)

        # --- reward being on an SC you don't own (neutral/enemy) after adjudication ---
        all_scs = set(game_copy.map.scs)
        score += W_FOREIGN_SC * sum(
            1 for u in game_copy.get_units(self.power_name)
            if (loc := u.split()[1].upper()) in all_scs and loc not in my_scs
        )

        # --- penalties: holds and bounces ---
        if my_orders and game_copy.phase_type == 'M':
            # penalties for holding
            for o in my_orders:
                if o.endswith(' H'):
                    loc = o.split()[1].upper()
                    score -= HOLD_ON_SC if loc in all_scs else HOLD_OFF_SC

            # exact bounce/void detection via engine order status
            try:
                status = game_copy.get_order_status(self.power_name) or {}
            except Exception:
                status = {}
            bounces = 0
            for o in my_orders:
                if ' - ' in o and 'VIA' not in o:
                    unit = ' '.join(o.split()[:2])
                    # if engine reports any non-empty status for this unit, count as failed move
                    if isinstance(status, dict) and status.get(unit):
                        bounces += 1
            score -= BOUNCE * bounces

        return score

    # ------------------------------- Graph creation and paths ----------------------------

    def dist_to_nearest_sc(self, loc, is_army, targets):
        if not targets:
            return None
        paths = self.army_paths if is_army else self.navy_paths
        d = None
        paths_from = paths.get(loc.upper())
        if not paths_from:
            return None
        for sc in targets:
            p = paths_from.get(sc.upper())
            if p:
                L = len(p) - 1  # edges count
                if d is None or L < d:
                    d = L
        return d

    # next hop to get to target SCs   
    def next_hop_nearest_sc(self, loc, is_army, targets):
        if not targets:
            return None
        paths = self.army_paths if is_army else self.navy_paths
        pf = paths.get(loc.upper())
        if not pf:
            return None

        # collect candidate shortest paths from loc to each target
        candidates = []
        for sc in targets:
            p = pf.get(sc.upper())
            if p and len(p) > 1:
                candidates.append(p)

        if not candidates:
            return None

        my_locs = {u.split()[1].upper() for u in self.game.get_units(self.power_name)}

        # prefer shortest paths first, and among those, prefer first steps not occupied by our unit
        candidates.sort(key=len)
        for p in candidates:
            if p[1].upper() not in my_locs:
                return p[1]
        # fallback if all first steps are currently ours (may still be fine)
        return candidates[0][1]
        
    def build_map_graphs(self):    # You can re-use this function if needed
        if not self.game:
            raise Exception('Game Not Initialised. Cannot Build Map Graphs.')

        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        locations = list(self.game.map.loc_type.keys())  # locations with '/' are not real provinces

        for i in locations:
            if self.game.map.loc_type[i] in ['LAND', 'COAST']:
                self.map_graph_army.add_node(i.upper())
            if self.game.map.loc_type[i] in ['WATER', 'COAST']:
                self.map_graph_navy.add_node(i.upper())

        locations = [i.upper() for i in locations]

        for i in locations:
            for j in locations:
                if self.game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)

    def precompute_paths(self):
        # precomputing shortest paths for lookups during action selection
        self.army_paths = {n: nx.single_source_shortest_path(self.map_graph_army, n)
                           for n in self.map_graph_army.nodes()}
        self.navy_paths = {n: nx.single_source_shortest_path(self.map_graph_navy, n)
                           for n in self.map_graph_navy.nodes()}