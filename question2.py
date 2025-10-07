"""
Question 2: temporal dynamics and sequential planning with dynamic obstacles
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys
from heapq import heappush, heappop
from collections import defaultdict


#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!", e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = True
visualizer = True

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0

#########################
# get_path() function:
# Return a list of (x,y) location tuples which connect the start and goal locations.
# The path should avoid conflicts with existing paths.
#########################

# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param agent_id The id of given agent
# @param existing_paths A list of lists of locations indicate existing paths. The index of each location is the time that
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.

def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap,
            agent_id: int, existing_paths: list, max_timestep: int):
    """
    Time-expanded A* (SIPP-style) planner that avoids collisions with existing_paths.
    Returns a list of (x,y) tuples indexed by timestep (time 0..arrival_time). If no
    plan is found, returns [].

    Parameters:
    start: (x,y) start coordinates
    start_direction: int heading (0=N,1=E,2=S,3=W)
    goal: (x,y) goal coordinates
    rail: GridTransitionMap
    agent_id: id (unused here; kept for signature)
    existing_paths: list of lists: each existing_paths[i][t] is the (x,y) location of agent i at time t (if present)
    max_timestep: int
    """

    # helpers: directions (0=N,1=E,2=S,3=W) with dx,dy
    DIRS = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}

    # Build reservation sets from existing_paths:
    #  - pos_reserved[(x,y)] = set(times) when (x,y) is occupied by some existing train
    #  - edge_reserved[((ax,ay),(bx,by),t)] = True when some train moves ax,ay -> bx,by from time t -> t+1
    
    pos_reserved = defaultdict(set)
    edge_reserved = set()

    for p in existing_paths:
        # p is a list of positions, index is time; some p may be shorter/longer than max_timestep
        for t, pos in enumerate(p):
            if pos is None:
                continue
            pos_reserved[pos].add(t)
            # if there is a next timestep, register edge movement (pos -> next_pos)
            if t+1 < len(p):
                nextpos = p[t+1]
                if nextpos is None:
                    continue
                edge_reserved.add((pos, nextpos, t))

    # quick membership helpers
    def pos_is_free_at(pos, t):
        # returns True if position pos is not reserved at time t
        return t not in pos_reserved.get(pos, set())

    def edge_is_free(a, b, t):
        # returns True if there is no other agent moving b->a at time t (swap) and no one occupying b at t+1
        # check swap:
        if (b, a, t) in edge_reserved:
            return False
        # check b occupied at t+1
        if (t+1) in pos_reserved.get(b, set()):
            return False
        return True

    # Heuristic: Manhattan distance (ignores heading). Return minimal remaining timesteps
    def heuristic(p):
        return abs(p[0]-goal[0]) + abs(p[1]-goal[1])

    # start validity: ensure start is free at time 0 (otherwise agent can't start). If occupied, no solution.
    if not pos_is_free_at(start, 0):
        # in some setups start can be allowed if reservation belongs to the same agent,
        # but existing_paths belong to other agents; per instructions, just return [] if blocked.
        return []

    # Priority Queue nodes: (f, g, (x,y), dir, t)
    open_heap = []
    g_values = dict()  # key: (pos,dir,t) -> g
    parent = dict()    # key: (pos,dir,t) -> (prev_pos, prev_dir, prev_t)

    start_key = (start[0], start[1], start_direction, 0)
    g_values[start_key] = 0
    heappush(open_heap, (heuristic(start) + 0, 0, start, start_direction, 0))

    while open_heap:
        f, g, pos, dir_, t = heappop(open_heap)
        key = (pos[0], pos[1], dir_, t)

        # prune if we've seen this state with a better g
        if g_values.get(key, float('inf')) < g:
            continue

        # If reached goal (position equals goal), reconstruct path
        if pos == goal:
            # ensure pos is free at time t (should be, since we only added free nodes)
            # reconstruct
            path = []
            cur = key
            while cur in parent or cur == start_key:
                x, y, d, ct = cur
                path.append((x, y))
                if cur == start_key:
                    break
                cur = parent[cur]
            path.reverse()
            return path  # list indexed by timestep 0..arrival

        # Do not expand beyond max_timestep
        if t >= max_timestep:
            continue

        # wait action: stay in the same cell for t+1 if free
        # allow waiting only if pos is free at t+1
        if pos_is_free_at(pos, t+1):
            wait_key = (pos[0], pos[1], dir_, t+1)
            new_g = g + 1
            if new_g < g_values.get(wait_key, float('inf')):
                g_values[wait_key] = new_g
                parent[wait_key] = key
                heappush(open_heap, (new_g + heuristic(pos), new_g, pos, dir_, t+1))

        # move actions: ask rail for transitions given current heading.
        # rail.get_transitions(x,y,dir) returns an array of 4 bools (N,E,S,W) indicating available exits.
        try:
            transitions = rail.get_transitions(pos[0], pos[1], dir_)
        except Exception:
            # If rail.get_transitions fails for some reason, skip expansions
            transitions = [False, False, False, False]

        # transitions is an iterable for cardinal directions, if it's not then try to coerce
        # Iterate possible outgoing directions
        for out_dir, allowed in enumerate(transitions):
            if not allowed:
                continue
            dx, dy = DIRS[out_dir]
            nb = (pos[0] + dx, pos[1] + dy)
            # check map bounds
            if not (0 <= nb[0] < rail.height and 0 <= nb[1] < rail.width):
                continue

            # check that target cell is free at t+1 and no edge-swap collision
            if not pos_is_free_at(nb, t+1):
                continue
            if not edge_is_free(pos, nb, t):
                continue

            # valid move: new heading is out_dir
            new_key = (nb[0], nb[1], out_dir, t+1)
            new_g = g + 1
            if new_g < g_values.get(new_key, float('inf')):
                g_values[new_key] = new_g
                parent[new_key] = key
                heappush(open_heap, (new_g + heuristic(nb), new_g, nb, out_dir, t+1))

    # no solution found
    return []


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,2)


















