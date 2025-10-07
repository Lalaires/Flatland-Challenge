"""
Question 1: single-agent pathfinding without collisions or time constraints
"""
from lib_piglet.utils.tools import eprint
import glob, os, sys
import heapq

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
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
# get_path() function return a list of (x,y) location tuples which connect the start and goal locations.
#########################


# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, max_timestep: int):
    ############
    # A* algorithm with Manhattan distance heuristic
    ############
    
    # Priority queue for A* algorithm: (f_score, g_score, position, direction, path)
    open_set = [(0, 0, start, start_direction, [start])]
    
    # Keep track of visited states (position, direction) with their best g_score
    visited = {}
    
    while open_set:
        f_score, g_score, current_pos, current_dir, path = heapq.heappop(open_set)
        
        # Create state key for visited tracking
        state_key = (current_pos, current_dir)
        
        # Skip if we've already found a better path to this state
        if state_key in visited and visited[state_key] <= g_score:
            continue
            
        visited[state_key] = g_score
        
        # Check if we've reached the goal
        if current_pos == goal:
            return path
            
        # Check if we've exceeded max timestep
        if g_score >= max_timestep:
            continue
            
        # Get valid transitions from current position and direction
        valid_transitions = rail.get_transitions(current_pos[0], current_pos[1], current_dir)
        
        # Explore each valid transition
        for action in range(len(valid_transitions)):
            if not valid_transitions[action]:
                continue
                
            # Calculate new position based on action
            new_x, new_y = current_pos
            
            if action == Directions.NORTH:
                new_x -= 1
            elif action == Directions.EAST:
                new_y += 1
            elif action == Directions.SOUTH:
                new_x += 1
            elif action == Directions.WEST:
                new_y -= 1
            
            new_pos = (new_x, new_y)
            
            # Check bounds
            if (new_x < 0 or new_x >= rail.height or 
                new_y < 0 or new_y >= rail.width):
                continue
            
            # Calculate costs
            new_g_score = g_score + 1
            manhattan_distance = abs(new_x - goal[0]) + abs(new_y - goal[1])
            new_f_score = new_g_score + manhattan_distance
            
            # Create new path
            new_path = path + [new_pos]
            
            # Add to open set
            heapq.heappush(open_set, (new_f_score, new_g_score, new_pos, action, new_path))
    
    # If no path found, return empty path or fallback to original greedy approach
    return []


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"single_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"single_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,1)



















