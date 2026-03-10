import heapq
from typing import List, Dict, Tuple, Set

def move(loc: Tuple[int], dir: int) -> Tuple[int]:
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0,0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def move_joint_state(locs: List[Tuple[int]], dirs: List[int]) -> List[Tuple[int]]:
    #both are lists of tuples
    assert(len(locs) == len(dirs))
    new_locs = []
    for i in range(len(locs)):
        new_locs.append(move(locs[i], dirs[i]))
    return new_locs


def generate_motions_recursive(num_agents: int, cur_agent: int):
    # directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    directions = range(5)

    joint_state_motions = [[dir] for dir in directions]
    for i in range(1, num_agents):
        expanded_motions = []
        for jsm in joint_state_motions:
            for dir in directions:
                # Expanded motions gets all combinations of previous joint state motions + new direction
                expanded_motions.append(jsm + [dir])
        joint_state_motions = expanded_motions
    

    return joint_state_motions


def is_valid_motion(old_loc, new_loc: List[Tuple[int]]):
    ##############################
    # Task 1.3/1.4: Check if a move from old_loc to new_loc is valid
    # Check if two agents are in the same location (vertex collision)

    num_agents = len(old_loc)
    vertex_occupied_new_loc: Set = set(new_loc)
    if len(vertex_occupied_new_loc) < num_agents:
        # This means there were duplicates removed when constructing a set, therefore a vertex was occupied at the same time by two agents
        return False

    # Check edge collision
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if old_loc[i] == new_loc[j] and old_loc[j] == new_loc[i]:
                # Edge collision detected, two agents swapped places
                return False
    
    return True


def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
                    or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3/1.4: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    
    constraints_table: Dict[int, List[Tuple[int]]] = dict()
    max_timestep = 0
    for constraint in constraints:
        if constraint['agent'] == agent:
            max_timestep = max(max_timestep, constraint['timestep'])
            if constraint['timestep'] in constraints_table:
                constraints_table[constraint['timestep']].append(constraint['loc'])
            else:
                constraints_table[constraint['timestep']] = [constraint['loc']]
    return constraints_table, max_timestep


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3/1.4: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    
    # Format:
    # ctable is {4: [[(1, 5)], [(1, 4)]], 1: [[(1, 2), (1, 3)]]} # Where t=4 has constraints on vertex (1,5)(1,4) and t=1 transition from (1,2)->(1,3)

    if next_time in constraint_table:
        # Vertex
        if [next_loc] in constraint_table[next_time]:
            return True
        # Edge
        if [curr_loc, next_loc] in constraint_table[next_time]:
            return True
        if [next_loc, curr_loc] in constraint_table[next_time]:
            return True
    
    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def in_map(map: List[List[bool]], loc: Tuple) -> bool:
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True


def all_in_map(map: List[List[bool]], locs: List[Tuple]) -> bool:
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, T=1000000):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        h_values    - precomputed heuristic values for each location on the map
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
        T           - timeout (nodes above this timestep will be pruned)
    """

    ##############################
    # Task 1.2/1.3/1.4: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    constraints_table, max_constraint_timestep = build_constraint_table(constraints=constraints, agent=agent)
    # print("ctable is " + str(constraints_table))
    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': 0}
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        #############################
        # Task 2.2: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc:
            goal_loc_constrained = False
            for t in range(curr['timestep'], max_constraint_timestep+1):
                if is_constrained(curr['loc'], curr['loc'], t, constraint_table=constraints_table):
                    goal_loc_constrained = True
                    break
            if not goal_loc_constrained:
                return get_path(curr)
        if curr['timestep'] >= T:
            continue 
        for dir in range(5):
            child_loc = move(curr['loc'], dir) #Logic to remain in same place is included in move()
            if not in_map(my_map, child_loc) or my_map[child_loc[0]][child_loc[1]]:
                continue
            if is_constrained(curr['loc'], child_loc, curr['timestep']+1, constraint_table=constraints_table):
                # print("is constrained")
                continue
            child = {'loc': child_loc,
                     'g_val': curr['g_val'] + 1,
                     'h_val': h_values[child_loc],
                     'parent': curr,
                     'timestep': curr['timestep'] + 1}
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions


def joint_state_a_star(my_map, starts, goals, h_values, num_agents):
    """ my_map      - binary obstacle map
        start_loc   - start positions
        goal_loc    - goal positions
        num_agent   - total number of agents in fleet
        h_values    - ASSUME this is distToGoal = h_values[goalNum][(x,y)]
    """

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = 0
    ##############################
    # Task 1.1: Iterate through starts and use list of h_values to calculate total h_value for root node
    #
    # TODO
    for i in range(num_agents):
        agent_start_loc = starts[i]
        h_value += h_values[i][agent_start_loc]
    print("Starting h_value is " + str(h_value))
    
    root = {'loc': starts, 'g_val': 0, 'h_val': h_value, 'parent': None}
    push_node(open_list, root)
    closed_list[tuple(root['loc'])] = root

    ##############################
    # Task 1.1:  Generate set of all possible motions in joint state space
    #
    directions = generate_motions_recursive(num_agents, 0)
    while len(open_list) > 0:
        curr = pop_node(open_list)

        if curr['loc'] == goals:
            return get_path(curr)

        for dir in directions:
            
            ##############################
            # Task 1.1:  Update position of each agent
            #
            child_loc = move_joint_state(curr['loc'], dir)

            if not all_in_map(my_map, child_loc):
                continue
            ##############################
            # Task 1.1:  Check if any agent is in an obstacle
            #
            valid_move = True
            
            for _child_loc in child_loc:
                (x,y) = _child_loc
                if my_map[x][y]:
                    valid_move = False
                    break
            
            if not valid_move:
                continue

            ##############################
            # Task 1.1:   check for collisions
            #
            # TODO
            if not is_valid_motion(curr['loc'], child_loc):
                continue

            ##############################
            # Task 1.1:  Calculate heuristic value
            #
            h_value = 0
            
            for i in range(num_agents):
                _child_loc = child_loc[i]
                h_value += h_values[i][_child_loc]

            # Create child node
            child = {'loc': child_loc,
                     'g_val': curr['g_val'] + num_agents,
                     'h_val': h_value,
                     'parent': curr}
            if tuple(child['loc']) in closed_list:
                existing_node = closed_list[tuple(child['loc'])]
                if compare_nodes(child, existing_node):
                    closed_list[tuple(child['loc'])] = child
                    push_node(open_list, child)
            else:
                closed_list[tuple(child['loc'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions
