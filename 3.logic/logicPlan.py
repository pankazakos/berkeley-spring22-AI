# logicPlan.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import string
from typing import Dict, List, Tuple, Callable, Generator, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin, expr
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict(
    [(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict(
    [(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North': (0, 1), 'South': (
    0, -1), 'East': (1, 0), 'West': (-1, 0)}


# ______________________________________________________________________________
# QUESTION 1

def sentence1() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.

    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"
    exprlist = []
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    exprlist.append(A | B)
    exprlist.append(~A % (~B | C))
    exprlist.append(disjoin(~A, (~B | C)))

    return conjoin(exprlist)
    "*** END YOUR CODE HERE ***"


def sentence2() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.

    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"
    exprlist = []
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    D = Expr('D')
    exprlist.append(C % (B | D))
    exprlist.append(A >> (~B & ~D))
    exprlist.append(~(B & ~C) >> A)
    exprlist.append(~D >> C)

    return conjoin(exprlist)
    "*** END YOUR CODE HERE ***"


def sentence3() -> Expr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    (Project update: for this question only, [0] and _t are both acceptable.)
    """
    "*** BEGIN YOUR CODE HERE ***"
    palive0 = PropSymbolExpr("PacmanAlive", time=0)
    palive1 = PropSymbolExpr("PacmanAlive", time=1)
    pborn0 = PropSymbolExpr("PacmanBorn", time=0)
    pkilled0 = PropSymbolExpr("PacmanKilled", time=0)

    expr1 = palive0 & ~pkilled0
    expr1 = expr1 | ~palive0 & pborn0
    expr1 = palive1 % expr1

    expr2 = ~(palive0 & pborn0)
    expr3 = pborn0

    exprlist = [expr1, expr2, expr3]

    return conjoin(exprlist)
    "*** END YOUR CODE HERE ***"


def findModel(sentence: Expr) -> Dict[Expr, bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    cnf_sentence = to_cnf(sentence)
    return pycoSAT(cnf_sentence)


def findModelCheck() -> Dict[Any, bool]:
    """Returns the result of findModel(Expr('a')) if lower cased expressions were allowed.
    You should not use findModel or Expr in this method.
    This can be solved with a one-line return statement.
    """
    class dummyClass:
        """dummy('A') has representation A, unlike a string 'A' that has repr 'A'.
        Of note: Expr('Name') has representation Name, not 'Name'.
        """

        def __init__(self, variable_name: str = 'A'):
            self.variable_name = variable_name

        def __repr__(self):
            return self.variable_name
    "*** BEGIN YOUR CODE HERE ***"
    return {dummyClass('a'): True}
    "*** END YOUR CODE HERE ***"


def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # If premise & ~conclusion, is unsatisfiable, return True
    return not findModel(premise & ~conclusion)
    "*** END YOUR CODE HERE ***"


def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """
    "*** BEGIN YOUR CODE HERE ***"
    return not pl_true(inverse_statement, assignments)
    "*** END YOUR CODE HERE ***"

# ______________________________________________________________________________
# QUESTION 2


def atLeastOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals  ist is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    return disjoin(literals)
    "*** END YOUR CODE HERE ***"


def atMostOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """
    "*** BEGIN YOUR CODE HERE ***"
    combinations = itertools.combinations(literals, 2)
    clauses = [(~comb[0] | ~comb[1]) for comb in combinations]
    return conjoin(clauses)
    "*** END YOUR CODE HERE ***"


def exactlyOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    onlyOne = []
    onlyOne.append(atLeastOne(literals))
    onlyOne.append(atMostOne(literals))
    return conjoin(onlyOne)
    "*** END YOUR CODE HERE ***"

# ______________________________________________________________________________
# QUESTION 3


def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]] = None) -> Expr:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    now, last = time, time - 1
    # enumerate all possible causes for P[x,y]_t
    possible_causes: List[Expr] = []
    # the if statements give a small performance boost and are required for q4 and q5 correctness
    if walls_grid[x][y+1] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x, y+1, time=last)
                               & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x, y-1, time=last)
                               & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x+1, y, time=last)
                               & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x-1, y, time=last)
                               & PropSymbolExpr('East', time=last))
    if not possible_causes:
        return None

    "*** BEGIN YOUR CODE HERE ***"
    # Example: P[1,1]_1 <=> P[1,0]_0 & North_0
    current = PropSymbolExpr(pacman_str, x, y, time=now)
    succesor = current % disjoin(possible_causes)
    return succesor
    "*** END YOUR CODE HERE ***"


def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    now, last = time, time - 1
    # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    moved_causes: List[Expr] = []
    if walls_grid[x][y+1] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x, y-1, time=last)
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x+1, y, time=last)
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x-1, y, time=last)
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(
        pacman_str, x, y, time=last), ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    # using merged variables, improves speed significantly
    failed_move_causes: List[Expr] = []
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(
            wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(
            wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(
            wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
        walls_grid: 2D array of either -1/0/1 or T/F. Used only for successorAxioms.
            Do NOT use this when making possible locations for pacman to be in.
        sensorModel(t, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
        successorAxioms(t, walls_grid, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
        - Results of calling sensorModel(...), unless None.
        - Results of calling successorAxioms(...), describing how Pacman can end in various
            locations on this time step. Consider edge cases. Don't call if None.
    """
    pacphysics_sentences = []

    "*** BEGIN YOUR CODE HERE ***"
    in_game = []  # in-game coordinates (may contain inner walls)
    for x, y in all_coords:
        pacphysics_sentences.append(PropSymbolExpr(wall_str, x, y) >>
                                    ~PropSymbolExpr(pacman_str, x, y, time=t))
        if (x, y) in non_outer_wall_coords:
            in_game.append(PropSymbolExpr(pacman_str, x, y, time=t))

    pacphysics_sentences.append(exactlyOne(in_game))
    directions = [PropSymbolExpr(action, time=t) for action in DIRECTIONS]
    pacphysics_sentences.append(exactlyOne(directions))

    # Add expressions from sensorModel and successorAxioms if given
    if (sensorModel != None):
        pacphysics_sentences.append(sensorModel(t, non_outer_wall_coords))
    if (successorAxioms != None and t > 0):
        pacphysics_sentences.append(successorAxioms(
            t, walls_grid, non_outer_wall_coords))
    "*** END YOUR CODE HERE ***"

    return conjoin(pacphysics_sentences)


def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - action1 = to ensure match with autograder solution
        - problem = an instance of logicAgents.LocMapProblem
    Note:
        - there's no sensorModel because we know everything about the world
        - the successorAxioms should be allLegalSuccessorAxioms where needed
    Return:
        - a model where Pacman is at (x1, y1) at time t = 1
        - a model where Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(
        range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(
        range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    "*** BEGIN YOUR CODE HERE ***"
    KB.append(conjoin(pacphysicsAxioms(0, all_coords, non_outer_wall_coords,
              walls_grid=walls_grid, successorAxioms=None)))
    KB.append(conjoin(pacphysicsAxioms(1, all_coords, non_outer_wall_coords,
              walls_grid=walls_grid, successorAxioms=allLegalSuccessorAxioms)))
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    KB.append(PropSymbolExpr(action0, time=0))
    KB.append(PropSymbolExpr(action1, time=1))

    # Find solution where pacman is at (x1, y1) given the KB
    model1 = findModel(PropSymbolExpr(
        pacman_str, x1, y1, time=1) & conjoin(KB))

    # Find solution where pacman is not at (x1, y1) given the KB
    model2 = findModel(~PropSymbolExpr(
        pacman_str, x1, y1, time=1) & conjoin(KB))

    return (model1, model2)
    "*** END YOUR CODE HERE ***"

# ______________________________________________________________________________
# QUESTION 4


def positionLogicPlan(problem) -> List:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls_grid = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls_grid.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2),
                                        range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = ['North', 'South', 'East', 'West']
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    # Add initial state of pacman
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))

    # For each timestep pacman is exactly at one in-game coordinate and takes exactly one action
    for t in range(50):
        print("timestep: ", t)
        in_game = []
        for x, y in non_wall_coords:
            in_game.append(PropSymbolExpr(pacman_str, x, y, time=t))
        KB.append(exactlyOne(in_game))
        expr_actions = []
        for action in actions:
            expr_actions.append(PropSymbolExpr(action, time=t))
        KB.append(exactlyOne(expr_actions))

        # Add successors to Knowledge base
        if t > 0:
            for x, y in non_wall_coords:
                succ_axioms = pacmanSuccessorAxiomSingle(x, y, t, walls_grid)
                KB.append(conjoin(succ_axioms))

        model = findModel(PropSymbolExpr(
            pacman_str, xg, yg, time=t) & conjoin(KB))

        # Proceed to next move until the goal given the KB is satisfiable
        if (model is False):
            print("model: False")
        else:
            path = extractActionSequence(model, actions)
            return path
    return []
    "*** END YOUR CODE HERE ***"

# ______________________________________________________________________________
# QUESTION 5


def foodLogicPlan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = ['North', 'South', 'East', 'West']

    KB = []

    "*** BEGIN YOUR CODE HERE ***"

    # pacman initial state at time = 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))

    # initialize food at time = 0
    expr_food = [PropSymbolExpr(food_str, x, y, time=0) for x, y in food]
    KB.append(conjoin(expr_food))

    for t in range(50):
        print("timestep: ", t)
        in_game = [PropSymbolExpr(pacman_str, x, y, time=t)
                   for x, y in non_wall_coords]
        KB.append(exactlyOne(in_game))
        action_exprs = [PropSymbolExpr(action, time=t) for action in actions]
        KB.append(exactlyOne(action_exprs))

        if t > 0:
            for x, y in non_wall_coords:
                succ_axioms = pacmanSuccessorAxiomSingle(x, y, t, walls)
                KB.append(conjoin(succ_axioms))

        """Food successor axiom: if pacman is not at [x, y] and there is food at [x, y],
        then there will be food at [x, y] at time t+1.
        ~P[x,y]_t & food[x,y]_t >> food[x,y]_t+1"""
        food_axioms = []
        not_food_exprs = []
        for x, y in food:
            not_food_exprs.append(~PropSymbolExpr(food_str, x, y, time=t))
            premise = []
            premise.append(~PropSymbolExpr(pacman_str, x, y, time=t))
            premise.append(PropSymbolExpr(food_str, x, y, time=t))
            succ_expr = conjoin(premise) >> PropSymbolExpr(
                food_str, x, y, time=t+1)
            food_axioms.append(succ_expr)

        KB.append(conjoin(food_axioms))

        # Goal is no food left
        goal_expr = conjoin(not_food_exprs)
        model = findModel(goal_expr & conjoin(KB))

        if (model is False):
            print("model: False")
        else:
            path = extractActionSequence(model, actions)
            return path

    return []
    "*** END YOUR CODE HERE ***"

# ______________________________________________________________________________
# QUESTION 6


def localization(problem, agent) -> Generator:
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(
        range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(
        range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    KB = []

    "*** BEGIN YOUR CODE HERE ***"

    wall_exprs = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(wall_exprs))
    not_wall_exprs = [~PropSymbolExpr(wall_str, x, y)
                      for x, y in all_coords if (x, y) not in walls_list]
    KB.append(conjoin(not_wall_exprs))

    for t in range(agent.num_timesteps):
        print("Timestep:", t)
        succ = allLegalSuccessorAxioms if t > 0 else None
        KB.append(pacphysicsAxioms(t, all_coords,
                                   non_outer_wall_coords, walls_grid=walls_grid, sensorModel=sensorAxioms, successorAxioms=succ))
        action = PropSymbolExpr(agent.actions[t], time=t)
        KB.append(action)
        percepts = agent.getPercepts()
        rules = fourBitPerceptRules(t, percepts)
        KB.append(rules)

        """ Find possible locations. Pacman is provably at location (x,y), if KB entails the location of pacman.
        If KB entails that pacman is not at (x,y), then pacman is provably not at location (x,y). If a location
        of pacman given the KB is satisfiable, then it is possible that pacman is there. 
        """
        possible_locations = []
        for x, y in non_outer_wall_coords:
            pac_at = PropSymbolExpr(pacman_str, x, y, time=t)
            not_pac_at = ~PropSymbolExpr(pacman_str, x, y, time=t)
            if (entails(conjoin(KB), pac_at)):
                KB.append(pac_at)
            if (entails(conjoin(KB), not_pac_at)):
                KB.append(not_pac_at)
            if (findModel(pac_at & conjoin(KB))):
                possible_locations.append((x, y))
        agent.moveToNextState(agent.actions[t])

        "*** END YOUR CODE HERE ***"
        yield possible_locations

# ______________________________________________________________________________
# QUESTION 7


def mapping(problem, agent) -> Generator:
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(
        range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(
        range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)]
                 for x in range(problem.getWidth()+2)]

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"

    # Initial location of pacman
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))
    # There is no wall at initial location of pacman.
    known_map[pac_x_0][pac_y_0] = 0
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))

    for t in range(agent.num_timesteps):
        print("Timestep:", t)
        KB.append(pacphysicsAxioms(t, all_coords,
                                   non_outer_wall_coords, walls_grid=known_map, sensorModel=sensorAxioms, successorAxioms=allLegalSuccessorAxioms))
        action = PropSymbolExpr(agent.actions[t], time=t)
        KB.append(action)
        percepts = agent.getPercepts()
        rules = fourBitPerceptRules(t, percepts)
        KB.append(rules)

        for x, y in non_outer_wall_coords:
            wall_at = PropSymbolExpr(wall_str, x, y)
            not_wall_at = ~PropSymbolExpr(wall_str, x, y)
            if (entails(conjoin(KB), wall_at)):
                KB.append(wall_at)
                known_map[x][y] = 1
            if (entails(conjoin(KB), not_wall_at)):
                KB.append(not_wall_at)
                known_map[x][y] = 0

        agent.moveToNextState(agent.actions[t])
        "*** END YOUR CODE HERE ***"
        yield known_map

# ______________________________________________________________________________
# QUESTION 8


def slam(problem, agent) -> Generator:
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(
        range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(
        range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)]
                 for x in range(problem.getWidth()+2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"

    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))
    known_map[pac_x_0][pac_y_0] = 0
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))

    for t in range(agent.num_timesteps):
        print("Timestep: ", t)
        KB.append(pacphysicsAxioms(t, all_coords,
                                   non_outer_wall_coords, walls_grid=known_map, sensorModel=SLAMSensorAxioms, successorAxioms=SLAMSuccessorAxioms))
        action = PropSymbolExpr(agent.actions[t], time=t)
        KB.append(action)
        percepts = agent.getPercepts()
        rules = numAdjWallsPerceptRules(t, percepts)
        KB.append(rules)

        possible_locations = []
        for x, y in non_outer_wall_coords:
            wall_at = PropSymbolExpr(wall_str, x, y)
            not_wall_at = ~PropSymbolExpr(wall_str, x, y)
            pac_at = PropSymbolExpr(pacman_str, x, y, time=t)
            not_pac_at = ~PropSymbolExpr(pacman_str, x, y, time=t)
            if (entails(conjoin(KB), wall_at)):
                KB.append(wall_at)
                known_map[x][y] = 1
            if (entails(conjoin(KB), not_wall_at)):
                KB.append(not_wall_at)
                known_map[x][y] = 0
            if (entails(conjoin(KB), pac_at)):
                KB.append(pac_at)
            if (entails(conjoin(KB), not_pac_at)):
                KB.append(not_pac_at)
            if (findModel(conjoin(KB) & pac_at)):
                possible_locations.append((x, y))

        agent.moveToNextState(agent.actions[t])

        "*** END YOUR CODE HERE ***"
        yield (known_map, possible_locations)


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

# ______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(
                pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(
            blocked_str_map[direction], time=t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(
            blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(
                blocked_str_map[direction], time=t)
        # The actual sensor readings
        percept_unit_clauses.append(percept_unit_clause)
    return conjoin(percept_unit_clauses)


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker numAdjWallsPerceptRules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(
                pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(
                pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(
            blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t)
                    for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = SLAMSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)

# ______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def modelToString(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.

    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False"
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    # return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualizeCoords(coords_list, problem) -> None:
    wallGrid = game.Grid(problem.walls.width,
                         problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualizeBoolArray(bool_arr, problem) -> None:
    wallGrid = game.Grid(problem.walls.width,
                         problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()
