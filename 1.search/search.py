# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self) -> list:
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state) -> bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state) -> list:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions) -> float:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Use Stack frontier for dfs
    frontier = util.Stack()
    # Expanded: nodes that have been visited once. set() is used for better performance
    expanded = set()

    # Add coordinates as tuple and path as list
    # Format for stack elements: ((),[])
    frontier.push((problem.getStartState(), []))

    # While there is a node in stack
    while True:
        if frontier.isEmpty():
            raise Exception("dfs failed: empty frontier")
        CurrentState, actions = frontier.pop()
        # Return list of actions if goal is reached
        if problem.isGoalState(CurrentState):
            return actions
        # Only for unvisited nodes
        if CurrentState not in expanded:
            expanded.add(CurrentState)
            # problem.getSuccessors() format: [((),'',int), ((),'',int), ((),'',int)]. 1-3 tuples depending on successors
            # Need to also specify cost in order to unpack
            for state, action, cost in problem.getSuccessors(CurrentState):
                path = actions + [action]
                # Push new coordinates and path in stack
                frontier.push((state, path))


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Use queue frontier for dfs
    frontier = util.Queue()
    expanded = set()
    frontier.push((problem.getStartState(), []))

    while True:
        if frontier.isEmpty():
            raise Exception("bfs failed: empty frontier")
        CurrentState, actions = frontier.pop()

        if problem.isGoalState(CurrentState):
            return actions
        if CurrentState not in expanded:
            expanded.add(CurrentState)
            for state, action, cost in problem.getSuccessors(CurrentState):
                path = actions + [action]
                frontier.push((state, path))


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Use Priority Queue frontier for ucs
    frontier = util.PriorityQueue()
    expanded = set()
    # Store cost in third element of tuple and priority as second element of PriorityQueue
    frontier.push((problem.getStartState(), [], 0), 0)

    while True:
        if frontier.isEmpty():
            raise Exception("ucs failed: empty frontier")

        CurrentState, actions, curr_cost = frontier.pop()
        if problem.isGoalState(CurrentState):
            return actions

        if CurrentState not in expanded:
            expanded.add(CurrentState)
            for state, action, cost in problem.getSuccessors(CurrentState):
                if state not in expanded:
                    path = actions + [action]
                    # ucs cost = cost of current node + cost of successor
                    frontier.push((state, path, curr_cost + cost),
                                  curr_cost + cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Use Priority Queue frontier for astar
    frontier = util.PriorityQueue()
    expanded = set()
    # Store cost in third element of tuple and priority as second element of PriorityQueue
    # Cost of first state and priority are 0
    frontier.push((problem.getStartState(), [], 0), 0)

    while True:
        if frontier.isEmpty():
            raise Exception("astar failed: empty frontier")

        CurrentState, actions, curr_cost = frontier.pop()
        if problem.isGoalState(CurrentState):
            return actions
        if CurrentState not in expanded:
            expanded.add(CurrentState)
            for state, action, cost in problem.getSuccessors(CurrentState):
                if state not in expanded:
                    path = actions + [action]
                    # astar cost = cost of current node + cost of successor + cost of heuristic of successor
                    hcost = curr_cost + cost + heuristic(state, problem)
                    frontier.push((state, path, curr_cost + cost), hcost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
