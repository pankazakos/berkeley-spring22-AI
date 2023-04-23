# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        """Note: ghostscore is bigger than foodscore, because the main priority is to avoid ghosts 
        and then collect food
        """
        ghostDists = [manhattanDistance(
            newPos, ghost.getPosition()) for ghost in newGhostStates]
        ghostscore = 0
        for dist in ghostDists:
            # Avoid being too close on a ghost if it is not scared enough.
            if dist != 0 and newScaredTimes[0] < 5:
                ghostscore += 3 / dist

        foodDists = [manhattanDistance(newPos, food)
                     for food in newFood.asList()]
        foodscore = 0
        for dist in foodDists:
            # Urge pacman to eat food that is close
            if dist != 0:
                foodscore += 1 / dist
        return successorGameState.getScore() - ghostscore + foodscore


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(player):
        Returns a list of legal actions for an agent
        player=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(player, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState: GameState, depth, player):

            # if all players have done evaluation, return to pacman (first agent) on the next depth of minimax
            if player == gameState.getNumAgents():
                depth -= 1
                player = self.index

            # if it is game over or depth is 0 return static evaluation of position
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), gameState.getLegalActions(player)

            if player == self.index:
                # Maximizing Player
                maxEval = float("-inf")
                best_action = "STOP"
                for action in gameState.getLegalActions(player):
                    state = gameState.generateSuccessor(player, action)
                    val = minimax(state, depth, player + 1)[0]
                    if val > maxEval:
                        maxEval = val
                        best_action = action
                return maxEval, best_action
            else:
                # Minimizing Player
                minEval = float("inf")
                best_action = "STOP"
                for action in gameState.getLegalActions(player):
                    state = gameState.generateSuccessor(player, action)
                    val = minimax(state, depth, player + 1)[0]
                    if val < minEval:
                        minEval = val
                        best_action = action
                return minEval, best_action

        pacman = self.index
        depth = self.depth
        best = minimax(gameState, depth, pacman)[1]
        return best


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState: GameState, depth, alpha, beta, player):

            # if all players have done evaluation, return to pacman (first agent) on the next depth of minimax
            if player == gameState.getNumAgents():
                depth -= 1
                player = self.index

            # if it is game over or depth is 0 return static evaluation of position
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), gameState.getLegalActions(player)

            if player == self.index:
                # Maximizing Player
                maxEval = float("-inf")
                best_action = "STOP"
                for action in gameState.getLegalActions(player):
                    state = gameState.generateSuccessor(player, action)
                    val = minimax(state, depth, alpha, beta, player + 1)[0]
                    if val > maxEval:
                        maxEval = val
                        best_action = action
                    alpha = max(alpha, val)
                    if beta < alpha:
                        break
                return maxEval, best_action
            else:
                # Minimizing Player
                minEval = float("inf")
                best_action = "STOP"
                for action in gameState.getLegalActions(player):
                    state = gameState.generateSuccessor(player, action)
                    val = minimax(state, depth, alpha, beta, player + 1)[0]
                    if val < minEval:
                        minEval = val
                        best_action = action
                    beta = min(beta, val)
                    if beta < alpha:
                        break
                return minEval, best_action

        pacman = self.index
        depth = self.depth
        best = minimax(gameState, depth, float(
            "-inf"), float("inf"), pacman)[1]
        return best


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(gameState: GameState, depth, player):

            # if all players have done evaluation, return to pacman (first agent) on the next depth of expectimax
            if player == gameState.getNumAgents():
                depth -= 1
                player = self.index

            # if it is game over or depth is 0 return static evaluation of position
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), gameState.getLegalActions(player)

            if player == self.index:
                # Maximizing Player
                maxEval = float("-inf")
                best_action = "STOP"
                for action in gameState.getLegalActions(player):
                    state = gameState.generateSuccessor(player, action)
                    val = expectimax(state, depth, player + 1)[0]
                    if val > maxEval:
                        maxEval = val
                        best_action = action
                return maxEval, best_action
            else:
                # Chance Player
                best_action = "STOP"
                s = 0
                actions = gameState.getLegalActions(player)
                for action in actions:
                    state = gameState.generateSuccessor(player, action)
                    val = expectimax(state, depth, player + 1)[0]
                    s += val
                avg = s / len(actions)
                return avg, best_action

        pacman = self.index
        depth = self.depth
        best = expectimax(gameState, depth, pacman)[1]
        return best


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    totalscore = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    pellets = currentGameState.getCapsules()

    ghostDists = [manhattanDistance(newPos, ghost.getPosition())
                  for ghost in newGhostStates]
    ghostscore = 0
    for dist in ghostDists:
        # Avoid being too close on a ghost if it is not scared enough.
        if dist > 0 and newScaredTimes[0] < 5:
            ghostscore += 3 / dist
    totalscore -= ghostscore

    foodDists = [manhattanDistance(newPos, food) for food in newFood.asList()]
    foodscore = 0
    for dist in foodDists:
        # Urge pacman to eat food that is close
        if dist > 0:
            if newScaredTimes[0] >= 5:
                foodscore += 5 / dist
            else:
                foodscore += 1 / dist
    totalscore += foodscore

    pelletDists = [manhattanDistance(newPos, pellet) for pellet in pellets]
    pelletscore = 0
    for dist in pelletDists:
        if dist > 0:
            pelletscore += 2 / dist
    totalscore += pelletscore

    return totalscore


# Abbreviation
better = betterEvaluationFunction
