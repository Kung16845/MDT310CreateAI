'''
  แก้ code และเพิ่มเติมได้ใน class YourTeamAgent เท่านั้น 
  ตอนส่งไฟล์ ให้แน่ใจว่า YourTeamAgent ไม่มี error และ run ได้
  ส่งแค่ submission.py ไฟล์เดียว
'''
from util import manhattanDistance
from game import Directions
import random, util,copy
from typing import Any, DefaultDict, List, Set, Tuple

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
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState,agentIndex=0) -> str:
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is agent 0 and agent 1.

    gameState.getPacmanState(agentIndex):
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore(agentIndex):
        Returns the score of agentIndex (0 or 1) corresponding to the current state of the game

    gameState.getScores():
        Returns all the scores of the agents in the game as a list where first score corresponds to agent 0
    
    gameState.getFood():
        Returns the food in the gameState

    gameState.getPacmanPosition(agentIndex):
        Returns the pacman (agentIndex 0 or 1) position in the gameState

    gameState.getCapsules():
        Returns the capsules in the gameState

    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(agentIndex)
    gameState.getScaredTimes(agentIndex)

    # print(legalMoves)
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action,agentIndex) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str,agentIndex=0) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action,agentIndex)
    newPos = successorGameState.getPacmanPosition(agentIndex)
    oldFood = currentGameState.getFood()

    return successorGameState.getScore(agentIndex)


def scoreEvaluationFunction(currentGameState: GameState,agentIndex=0) -> float:
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore(agentIndex)

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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '8',agentIndex=0):
    self.index = agentIndex 
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Minimax agent
  """

  def getAction(self, gameState: GameState,agentIndex = 0) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent (0 or 1) takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore(agentIndex):
        Returns the score of agentIndex (0 or 1) corresponding to the current state of the game

      gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified (0 or 1). Returns Pac-Man's legal moves by default.

      gameState.getPacmanState(agentIndex):
          Returns an AgentState (0 or 1) object for pacman (in game.py)
          state.configuration.pos gives the current position
          state.direction gives the travel vector

      gameState.getNumAgents():
          Returns the total number of agents in the game

      gameState.getScores():
          Returns all the scores of the agents in the game as a list where first score corresponds to agent 0
      
      gameState.getFood():
          Returns the food in the gameState

      gameState.getPacmanPosition(agentIndex):
          Returns the pacman (agentIndex = 0 or 1) position in the gameState

      gameState.getCapsules():
          Returns the capsules in the gameState

      self.depth:
        The depth to which search should continue

    """
    self.index = agentIndex
    bestVal = -float('inf')
    bestAction = None
    scoreStart = copy.deepcopy(gameState.getScores())
    legalMoves = gameState.getLegalActions(agentIndex)
    
    if len(legalMoves) == 1:
      return legalMoves[0]
    else: 
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        val = self.minimax(successorGameState,(agentIndex+1)%2,self.depth-1)
        if val > bestVal:
          bestVal = val
          bestAction = action
      # print("score ",gameState.getScore(self.index))
      # print("score ",gameState.getScores())
      return bestAction

  def minimax(self,gameState: GameState,agentIndex,depth):
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return self.evaluationFunction(gameState,agentIndex)
    
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(agentIndex)
    # print(legalMoves)
    if agentIndex == self.index:
      best = -float('inf')
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        best = max(best,self.minimax(successorGameState,(agentIndex+1)%2,depth-1))
      return best
    else:
      best = float('inf')
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        best = min(best,self.minimax(successorGameState,(agentIndex+1)%2,depth-1))
      return best

  def evaluationFunction(self, currentGameState: GameState, action: str,agentIndex=0) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action,agentIndex)
    newPos = successorGameState.getPacmanPosition(agentIndex)
    oldFood = currentGameState.getFood()

    return successorGameState.getScore()

######################################################################################
# class YourTeamAgent(MultiAgentSearchAgent):
#   """
#     Your team agent
#     แก้ เพิ่มเติม ได้ใน class นี้เท่านั้น
#     แต่ห้ามแก้ชื่อ class หรือ method ที่กำหนดให้
#     แต่เพิ่ม method เองได้ และเรียกใช้ได้ใน method ใน class นี้
#   """
#   def getAction(self, gameState: GameState,agentIndex = 0) -> str:
#     pass
#     # ต้อง return action ที่ดีที่สุดด้วยนะ
#     #  return bestAction

#   def evaluationFunction(self, currentGameState: GameState, action: str,agentIndex=0) -> float:
#     # อาจจะไม่ใช้ก็ได้ แต่ถ้าจะใช้ ให้ return ค่าที่ดีที่สุด
#     pass
#######################################################################################

import heapq
import random

class YourTeamAgent(MultiAgentSearchAgent):
    def getAction(self, gameState, agentIndex=0) -> str:
        """Choose the best action, avoiding other agents or chasing them depending on their state."""
        # Get the current position of our Pacman agent
        current_position = gameState.getPacmanPosition(agentIndex)

        # Get information about the other agent
        other_index = (agentIndex + 1) % gameState.getNumAgents()
        other_position = gameState.getPacmanPosition(other_index)
        other_state = gameState.getPacmanState(other_index)

        # Get capsules and food positions
        capsules = gameState.getCapsules()
        food = gameState.getFood().asList()

        # Retrieve all legal actions
        legal_actions = gameState.getLegalActions(agentIndex)

        # Determine whether we are in "getCapsule" mode or "find" mode
        if capsules:
            mode = "getCapsule"
            targets = capsules
        elif food:
            mode = "getFood"
            targets = food
        else:
            mode = "random"
            targets = []

        # If no valid targets exist, choose a random legal action
        if not targets:
            return random.choice(legal_actions) if legal_actions else Directions.STOP

        # If targeting a capsule but the other agent is not vulnerable, avoid them
        if mode == "getCapsule" and other_state.scaredTimer == 0:
            path = self.a_star_search_avoid(gameState, current_position, targets, other_position)
        else:
            # If the other agent is vulnerable, chase them
            if other_state.scaredTimer > 0:
                targets = [other_position]
                mode = "find"

            path = self.a_star_search(gameState, current_position, targets)

        if path:
            return path[0]
        else:
            # If no path is found, choose a random legal action
            return random.choice(legal_actions) if legal_actions else Directions.STOP

    def a_star_search(self, gameState, start, targets):
        """A* Search to find the best path to any target."""
        if not targets:
            return None

        walls = gameState.getWalls()

        def heuristic(pos, goal):
            return util.manhattanDistance(pos, goal)

        def neighbors(pos):
            x, y = pos
            possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
            deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            neighbors = []

            for direction, (dx, dy) in zip(possible_directions, deltas):
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < walls.width and 0 <= new_y < walls.height and not walls[new_x][new_y]:
                    neighbors.append((direction, (new_x, new_y)))
            return neighbors

        frontier = []
        heapq.heappush(frontier, (0, start, []))
        explored = set()

        while frontier:
            cost, current, path = heapq.heappop(frontier)

            if current in explored:
                continue
            explored.add(current)

            if current in targets:
                return path

            for direction, neighbor in neighbors(current):
                new_cost = cost + 1 + min([heuristic(neighbor, target) for target in targets])
                heapq.heappush(frontier, (new_cost, neighbor, path + [direction]))

        return None

    def a_star_search_avoid(self, gameState, start, targets, avoid_pos):
        """A* Search to find the best path to any target, avoiding a specific position."""
        if not targets:
            return None

        walls = gameState.getWalls()

        def heuristic(pos, goal):
            return util.manhattanDistance(pos, goal)

        def neighbors(pos):
            x, y = pos
            possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
            deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            neighbors = []

            for direction, (dx, dy) in zip(possible_directions, deltas):
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < walls.width and 0 <= new_y < walls.height and not walls[new_x][new_y]:
                    if (new_x, new_y) != avoid_pos:
                        neighbors.append((direction, (new_x, new_y)))
            return neighbors

        frontier = []
        heapq.heappush(frontier, (0, start, []))
        explored = set()

        while frontier:
            cost, current, path = heapq.heappop(frontier)

            if current in explored:
                continue
            explored.add(current)

            if current in targets:
                return path

            for direction, neighbor in neighbors(current):
                new_cost = cost + 1 + min([heuristic(neighbor, target) for target in targets])
                heapq.heappush(frontier, (new_cost, neighbor, path + [direction]))

        return None








