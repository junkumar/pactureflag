# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint

from util import Counter
from distanceCalculator import manhattanDistance

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class CustomAgents(AgentFactory):

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

  def choose(self, agentStr, index):
    if agentStr == 'keys':
      global NUM_KEYBOARD_AGENTS
      NUM_KEYBOARD_AGENTS += 1
      if NUM_KEYBOARD_AGENTS == 1:
        return keyboardAgents.KeyboardAgent(index)
      elif NUM_KEYBOARD_AGENTS == 2:
        return keyboardAgents.KeyboardAgent2(index)
      else:
        raise Exception('Max of two keyboard agents supported')
    elif agentStr == 'offense':
      return OffensiveReflexAgent(index)
    elif agentStr == 'defense':
      return DefensiveReflexAgent(index)
    else:
      raise Exception("No staff agent identified by " + agentStr)

class AllOffenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)

  def getAgent(self, index):
    return OffensiveReflexAgent(index)

class OffenseDefenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)
    self.offense = False

  def getAgent(self, index):
    self.offense = not self.offense
    if self.offense:
      return OffensiveReflexAgent(index)
    else:
      return CustomDefensiveAgent(index)

##########
# Agents #
##########

class CustomCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  
  ########################### MAP FUNCTIONS ############################
  
  def setValidPositions(self, gameState):
    """
    Sets the field: validPositions to be a list of all valid position
    tuples on the map
    """
    self.validPositions = []
    walls = gameState.getWalls()
    for x in range(walls.width):
      for y in range(walls.height):
        if not walls[x][y]:
          self.validPositions.append((x,y))

  def getValidNeighboringPositions(self, gameState, (x,y)):
    """
    Returns a list of valid neigboring tuple positions to the given position
    (x,y). The position (x,y) itself is returned in the list
    """
    walls = gameState.getWalls()
    positions = [(x,y)]
    if x-1 >= 0 and not walls[x-1][y]: positions.append((x-1,y))
    if y+1 < walls.height and not walls[x][y+1]: positions.append((x,y+1))
    if x+1 < walls.width and not walls[x+1][y]: positions.append((x+1,y))
    if y-1 >= 0 and not walls[x][y-1]: positions.append((x,y-1))
    
    return positions
    
  ######################## INFERENCE FUNCTIONS ########################
    
  def initializeDistribution(self, gameState, agent):
    """
    Initializes the belief distribution in the field: beliefDistributions
    that corresponds to that of the given agent.  All valid positions on
    the map are given an equal probability
    """
    self.beliefDistributions[agent] = Counter()
    walls = gameState.getWalls()
    for (x,y) in self.validPositions:
      if gameState.isOnRedTeam(agent) and x <= walls.width/2 or \
        not gameState.isOnRedTeam(agent) and x >= walls.width/2:
        self.beliefDistributions[agent][(x,y)] = 1
    self.beliefDistributions[agent].normalize()
          
  def initializeBeliefDistributions(self, gameState):
    """
    Initializes the belief distributions in the field: beliefDistributions
    for all enemy agents
    """
    self.beliefDistributions = dict()
    for agent in self.getOpponents(gameState):
      distribution = Counter()
      self.initializeDistribution(gameState, agent)
      
  def observe(self, observedState):
    """
    Inference observation function:
    Combines the existing belief distributions with the noisy distances
    measured to each enemy agent and updates the distributions accordingly
    """
    agentPosition = observedState.getAgentPosition(self.index)
    noisyDistances = observedState.getAgentDistances()
    
    newDistributions = dict()
    for agent in self.getOpponents(observedState):
      if self.beliefDistributions[agent].totalCount() == 0:
        self.initializeDistribution(observedState, agent)
      distribution = Counter()
      if observedState.data.agentStates[agent].configuration != None:
        distribution[observedState.data.agentStates[agent].configuration.getPosition()] = 1
      else:
        for pos in self.validPositions:
          distance = manhattanDistance(agentPosition, pos)
          distribution[pos] = self.beliefDistributions[agent][pos] * \
            observedState.getDistanceProb(distance, noisyDistances[agent])
        distribution.normalize()
      newDistributions[agent] = distribution
    self.beliefDistributions = newDistributions
 
  def elapseTime(self, observedState):
    """
    Inference time elapse function:
    Updates the belief distributions for all enemy agents based on their
    possible moves and the likelihood of each move
    """
    newDistributions = dict()
    for agent in self.getOpponents(observedState):
      distribution = Counter()
      for pos in self.validPositions:
        newPosDist = Counter()
        for neighboringPos in self.getValidNeighboringPositions(observedState, pos):
          newPosDist[neighboringPos] = 1
        newPosDist.normalize()
        for newPos, prob in newPosDist.items():
          distribution[newPos] += self.beliefDistributions[agent][pos] * prob
      distribution.normalize()
      newDistributions[agent] = distribution
    self.beliefDistributions = newDistributions
    
  ###################### CONVENIENCE FUNCTIONS ######################
    
  def getMostLikelyPosition(self, agent):
    """
    Returns the most likely position as a (x,y) tuple for the given agent
    """
    return self.beliefDistributions[agent].argMax()
    
  def getClosestAttacker(self, observedState):
    """
    Returns the agent number for the closest attacker (invaders i.e. pacmen)
    are searched for first, if no invaders are found, the closest defender
    (ghost) is returned
    """    
    myPos = observedState.getAgentPosition(self.index)
    closestAttacker = None
    isPacman = False
    minDistance = float('inf')

    for agent in self.getOpponents(observedState):
      attackerPos = observedState.getAgentPosition(agent)
      if attackerPos is None: attackerPos = self.getMostLikelyPosition(agent)
      attackerDist = self.getMazeDistance(myPos, attackerPos)
      if (not isPacman and (attackerDist < minDistance or \
        observedState.getAgentState(agent).isPacman)) or \
        (observedState.getAgentState(agent).isPacman and \
        attackerDist < minDistance):
        if observedState.getAgentState(agent).isPacman: isPacman = True
        minDistance = attackerDist
        closestAttacker = agent
          
    return closestAttacker
    
  def registerInitialState(self, gameState):
    """
    State initializion function that (in addition to superclass function)
    calculates valid map positions and initializes enemy belief distributions
    """
    self.red = gameState.isOnRedTeam(self.index)
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    self.distancer.getMazeDistances()
    
    self.setValidPositions(gameState)
    self.initializeBeliefDistributions(gameState)
 
    import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display
      
  ######## THESE FUNCTIONS ARE CALLED/OVERRIDDEN BY REFLEX AGENTS #######

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
   
    """
    TEST CODE
    distributions = []
    for agent in range(gameState.getNumAgents()):
      if agent in self.beliefDistributions:
        distributions.append(self.beliefDistributions[agent])
      else:
        distributions.append(None)    
    for agent in self.getOpponents(gameState):
      print "Agent " + str(agent) + "'s most likely position is: " + str(self.getMostLikelyPosition(agent))
    self.displayDistributionsOverPositions(distributions)
    """
    
    observedState = self.getCurrentObservation()
    self.observe(observedState)
    self.elapseTime(observedState)

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class CustomExMaxAgent(CustomCaptureAgent):
  """
  A base class for expectimax agents that chooses score-maximizing actions
  To use this agent derive the offensive/defensive agents from this instead of CustomCaptureAgent
  """
  def expectimax(self, state, depth, visibleAgents, visIndex):
    agent = visibleAgents[visIndex]
    # print "Blue", state.getBlueTeamIndices()
    # print "Red", state.getRedTeamIndices()
    # print "This agent is", agent
    # print "self.index is", self.index
    # print "depth is", depth

    # Base case
    # if depth == 0 or state.isWin() or state.isLose():
    if depth == 0 or state.isOver():
      return self.evaluate(state, Directions.STOP)

    if visIndex == 0:
      nextagent = len(visibleAgents) - 1
    else:
      nextagent = visIndex - 1

    if agent == self.index:  # My ideal decision
      value = float('-inf')
      for action in state.getLegalActions(agent):
        successor = self.getSuccessor(state, action)
        result = self.expectimax(successor, (depth-1 if visIndex == 0 else depth), visibleAgents, nextagent)
        value = value if value > result else result
      return value   # Return max value

    else: # My opponents' decision
      value = 0
      numOptions = len(state.getLegalActions(agent))
      for action in state.getLegalActions(agent):
        successor = state.generateSuccessor(agent, action)
        value += (1.0/numOptions) * self.expectimax(successor, (depth-1 if visIndex == 0 else depth), visibleAgents, nextagent)
      return value  # Return sum of expected values

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    if gameState.isOnRedTeam(self.index): otherTeam = gameState.getBlueTeamIndices()
    else: otherTeam = gameState.getRedTeamIndices()

    value = None
    bestAction = Directions.STOP
    for action in gameState.getLegalActions(self.index):
      successor = self.getSuccessor(gameState, action)
      self.depth = 2 #FIXME - should do this in __init somewhere.
      allAgents = range(0, gameState.getNumAgents()-1)
      visibleAgents = [a for a in allAgents if gameState.getAgentState(a).getPosition() != None]
      # print "visibleAgents ", visibleAgents 
      result = self.expectimax(successor, self.depth, visibleAgents,
                               len(visibleAgents)-1)
      if result > value:
        value = result
        bestAction = action

    observedState = self.getCurrentObservation()
    self.observe(observedState)
    self.elapseTime(observedState)

    return bestAction

    
######################### CUSTOM DEFENSIVE AGENT #########################
    
class CustomDefensiveAgent(CustomCaptureAgent):

  def chooseAction(self, gameState):
  
    observedState = self.getCurrentObservation()
    self.observe(observedState)
    self.elapseTime(observedState)

    actions = observedState.getLegalActions(self.index)
    myPos = observedState.getAgentPosition(self.index)
    closestAttacker = self.getClosestAttacker(observedState)
    bestAction = Directions.STOP
    
    attackerPos = observedState.getAgentPosition(closestAttacker)
    if attackerPos is None: attackerPos = self.getMostLikelyPosition(closestAttacker)
    minDistance = self.getMazeDistance(myPos, attackerPos)
    for action in actions:
      successor = observedState.generateSuccessor(self.index, action)
      myNewPos = successor.getAgentPosition(self.index)
      newDist = self.getMazeDistance(myNewPos, attackerPos)
      if newDist < minDistance and not successor.getAgentState(self.index).isPacman:
        minDistance = newDist
        bestAction = action

    return bestAction
    
######################### SIMPLE REFLEX AGENTS ##########################

class OffensiveReflexAgent(CustomCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      self.distancer.getMazeDistances()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(CustomExMaxAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      self.distancer.getMazeDistances()
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


