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
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

        #Finding the closest Food Particle..
        minFoodDistance = 0
        if successorGameState.getNumFood() is not 0:
        	minFoodDistance = 1.0/min([util.manhattanDistance(newPos,food) for food in newFood.asList()])

        #Finding the closest Non-Scared Ghost..
        newScaredGhostStateDistances = [util.manhattanDistance(newPos,ghostState.getPosition()) for ghostState in successorGameState.getGhostStates() if ghostState.scaredTimer is 0 and int(util.manhattanDistance(newPos,ghostState.getPosition())) is not 0]
        if len(newScaredGhostStateDistances) is 0:
        	return successorGameState.getScore()+(minFoodDistance)
        return successorGameState.getScore()+(minFoodDistance)-(1.0/min(newScaredGhostStateDistances))

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

"""
	Below are my functions defined for minimax, expectimax and alpha-beta pruning
"""
def alpha_beta_pruning(minimaxValue,isMaxState,isAlphaBetaPruning,alpha,beta):
	returnValue = False
	if isAlphaBetaPruning is True:
		if isMaxState is 0:
			if minimaxValue>beta:
				returnValue = True
			alpha = max(alpha,minimaxValue)
		else:
			if minimaxValue<alpha:
				returnValue = True
			beta = min(beta,minimaxValue)
	return (alpha,beta,returnValue)

def minimax_values(self,gameState,depth,agentIndex=0,isAlphaBetaPruning=False,isExpectimax=False,alpha=-99999,beta=99999):
    if agentIndex==gameState.getNumAgents():
    	if depth==self.depth:
    		return self.evaluationFunction(gameState)
    	return minimax_values(self,gameState,depth+1,0,isAlphaBetaPruning,isExpectimax,alpha,beta)
    if len(gameState.getLegalActions(agentIndex)) is 0:
    	return minimax_values(self,gameState,depth,agentIndex+1,isAlphaBetaPruning,isExpectimax,alpha,beta)
    minimaxValues = []
    expectimaxValue = 0.0
    for action in gameState.getLegalActions(agentIndex):
    	minimaxValue = minimax_values(self,gameState.generateSuccessor(agentIndex,action),depth,agentIndex+1,isAlphaBetaPruning,isExpectimax,alpha,beta)
    	minimaxValues.append((minimaxValue,action))
    	alpha, beta, isPruned = alpha_beta_pruning(minimaxValue,agentIndex,isAlphaBetaPruning,alpha,beta)
    	if isPruned:
    		break
    	if isExpectimax:
    		expectimaxValue+=(1.0/len(gameState.getLegalActions(agentIndex)))*minimaxValue
    if agentIndex is 0:
    	if depth is 1:
    		return max(minimaxValues)[1]
    	return max(minimaxValues)[0]
    if isExpectimax:
    	return expectimaxValue
    return min(minimaxValues)[0]

"""
	End for my defined functions for minimax, expectimax and alpha-beta pruning
"""

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # return self.max_value(gameState,1)
        return minimax_values(self,gameState,1)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return minimax_values(self,gameState,1,0,True)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return minimax_values(self,gameState,1,0,False,True)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      MY DESCRIPTION:
      	For calculating the evaluation function of the current state I have considered the following:-
      		1) The distances of all remaining food particles from pacman - (This is done because, the lesser the sum of the distances of food from pacman, the better it is for the pacman to eat all the remaining dots.)
      		2) The distances of all remaining capsules from pacman - (This is included as eating a capsule, enables us to eat ghosts, which in turn increases our points. So similar to the food, the closer pacman is to the capsules, the better the position is for pacman.)
      		3) The number of scared ghosts present at this state - (This is the key element required for this function to succeed. This is because, if this state is a capsule, then the net score as observed in the code below would have a very high value, as the moment a capsule is consumed, all ghosts will become scared thereby increasing chances of pacman eating the scared ghosts. In simpler terms, the higher the number of scared ghosts, the more are our chances of getting more points by eating them)

      	Thing that have not been considered:-
      		1) The distances of either ghosts or scared ghosts - (This is not included for the simple reason that, the movements of the pacman should not be affected by the movement of the ghosts except for the time when pacman is in close proximity with a ghost - which inturn will be automatically handled by state.score() which is already defined in pacman.py)
      		2) The number of food and capsules - (This is irrelevent as we are already considering all food and capsules while calculating their sum. So it would be meaningless to consider them here.)

      	Weights for each feature:-
      		1) 50 for num of scared ghosts - The highest weight is given to this as eating a scared ghost would earn us 200 points each.
      		2) 20 for Food - Primary goal of pacman is eating all the food.
      		3) 10 for Capsules - Secondary goal for pacman as it wouldn't yield a score but collecting them would give rise to scared ghosts.
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    
    #Cost Related to Food
    foodPositions = currentGameState.getFood()
    foodFinalCost = 0.0
    if currentGameState.getNumFood()>0:
    	foodFinalCost = 20*(1.0/sum([util.manhattanDistance(pacmanPosition,food) for food in foodPositions.asList()]))

    #Cost Related to the Ghosts (Calculating score based on number of scared ghosts..)
    ghostFinalCost = 0.0
    ghostFinalCost = 50*(len([ghostState for ghostState in currentGameState.getGhostStates() if ghostState.scaredTimer>1]))

    #Cost related to capsules
    capsulePositions = currentGameState.getCapsules()
    capsuleFinalCost = 0.0
    if len(capsulePositions)>0:
    	capsuleFinalCost = 10*(1.0/sum([util.manhattanDistance(pacmanPosition,capsule) for capsule in capsulePositions]))
    
    return currentGameState.getScore()+foodFinalCost+capsuleFinalCost+ghostFinalCost

# Abbreviation
better = betterEvaluationFunction

