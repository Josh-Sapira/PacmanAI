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


from math import inf
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        uneaten = oldFood.asList()
        walls = successorGameState.getWalls()
        top, right = walls.height-2, walls.width-2
        size = (top*right)
        if len(uneaten) == 0: return 0
        closest = uneaten[0]
        closestFood = util.manhattanDistance(newPos, closest)
        for point in uneaten[1:]:
            newCost = util.manhattanDistance(newPos, point)
            if newCost < closestFood:
                closest = point
                closestFood = newCost
        closestFood = size-closestFood
        score = 0
        x,y = newPos
        eatFood = oldFood[x][y]
        ghostScore =0
        count =0
        for i in newGhostStates:
            count +=1
            distance = util.manhattanDistance(newPos, i.getPosition())
            if(distance<2):
                return 1000*-1
            distance = -1/distance
            ghostScore += (distance)
        ghostScore/=count
        score+=closestFood
        score+=ghostScore
        if(eatFood):
            score += 1000
        return score

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()

                # Choose one of the best actions
        scores = []
        for action in legalMoves:
            action2, val = self.findMoves(gameState.generateSuccessor(0, action),1,1,action)
            scores.append((action, val))
        max = -inf
        maxAct = None
        for i in scores:
            act, val = i  
            if(val  >= max):
                max = val
                maxAct = act
        return maxAct
    def findMoves(self,gameState, id,depth,action):
        if(gameState.isWin()):
            return action,self.evaluationFunction(gameState)
        if(gameState.isLose()):
            return action,self.evaluationFunction(gameState)
        if((depth == (self.depth)) and (id == (gameState.getNumAgents()))):
                return action, self.evaluationFunction(gameState)

        
        nextId = id +1
        nextDepth = depth
        
        
        if(id == gameState.getNumAgents()):
            nextId =1
            id = 0
            nextDepth = depth +1
        legalMoves = gameState.getLegalActions(int(id))

                # Choose one of the best actions
        scores = []
        for action in legalMoves:
            scores.append(self.findMoves(gameState.generateSuccessor(id, action),nextId,nextDepth,action))
        min = inf
        minAct = None
        max = -inf
        maxAct = None
        for i in scores:
            act, val = i  
            if(val  <= min):
                min = val
                minAct = act 
            if(val  >= max):
                max = val
                maxAct = act
    
        if(id == 0):
            
            return (maxAct,max)
        else:
            
            return(minAct,min)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        min = inf
        minAct = None
        max = -inf
        maxAct = None  
        alpha = -inf
        beta = inf 
        alpha2 = alpha
        beta2 = beta 
        count =0   
        legalMoves = gameState.getLegalActions()
        scores = []
        
        for action in legalMoves:
            action2, val = self.findMoves(gameState.generateSuccessor(0, action),1,1,action,alpha2,beta2)
            if(val > max):
                max = val
                maxAct = action
            if val > alpha2:
                alpha2 = val
            if alpha2 >= beta2:
                return action,val

        return maxAct


    def findMoves(self,gameState, id,depth,action, alpha,beta):
            if(gameState.isWin()):
                return action,self.evaluationFunction(gameState)
            if(gameState.isLose()):
                return action,self.evaluationFunction(gameState)
            if((depth == (self.depth)) and (id == (gameState.getNumAgents()))):
                    return action, self.evaluationFunction(gameState)

            
            nextId = id +1
            nextDepth = depth
            
            
            if(id == gameState.getNumAgents()):
                nextId =1
                id = 0
                nextDepth = depth +1
            legalMoves = gameState.getLegalActions(int(id))
            total = 0 
            min = inf
            minAct = None
            max = -inf
            maxAct = None   
            alpha2 = alpha
            beta2 = beta 
            count =0   

            for action in legalMoves:
                # print(action)
                count+=1
                move,val = self.findMoves(gameState.generateSuccessor(id, action),nextId,nextDepth,action,alpha2,beta2)
                # print(id)
                # print(gameState.getNumAgents())
                # print(alpha2)
                
                if(id ==0):
                    if(val > max):
                        max = val
                        maxAct = action
                    if val > alpha2:
                        alpha2 = val
                    if alpha2 > beta2:
                        # print("PRUNE")
                        # print(legalMoves[count:])
                        return action,val
                else:
                    if(val < min):
                        min = val
                        minAct = action
                    if val < beta2:
                        beta2 = val
                        # print(beta2)
                        # print(val)
                        # print("NEW BETA")
                        # print(alpha2 >= beta2)
                        # print(legalMoves[count:])
                        
                    if alpha2 >beta2:
                        return action,val
                
            if(id == 0):
                return (maxAct,max)
            else:
                return(minAct,min)


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
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        scores = []
        for action in legalMoves:
            action2, val = self.findMoves(gameState.generateSuccessor(0, action),1,1,action)
            scores.append((action, val))
        max = -inf
        maxAct = None
        for i in scores:
            act, val = i  
            if(val  >= max):
                max = val
                maxAct = act
        return maxAct




    def findMoves(self,gameState, id,depth,action):
        if(gameState.isWin()):
            return action,self.evaluationFunction(gameState)
        if(gameState.isLose()):
            return action,self.evaluationFunction(gameState)
        if((depth == (self.depth)) and (id == (gameState.getNumAgents()))):
                return action, self.evaluationFunction(gameState)

        
        nextId = id +1
        nextDepth = depth
        
        
        if(id == gameState.getNumAgents()):
            nextId =1
            id = 0
            nextDepth = depth +1
        legalMoves = gameState.getLegalActions(int(id))

                # Choose one of the best actions
        scores = []
        for action in legalMoves:
            scores.append(self.findMoves(gameState.generateSuccessor(id, action),nextId,nextDepth,action))
        total = 0 
        min = inf
        minAct = None
        max = -inf
        maxAct = None
        
        for i in scores:
            act, val = i  
            total += val
            if(val  <= min):
                min = val
                minAct = act 
            if(val  >= max):
                max = val
                maxAct = act
    
        if(id == 0):
            return (maxAct,max)
        else:
            
            return(minAct,total/len(scores))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    walls = currentGameState.getWalls()
    top, right = walls.height, walls.width
    size = (top*right)
    "*** YOUR CODE HERE ***"
    uneaten = food.asList()
    score =0
    if len(uneaten) == 0: 
        score = 10000000
    else:
        closest = uneaten[0]
        x,y = pos
        xF,yF = closest
        closestFood = dis(pos, closest)
        total =closestFood
        for point in uneaten[1:]:
            newCost = dis(pos, point)
            total += newCost
            if newCost < closestFood:
                closest = point
                closestFood = newCost
        total/=len(uneaten)
        score -= 10*(closestFood)/size
        score -= len(uneaten)*100
        # score -= againstTwoWalls(pos,walls)*50

    

    
    scared = True
    for i in newScaredTimes:
        if (i ==0):
            scared = False
            break
    count =0
    ghostScore =0
    for i in ghostStates:
        count +=1
        distance = util.manhattanDistance(pos, i.getPosition())
        if(distance<2 and not scared):
            return 100000*-1
        ghostScore += (distance)
    
    ghostScore/=count
    if(not scared):
        score+=ghostScore/size
    else:
        score = -ghostScore
    return score


def dis(pos, pos2):
    x,y = pos
    x2,y2 = pos2
    xd = (x-x2)**2
    yd = (y-y2)**2
    return (xd+yd)**(1/2)
# Abbreviation
def againstTwoWalls(pos,walls):
    x,y = pos
    height = walls.height
    width = walls.width
    
    wall =0
    if(x >1):
        if(walls[x-1][y]):
            wall +=1

    if(y>1):
        if(walls[x][y-1]):
            wall +=1

    if(y != height-1):
        if(walls[x][y+1]):
            wall +=1
        
    if(x != width-1):
        if(walls[x+1][y]):
            wall +=1
    if(wall > 2):
        return 0.1
    return 0
better = betterEvaluationFunction
