# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import math
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #number of iterations 
        for i in range(self.iterations):
            
            ct = util.Counter()
            #for each state 
            for state in self.mdp.getStates():

                #for each action from the state 
                max = float("-inf")
                for action in self.mdp.getPossibleActions(state):

                    #max action
                    total = 0

                    #for each next state get prob 
                    states = self.mdp.getTransitionStatesAndProbs(state,action)

                    #for each next state
                    for nextState, prob in states:
                        reward = self.mdp.getReward(state,action,nextState)
                        sec = self.discount * self.values[nextState]
                        total += prob*(reward+sec)
                    
                    #if total > max 
                    #new max 
                    if total > max:
                        max = total
                if(max == float("-inf")):
                    max = 0
                ct[state] = max
            self.values = ct

                
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getTransitionStatesAndProbs(state, action)
        total = 0
        for nextState, prob in states:
            reward = self.mdp.getReward(state, action, nextState)
            total += prob * (reward + self.discount * self.values[nextState])
        return total


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max = -math.inf
        ret = None
        for action in self.mdp.getPossibleActions(state):
            total = 0
            states = self.mdp.getTransitionStatesAndProbs(state,action)
            for nextState, prob in states:
                reward = self.mdp.getReward(state,action,nextState)
                sec = self.discount * self.values[nextState]
                total += prob*(reward+sec)
            if total > max:
                max = total
                ret = action
        return ret
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        numStates = len(states)
        st = 0
        for i in range(self.iterations):
          state = states[st]
          st +=1 
          if(st == numStates):
            st =0
          if (not self.mdp.isTerminal(state)):
            li = []
            for action in self.mdp.getPossibleActions(state):
              val = self.computeQValueFromValues(state, action)
              li.append(val)
            self.values[state] = max(li)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        pred = {}
        states = self.mdp.getStates()
        pq = util.PriorityQueue()
        #find predessors 
        for state in states:
            if(not self.mdp.isTerminal(state)):
                max = -math.inf
                for action in self.mdp.getPossibleActions(state):
                    val = self.computeQValueFromValues(state, action)
                    if(val >max):
                        max = val
                    for newState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if (newState in pred):
                            pred[newState].add(state)
                        else:
                            pred[newState] = {state}
                diff = abs(self.values[state]- max)
                pq.push(state,-diff)

                
        for i in range(self.iterations):
            if(pq.isEmpty()):
                return
            state = pq.pop()
            if(not self.mdp.isTerminal(state)):
                max = -math.inf
                for action in self.mdp.getPossibleActions(state):
                    val = self.computeQValueFromValues(state, action)
                    if(val >max):
                        max = val
                    
                self.values[state] = max
            for prev in pred[state]:
                if (not self.mdp.isTerminal(prev)):
                    max = -math.inf
                    for action in self.mdp.getPossibleActions(prev):
                        val = self.computeQValueFromValues(prev, action)
                        if (val > max):
                            max = val
                    diff = abs(self.values[prev] - max)
                    if (diff > self.theta):
                        pq.update(prev,-diff)

        



