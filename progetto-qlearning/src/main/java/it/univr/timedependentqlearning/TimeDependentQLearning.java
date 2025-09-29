package it.univr.timedependentqlearning;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import it.univr.usefulmethodsarrays.UsefulMethodsArrays;


/**
 * This abstract class provides the implementation of a model-free reinforcement learning algorithm
 * to solve stochastic control problems in the setting of time-dependent controlled Markov chains with discrete time
 * and discrete state space.
 * 
 * The class implements a time-dependent version of the Q-learning algorithm, where the value of the Q-function depends
 * explicitly on both the current state and the current time. This allows the method to handle fixed, finite-horizon decision problems.
 * Episodes are of fixed length equal to numberOfTimes (i.e., they always terminate at final time, regardless of the state).
 * The structure of the problem assumes that the following elements are independent of time:
 *   - The set of admissible actions for each state;
 *   - The running reward function f, which only depends on the current state x and the action a;
 *   - The transition dynamics, i.e., the law used to sample the next state from the current one, given the chosen action.
 * 
 * The Bellman equation is used in the classical form:
 * Q(t, x, a) <- Q(t, x, a) + lambda ⋅ [f(x, a) + gamma ⋅ max_{b in A(y)} Q(t+1,y,b) − Q(t,x,a)],
 * where:
 *   - f(x, a) is the running reward obtained from taking action a in state x;
 *   - y is the next state reached from (x, a), according to unknown dynamics simulated through an abstract method;
 *   - A(y) is the set of admissible actions in state y;
 *   - lambda is the learning rate;
 *   - gamma is the discount factor.
 * 
 * At final time, rewards are assigned to the terminal states using the array rewardsAtStatesAtFinalTime. These values are
 * interpreted as terminal utilities or payoffs.
 * 
 * @author Andrea Mazzon
 */

public abstract class TimeDependentQLearning {

	// Terminal rewards assigned at final time, one for each state.
	private double[] rewardsAtStatesAtFinalTime; //--> lunghezza pari al numero degli stati

	// Discount factor gamma used in the Bellman equation: it must be in [0,1].
	private double discountFactor;

	// Running rewards f(x,a): they are supposed not to depend in time. runningRewards[i][j] = reward of choosing action j in state i.
	private double[][] runningRewards; //--> indipendente dal tempo e di dimensione pari a numero di stati * numero di azioni

	// Value function V(t,x): optimal expected return from state x at time t. valueFunctions[k][i] = V(t_k,x_i)
	private double[][] valueFunctions; //--> indipendente dall'azione e di dimensione pari a numero di tempi * numero di stati

	// Optimal policy: optimalActionsIndices[k][i] is the index of the action maximizing Q at time t_k and state x_i.
	private int[][] optimalActionsIndices;  //--> indipendente dall'azione e di dimensione pari a numero di tempi * numero di stati

	/*
	 *  Q-values Q(t,x,a): current estimate of action-value function at each (time, state, action) triple:
	 *  currentQValue[k,i,j]=Q(t_k, x_i, a_j)
	 */
	private double[][][] currentQValue;  //--> dimensione pari a numero di tempi * numero di stati * numero di azioni

	// Total number of states in the discrete state space
	private int numberOfStates;

	// Total number of episodes to perform
	private int numberOfEpisodes;

	// Total number of time steps
	private int numberOfTimes;
	
	/*
	 * The learning rate lambda that enters in the update rule 
	 * Q(t, x, a) <- Q(t, x, a) + lambda ⋅ [f(x, a) + gamma ⋅ max_{b in A(y)} Q(t+1,y,b) − Q(t,x,a)]
	 */
	private double learningRate;

	/*
	 * A parameter in [0,1] that characterizes the exploration probability: an action a at a given state x is
	 * randomly chosen in the set of possible actions for x with probability equal to explorationProbability, and is instead
	 * chosen as the maximizing action for the Q-value in x with probability equal to 1 - explorationProbability
	 */
	private double explorationProbability;

	//used to generate the random numbers to determine exploration or exploitation and to choose the random action for exploration
	private Random generator = new Random();


	/**
	 * It constructs an object to solve a stochastic control problem in the setting of controlled Markov chains for discrete time and
	 * discrete space, under the hypothesis that the transition probabilities from one state to the other (i.e., the probabilities
	 * defining "the environment") are not known.
	 * 
	 * 
	 * @param rewardsAtStatesAtFinalTime the terminal rewards g(x) for each state x, assigned only at final time
	 * @param discountFactor the discount factor gamma in [0,1] used in the Bellman update
	 * @param runningRewards matrix of immediate rewards f(x,a), time-independent; runningRewards[i][j] is the reward
	 *                       for taking action j in state i
	 * @param numberOfTimes the number of time steps per episode 
	 * @param numberOfEpisodes the number of independent episodes used for Q-learning
	 * @param learningRate the learning rate lambda used in the Q-value update rule
	 * @param explorationProbability the probability epsilon of choosing a random action instead of a greedy one
	 */

	public TimeDependentQLearning(double[] rewardsAtStatesAtFinalTime, double discountFactor, double[][] runningRewards, int numberOfTimes, int numberOfEpisodes,
			double learningRate, double explorationProbability) {
		this.rewardsAtStatesAtFinalTime = rewardsAtStatesAtFinalTime;
		numberOfStates = rewardsAtStatesAtFinalTime.length;
		this.discountFactor = discountFactor;
		this.runningRewards = runningRewards;
		this.numberOfTimes = numberOfTimes;
		this.numberOfEpisodes = numberOfEpisodes; 
		this.learningRate = learningRate;
		this.explorationProbability = explorationProbability;
	}
	
	private void generateOptimalValueAndPolicy() {

		//value functions are optimal actions are now matrices, as they also depend on time
		valueFunctions = new double[numberOfTimes][numberOfStates];
		optimalActionsIndices = new int[numberOfTimes][numberOfStates];

		//it has to be implemented in the derived classes
		int numberOfActions = getNumberOfActions();
		currentQValue = new double[numberOfTimes][numberOfStates][numberOfActions];

		for (int stateIndex = 0; stateIndex < numberOfStates; stateIndex ++) {
			//the index of the actions which are allowed for that state
			int[] possibleActionsIndices = computePossibleActionsIndices(stateIndex);

			//we make it a list because then it's easier to check if it contains the given action indices
			List<Integer> possibleActionsIndicesAsList = Arrays.stream(possibleActionsIndices).boxed().toList();			
			
			//times before final time
			for (int timeIndex = 0; timeIndex < numberOfTimes - 1 ; timeIndex ++) {
				for (int actionIndex = 0; actionIndex < numberOfActions; actionIndex ++) {
					currentQValue[timeIndex][stateIndex][actionIndex]=possibleActionsIndicesAsList.contains(actionIndex) ? 0 : Double.NEGATIVE_INFINITY;
				}
			}
			
			//now we are at final time: the values of the Q matrix are given by the final reward for any action
			for (int actionIndex = 0; actionIndex < numberOfActions; actionIndex ++) {
				currentQValue[numberOfTimes - 1][stateIndex][actionIndex]=rewardsAtStatesAtFinalTime[stateIndex];
			}
			
		}

		//now we go through the episodes

		//any episode starts from a randomly chosen state and terminates at final time
		for (int episodeIndex = 0; episodeIndex < numberOfEpisodes; episodeIndex ++) {

			//we generate a state: we take it in any case, as there are no absorbing states
			int stateIndex = generator.nextInt(numberOfStates);

			//this will be updated
			int chosenActionIndex;

			for (int timeIndex = 0; timeIndex < numberOfTimes - 1; timeIndex ++) {

				//choice of the next action
				if (generator.nextDouble()< explorationProbability){ // --> exploration: randomly chosen action
					int[] possibleActionsIndices = computePossibleActionsIndices(stateIndex);
					chosenActionIndex = possibleActionsIndices[generator.nextInt(possibleActionsIndices.length)];
				} else { // --> exploitation: one maximizing action                           
					chosenActionIndex = UsefulMethodsArrays.getMaximizingIndex(currentQValue[timeIndex][stateIndex]);
				}

				/*
				 * The index of the new state, randomly picked in a way which depends on the action and on the state
				 * (and not directly on time).
				 * Since the way is chosen depends on the specific problem, the method is abstract and gets implemented
				 * in the derived classes.
				 */
				int newStateIndex = generateStateIndex(stateIndex, chosenActionIndex);

				//currentQValue[timeIndex + 1][newStateIndex] is a row of the 3-dimensional matrix
				double maximumForGivenStateIndex = UsefulMethodsArrays.getMax(currentQValue[timeIndex + 1][newStateIndex]);

				//update
				currentQValue[timeIndex][stateIndex][chosenActionIndex] = currentQValue[timeIndex][stateIndex][chosenActionIndex] +
						learningRate * (runningRewards[stateIndex][chosenActionIndex] + discountFactor*maximumForGivenStateIndex-currentQValue[timeIndex][stateIndex][chosenActionIndex]) ;

				stateIndex = newStateIndex;
			}
		}

		//now we have run all the episodes, so we have our "final" currentQValue matrix. We then compute the value functions and the optimal actions

		for (int timeIndex = 0; timeIndex <= numberOfTimes - 1; timeIndex ++) {
			for (int stateIndex = 0; stateIndex <= numberOfStates - 1; stateIndex ++) {
				valueFunctions[timeIndex][stateIndex] = UsefulMethodsArrays.getMax(currentQValue[timeIndex][stateIndex]);
				optimalActionsIndices[timeIndex][stateIndex] = UsefulMethodsArrays.getMaximizingIndex(currentQValue[timeIndex][stateIndex]);
			}
		}
	}


	//we may need it for the methods of the derived classes
	protected double[][][] getCurrentQValue() {
		return currentQValue.clone();
	}

	/**
	 * It returns a matrix of doubles representing the value functions for every time (on rows) and state (on columns)
	 * 
	 * @return a matrix of doubles representing the value functions for every time and every state
	 */
	public double[][] getValueFunctions() {
		if (currentQValue == null) {
			generateOptimalValueAndPolicy();
		}
		return valueFunctions.clone();
	}

	/**
	 * It returns a a matrix of doubles representing the optimal actions providing the value functions for every time
	 * (on rows) and state (on columns)
	 * 
	 * @return a matrix of doubles representing the optimal actions for every time and every state
	 */
	public int[][] getOptimalActionsIndices() {
		if (valueFunctions == null) {
			generateOptimalValueAndPolicy();
		}
		return optimalActionsIndices.clone();
	}


	/**
	 * It returns the discount factor 
	 * 
	 * @return the discount factor
	 */
	public double getDiscountFactor() {
		return discountFactor;
	}
	
	/**
	 * It returns the number of possible states 
	 * 
	 * @return the number of possible states 
	 */
	public int getNumberOfStates() {
		return numberOfStates;
	}

	// DA IMPLEMENTARE NELLA CLASSE DERIVATA
	
	/**
	 * It returns the total number of possible actions (indipendent from time and state)
	 * @return the total number of possible actions (indipendent from time and state)
	 */
	protected abstract int getNumberOfActions();
	
	/**
	 * It computes and returns an array of integers which represents the indices of the actions that are allowed for the given state
	 * (independent on time)
	 * @return an array of integers which represents the indices of the actions that are allowed for the given state
	 */
	protected abstract int[] computePossibleActionsIndices(int stateIndex);

	/**
	 * It (randomly) generates the index of the next state, based on the old state index and on the chosen action index
	 * (independent on time)
	 * 
	 * @param oldStateIndex
	 * @param actionIndex
	 * @return the index of the next state
	 */
	protected abstract int generateStateIndex(int oldStateIndex, int actionIndex); 
}

