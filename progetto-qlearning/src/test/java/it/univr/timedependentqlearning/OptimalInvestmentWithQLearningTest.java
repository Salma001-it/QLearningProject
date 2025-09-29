package it.univr.timedependentqlearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

import net.finmath.plots.Named;
import net.finmath.plots.Plot2D;

public class OptimalInvestmentWithQLearningTest {

	/**
	 * This class tests the implementation of TimeDependentQLearning and its derived class OptimalInvestmentTest
	 * 
	 * @author Andrea Mazzon
	 *
	 */
	public static void main(String[] args) {

		//parameters for wealth (i.e., states) interval
		double minimumWealthValue = 0.0;
		double maximumWealthValue = 15;
		double wealthStep = 0.02;
		
		//parameters for times
		double timeStep = 0.05;
		int numberOfTimes = 20; 

		//possible actions go from 0 to maximumInvestment with this step
		double investmentStep = 0.01;
		double maximumInvestment = 2.0;
		
		//model parameters
		double constantDrift = 0.05;
		double constantVolatility = 0.4;
		
		double interestRate = 0.0;		
		//discount factor from one time to the other: computed from the interest rate
		double discountFactor = Math.exp(-interestRate*timeStep);
		
		//method parameters
		int numberOfEpisodes =  100000;
		double learningRate = 0.3;
		double explorationProbability = 0.1;// usually, one needs to choose a small exploration probability
		
		double exponentForFinalRewardFunction = 0.5;
		DoubleUnaryOperator utilityFunction = (x) -> Math.pow(x, exponentForFinalRewardFunction);
		
		OptimalInvestmentWithQLearning problemSolver = new OptimalInvestmentWithQLearning(constantDrift, constantVolatility, discountFactor, 
				utilityFunction, minimumWealthValue, maximumWealthValue, wealthStep,  timeStep, 
				numberOfTimes, investmentStep, maximumInvestment, numberOfEpisodes, learningRate, explorationProbability);
		
		int numberOfStates = problemSolver.getNumberOfStates();
		
		double[][] valueFunctions = problemSolver.getValueFunctions();
		
		int[][] optimalActionsIndices = problemSolver.getOptimalActionsIndices();
		
		int timeIndex = 0;
		
        // Output di ulteriori test nel pdf
    
		final Plot2D plotValueFunctions = new Plot2D(minimumWealthValue, maximumWealthValue, numberOfStates,  				
				Arrays.asList(
				new Named<DoubleUnaryOperator>("Value function", x -> valueFunctions[timeIndex][(int) ((x-minimumWealthValue)/wealthStep) ]) 
				));
		plotValueFunctions.setTitle("Value function for time " + timeIndex*timeStep);
		plotValueFunctions.setXAxisLabel("State");
		plotValueFunctions.setYAxisLabel("Value function");

		plotValueFunctions.show();
		
		final Plot2D plotOptimalActions = new Plot2D(minimumWealthValue, maximumWealthValue, numberOfStates , 
				Arrays.asList(
				new Named<DoubleUnaryOperator>("Optimal action", x -> optimalActionsIndices[timeIndex][(int) ((x-minimumWealthValue)/wealthStep) ])
				));
		plotOptimalActions.setTitle("Optimal investment on stock for time " + timeIndex*timeStep);
		plotOptimalActions.setXAxisLabel("State");
		plotOptimalActions.setYAxisLabel("Optimal investment");

		plotOptimalActions.show();	
		
		// test ad un altro istanto temporale
		int timeIndex2 = numberOfTimes-2;
    
		final Plot2D plotValueFunctions2 = new Plot2D(minimumWealthValue, maximumWealthValue, numberOfStates,  
			Arrays.asList(
				new Named<DoubleUnaryOperator>("Value function", x -> valueFunctions[timeIndex2][(int) ((x-minimumWealthValue)/wealthStep) ]) 
				));
		plotValueFunctions2.setTitle("Value function for time " + timeIndex2*timeStep);
		plotValueFunctions2.setXAxisLabel("State");
		plotValueFunctions2.setYAxisLabel("Value function");

		plotValueFunctions2.show();
		
		final Plot2D plotOptimalActions2 = new Plot2D(minimumWealthValue, maximumWealthValue, numberOfStates , 
				Arrays.asList(
				new Named<DoubleUnaryOperator>("Optimal action", x -> optimalActionsIndices[timeIndex2][(int) ((x-minimumWealthValue)/wealthStep) ])
				));
		plotOptimalActions2.setTitle("Optimal investment on stock for time " + timeIndex2*timeStep);
		plotOptimalActions2.setXAxisLabel("State");
		plotOptimalActions2.setYAxisLabel("Optimal investment");

		plotOptimalActions2.show();	

	}
}