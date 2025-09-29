package it.univr.timedependentqlearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;
import java.util.Random;

public class OptimalInvestmentWithQLearning extends TimeDependentQLearning {
	
	private double investmentStep; 
	
	private double maximumInvestment; 

	private double minimumWealthValue; 
	
	private double maximumWealthValue; 
	
	private double wealthStep;
	
	private double timeStep; 
	
	private double constantDrift;
	
	private double constantVolatility;		
	
	private DoubleUnaryOperator utilityFunction;
	
	private Random random = new Random();
	
	public OptimalInvestmentWithQLearning(
			double constantDrift, double constantVolatility,
			double discountFactor,  DoubleUnaryOperator utilityFunction, double minimumWealthValue, double maximumWealthValue,
			double wealthStep, double timeStep, int numberOfTimes, double investmentStep, double maximumInvestment, 
			int numberOfEpisodes, double learningRate, double explorationProbability) {
			super( IntStream.range(0, (int) Math.floor((maximumWealthValue - minimumWealthValue) / wealthStep) + 1)
				    .mapToDouble(i -> utilityFunction.applyAsDouble(minimumWealthValue + i * wealthStep))
				    .toArray(),
				    //rewardsAtStatesAtFinalTime
					discountFactor, 
					new double[(int) Math.floor((maximumWealthValue - minimumWealthValue) / wealthStep) + 1][(int) (maximumInvestment / investmentStep) + 1],
					//runningRewards 
					numberOfTimes, 
					numberOfEpisodes, 
					learningRate,
					explorationProbability);
		this.investmentStep = investmentStep;
		this.maximumInvestment = maximumInvestment;
		this.minimumWealthValue = minimumWealthValue;
		this.maximumWealthValue = maximumWealthValue;
		this.wealthStep = wealthStep;
		this.timeStep = timeStep;
		this.constantDrift = constantDrift;
		this.constantVolatility = constantVolatility;
		this.utilityFunction = utilityFunction;
	}

	@Override
	protected int getNumberOfActions() {
		int numberOfActions = (int) (maximumInvestment/investmentStep) + 1;
		return numberOfActions;
	}

	@Override
	protected int[] computePossibleActionsIndices(int stateIndex) {
		// Calcolo ricchezza disponibile
        double currentWealth = minimumWealthValue + stateIndex * wealthStep;
        int number = getNumberOfActions();
        double aValue = (maximumWealthValue - currentWealth) / currentWealth;
        double[] actions = new double[number];

        for(int i = 0; i < number; i++) {
                        actions[i] = investmentStep * i;
        }

        double[] possibleActions = Arrays.stream(actions).filter(x -> x <= aValue).toArray();
        int lengthPossibleActions = possibleActions.length;
        
        // vettore con indici da 0 (incluso) a lengthPossibleActions (escluso) che indicano gli indici delle possibili azioni da intraprendere nello stato di ricchezza 
        return IntStream.range(0, lengthPossibleActions).toArray();
	}

	@Override
	protected int generateStateIndex(int oldStateIndex, int actionIndex) {
		// x' = x + drift(x,t)*delta_t + volatility(x,t)*sqrt(delta_t)*dW con dW v.al. N(0,1) 
		// x' in [x_min , x_max]
		
		// Calcolo azione corrispondente ad actionIndex
	    double percInStock = actionIndex * investmentStep;    
	    
	    // Calcolo ricchezza attuale
	    double currentWealth = minimumWealthValue + oldStateIndex * wealthStep;
	    
	    double interestRate = - Math.log(getDiscountFactor())/timeStep;
	    
	    double dW = random.nextGaussian();
	    
	    // Drift del problema di Merton
	    double driftPart = percInStock*(constantDrift - interestRate) + interestRate;
	    // Diffusione del problema di Merton
	    double diffusionPart = constantVolatility*percInStock;
	    
	    //  Simulo la dinamica stocastica del capitale con un passo di Eulero
	    double newWealth = currentWealth + currentWealth*driftPart* timeStep + currentWealth*diffusionPart* Math.sqrt(timeStep) * dW;
	    
	    // Controllo che nuovo stato x rispetti intervallo
	    newWealth = Math.max(minimumWealthValue, Math.min(maximumWealthValue, newWealth));
	    
	    // Calcolo il corrispondente indice i al nuovo stato
	    int newStateIndex = (int) Math.round((newWealth - minimumWealthValue) / wealthStep);
	    
	    // Ultimo controllo rispetto intervallo
	    int maxStateIndex = (int) Math.round((maximumWealthValue - minimumWealthValue) / wealthStep);
	    newStateIndex = Math.max(0, Math.min(newStateIndex, maxStateIndex));
	    
	    return newStateIndex;
	    
	}

}
