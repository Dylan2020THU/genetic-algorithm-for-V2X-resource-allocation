# About
In this work, the genetic algorithm is used to address the V2X resource allocation problem.
# Pseudocode
The framework: crossover and mutation happen in each loop.

	  // start with an initial time
	  t := 0;

	  // initialization with a random population
	  P (t);

	  // evaluate fitness of all initial individuals of a population
	  evaluate Fitness (t);

	  // test for termination criterion (time, fitness, etc.)
	  while not done do

	       // increase the time counter
	       t := t + 1;

	       // select a sub-population for offspring production
	       P' := selectparents P (t);

	       // crossover using the selected parents
	       crossover P' (t);

	       // mutation
	       mutate P' (t);

	       // evaluate it's new fitness
	       evaluate P' (t);

	       // select the survivors from actual fitness
	       P := survive P,P' (t);
