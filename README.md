# About
In this work, the genetic algorithm is used to address the V2X resource allocation.
# Pseudocode
	  // start with an initial time
	  t := 0;

	  // initialize a usually random population of individuals
	  initpopulation P (t);

	  // evaluate fitness of all initial individuals of population
	  evaluate P (t);

	  // test for termination criterion (time, fitness, etc.)
	  while not done do

	       // increase the time counter
	       t := t + 1;

	       // select a sub-population for offspring production
	       P' := selectparents P (t);

	       // recombine the "genes" of selected parents
	       recombine P' (t);

	       // perturb the mated population stochastically
	       mutate P' (t);

	       // evaluate it's new fitness
	       evaluate P' (t);

	       // select the survivors from actual fitness
	       P := survive P,P' (t);
	  od
     end GA.
