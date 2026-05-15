# main.py
from experiments import *
from policies import softmax

def main():
    #experiment_staghunt_epsilongreedy()
    #experiment_subsidy_game_epsilongreedy()
   # experiment_prisoners_dilemma_epsilongreedy()
   # experiment_rps_epsilongreedy()
    
    #experiment_staghunt_boltzmann()
    #experiment_subsidy_game_boltzmann()
    #experiment_prisoners_dilemma_boltzmann()

    # experiment_stag_hunt_lenient_boltzmann()
    #experiment_subsidy_game_lenient_boltzmann()
    #experiment_prisoners_dilemma_lenient_boltzmann()
    experiment_biased_rps_lenient_boltzmann()

if __name__ == "__main__":
    main()

