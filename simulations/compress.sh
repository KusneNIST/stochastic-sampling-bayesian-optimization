#!/bin/bash

zip -r 867-supplement.zip appendix.pdf 
zip -r 867-supplement.zip README.md
zip -r 867-supplement.zip output/stochastic-sequential-d2 output/stochastic-batch5-d2 output/lac*sequential output/lac*batch5
zip -r 867-supplement.zip optimizer.py modeler.py objective.py distribution.py aquisition.py aggregate.py parameter.py plot.py run.py run_lac_mut.py
zip -r 867-supplement.zip performance.ipynb figures.ipynb lac-promoter.ipynb two-armed-bandit.ipynb
zip -r 867-supplement.zip data/*params*

