# NonGenCom
Library with independent functions to compare two databases (Forensic Case Database and Missing Person Database), generating scores based on non-genetic variables.

## How to install (temp)
Install the package using `pip install .` from base folder. 
> Tip: if you are a developer add the `-e` flag so any changes you make to the code can directly affect execution.

# Tasks
Functions to compute scores for the variables Biological Sex, Age, Body Characteristics, Dental Characteristics, Stature and Date of dead. 

## Inputs
In *nonGenCom/scenery_and_context_inputss/* , the inputs are defined for all the variables: 
Age (ranges, contexts, and sigma) 
Biological sex (fc_sceneries, mp_sceneries and contexts) 
Body (fc_sceneries, mp_sceneries and contexts) 

## Variables
In *nonGenCom/Variables/*  the two type of variables are defined:

*CategoricalVariable.py* defines the main (abstract) class for the categorical type variables until now (Biological Sex, Body). 
*ContinuosVariable.py* defines the main (abstract) class for the continuous type variables until now (age, stature). 


## Functions
In *nonGenCom/Variables/* , each function to compute the score for a variable are defined inside each of the subclasses of Variable: *BiologicalSex.py*, *AgeV2.py* and *AgeV3.py* (Age is an abstract class and should not be used as variable, this hierarchy might be subject to change).

## Examples of use
In *examples/* are shown some examples, like:

*fc_scoreCalculator_example.py* for implementing FC-selection search based on a set of elements of FC Database against all the MP Database.

*mp_scoreCalculator_example.py* for implementing MP-selection search based on a set of elements of MP Database against all the FC Database.

*age_examples.py* for calculating scores. 

*validation.py* to calculate the proportion of scores greater/equal/less than the correspondent case, for a specific score chosen. 

In *examples/resources*  are the inputs for the examples, like the two Databases of FC and MP for examples. 

