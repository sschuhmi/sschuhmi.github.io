# Predicting Results of Modern Football Matches using StatsBomb analysis data and advanced Multi-Output Machine Learning Regressors

![Football](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/football.jpg?raw=true)

# Project Definition

## Project Overview

Football - also known as soccer in the USA - is one of the most famous sports in the world. In the last years, the amount of analysis data that are generated in modern football matches using cameras, drones or
other advanced technologies has significantly increased. StatsBomb [[1]](#ref1) has become one of the most famous analysis data providers in this area, delivering tons of analysis data of matches all over the planet. Besides data where the customer needs to pay for, there is also a large StatsBomb open data repository in GitHub available [[2]](#ref2) which includes more than 3500 matches from 21 competitions, reaching from national competitions up to continental and global championships like the FIFA World Cup 2022.

An interesting question that arises with this constantly increasing amount of football analysis data is the following:

*“Suppose you have the analysis data of a specific match available, without knowing who know the match or if there was a draw. Solely based on the analysis of data of previous matches where the winners are known, is it possible to predict the winner of this match with a high accuracy by taking advantage of modern Machine Learning (ML) algorithms?”*

That´s exactly what is the focus of this project: We exploit the large open data set and use Machine Learning multi-output Classifiers and Regressors to predict the winner of a match, compared to a simple non-ML Classifier which randomly decides one of the possible three results - first team wins, second team wins, or there is a draw between the teams.

## Problem Statement
The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion of the expected solution, has been made.

The problem which needs to be solved here is the prediction of the winner of a match without exploiting the score data (i.e., which team shot how many goals).
A football match always consists of exactly two teams that play against each other: The first team is called the 'home team', while the second team is called the 'away team'. In order to win a match, a team has to score more goals than the other team after full time (90 minutes; note that in international championships, it is common to continue the match in case of a draw after full time by extra time and/or penalty shooting; however, we do not regard extra time or penalty shootings here). If both teams scored the same number of goals, the result is called a 'draw'.

Thus in a single match, there are always 3 possible results where exactly one will be the outcome:
- 'win_home': The home team scores more goals than the away team and wins
- 'win_none': The home team and the away team score exactly the same number of goals, i.e. the result is a draw
- 'win_away': The away team scores more goals than the home team and wins

In order to be processable by an algorithm and allow the observation of multiple matches at the same time, the 3 variables are in the following encoded as array of three columns with binary values, where the rows represent the matches and the columns represent the three possible outcomes (i.e., results) of the matches (1 if the result has entered, otherwise 0).

As an example, consider the following Figure 1 which represents score data from five matches with the number of goals of the home team ('home_score') and the number of goals of the away team ('away_score') and the resulting (3x5)-outcome vector
![Fig1](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Result_Vector.jpg?raw=true)

## Metrics


# Analysis

## Data Exploration

Currently in October 2024, StatsBomb´s open data represents only a small, but yet comprehensive subset of the overall StatsBomb data. 

The data is provided as JSON files exported from the StatsBomb Data API, in the following structure:
- 21 competitions with 74 seasons
- Almost 3500 matches, spread among the above mentioned competitions and seasons
- Events and lineups for each of these matches, included additional StatsBomb 360 data for selected matches

A fine-grained documentation of the files, their formats and their contents is also included in the GitHub repository. [[2]](#ref2)

## Data Visualization


# Methodology

## Data Preprocessing

## Implementation

## Refinement


# Results

## Model Evaluation and Validation

## Justification


# Conclusion

## Reflection

## Improvement

While the results presented in the previous section show that optimized Multi-Output Regressors which use advanced algorithms like Ridge or SVD help to significantly improve the prediction accuracy,
it has to be mentioned that there is still a large optimization potential, considering that the best accuracy was only slightly above 50%.

However, regarding that the data set consisted only of 120 out of the 3500 matches from open data (and the whole StatsBomb data comprises even much more than those matches!), it is obvious that there exists a large potential 
by increasing the training set for the ML predictors.

Futhermore, we only used a quantitative approach by simply counting the number of events per event type. As there is also qualitative information included in the data (i.e, different results of the event types), this data could additionally be used to have an even more fine-grained feature set to further improve prediction accuracies.

# Acknowledges

Acknowledges go to StatsBomb for providing their open data set.
![StatsBomb](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/StatsBomb-logo_small.png?raw=true)

# References
1. <a name="ref1">[StatsBomb homepage](https://statsbomb.com/)</a>
2. <a name="ref2">[StatsBomb Open Data - public GitHub Repository](https://github.com/statsbomb/open-data)</a>
