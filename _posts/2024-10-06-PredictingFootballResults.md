# Predicting Results of Modern Football Matches using StatsBomb analysis data and advanced Multi-Output Machine Learning Regressors

![Football](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/football.jpg?raw=true)

# Project Definition

Football - also known as soccer in the USA - is one of the most famous sports in the world. In the last years, the amount of analysis data that are generated in modern football matches using cameras, drones or
other advanced technologies has significantly increased. StatsBomb [[1]](#ref1) has become one of the most famous analysis data providers in this area, delivering tons of analysis data of matches all over the planet. Besides data where the customer needs to pay for, there is also a large StatsBomb open data repository in GitHub available [[2]](#ref2) which includes more than ... matches from ... competitions, reaching from national competitions up to continental and global championships like the FIFA World Cup 2022.

An interesting question that arises with this large amount of data is the following:
*“Suppose you only have the analysis data of a specific match available, without knowing who know the match or if there was a draw. Is it possible to predict the winner with a high accuracy based on the analysis data and modern Machine Learning algorithms?”*

That´s exactly what is the focus of this project: We exploit the large open data set and use Machine Learning multi-output Classifiers and Regressors to predict the winner of a match, compared to a simple non-ML Classifier which randomly decides one of the possible three results - first team wins, second team wins, or there is a draw between the teams.
## Project Overview

## Problem Statement

## Metrics


# Analysis

## Data Exploration

Currently in October 2024, StatsBomb´s open data represents only a small, but yet comprehensive subset of the overall StatsBomb data. 

The data is provided as JSON files exported from the StatsBomb Data API, in the following structure:
- 21 Competition with 74 seasons
- Almost 3500 matches, spread among the above mentioned competitions and seasons
- Events and lineups for each of these matches, included additional StatsBomb 360 data for selected matches.

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


# Acknowledges

Acknowledges go to StatsBomb for providing their open data set.
![StatsBomb](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/StatsBomb-logo.png?raw=true&width=300)

# References
1. <a name="ref1">[StatsBomb homepage](https://statsbomb.com/)</a>
2. <a name="ref2">[StatsBomb Open Data - public GitHub Repository](https://github.com/statsbomb/open-data)</a>
