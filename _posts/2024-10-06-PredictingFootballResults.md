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
- <b>'win_home':</b> The home team scores more goals than the away team and wins
- <b>'win_none'</b>: The home team and the away team score exactly the same number of goals, i.e. the result is a draw
- <b>'win_away'</b>: The away team scores more goals than the home team and wins

In order to be processable by an algorithm and allow the observation of multiple matches at the same time, the 3 variables are in the following encoded as array of three columns with binary values, where the rows represent the matches and the columns represent the three possible outcomes (i.e., results) of the matches (1 if the result has entered, otherwise 0).

As an example, consider the following Fig. 1 which represents score data from five matches with the number of goals of the home team ('home_score') and the number of goals of the away team ('away_score') and the respective (5x3) result matrix including three row vectors [1, 0, 0] where the home team won (match_ids 3939976, 3939971, 3939969), one row vector with values [0, 0, 1] where the away team won (match_id 3939972), and one row with values [0, 1, 0] where the result was a draw (match_id 3939970).

![StatsBomb](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Result_Matrix.PNG?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 1: Result Matrix as output vector for prediction
</p>

The goal of a prediction algorithm is to minimize to calculate result matrices y_pred which are close to or, in the optimum case, represent exactly the result matrix y of the actual results.
Mathematically set, the sum of |y_pred - y| (row-wise calculated) should be as minimal as possible. If the prediction exactly matches the actual results, the sum of |y_pred - y| is zero.

## Metrics

To quantify the results of the classification algorithms, common classification metrics in Machine Learning [[3]](#ref3) are used, being combined in a classification report which is provided by the used scikit-learn framework [[4]](#ref4).
These comprise the following:
- <b>Precision score:</b> Precision is the ability of the classifier not to label a sample which actually is negative as positive. This metric is important here to find out if there are many false positives where a result was predicted (i.e., y_pred = 1) which actually did not occur (i.e., y_true = 0).
- <b>Recall score:</b> Contrary to precision, recall is the ability of the classifier to find all the positive samples. This metric is important to find out if a classifier has many undesired false negatives, where a result which actually occurred (i.e., y_true = 1) was not predicted by classifier (i.e., y_pred = 0)
- <b>F1 score:</b> This metric can be interpreted as a weighted harmonic mean of the precision and recall and takes care both of false positives and false negatives. It considers precision and recall with the same fraction and calculates as *2 * precision * recall / (precision + recall)*
- <b>Accuracy score:</b> The accuracy_score represents the fraction of correctly classified samples. Here, a sample represents a single row vector within the results matrix. As our main goal is to have the most matches between the actual and the predicted results (i.e., the number rows of the result matrix should be maximized), the accuracy (calculated on a row-wise manner) is considered t be the most important of the above metrics here

Mathematical details on these metrics can also be found at [[4]](#ref4).

As mentioned, our focus is on the highest possible accuracy, but the other metrics are also taken into considertion in the evaluations to additionally record if an algorithm performs good or bad in terms of false positives and/or false negatives.

# Analysis

## Data Exploration

Currently in October 2024, StatsBomb´s open data represents only a small, but yet comprehensive subset of the overall StatsBomb data. 

The data is provided as JSON files exported from the StatsBomb Data API, in the following structure:
- 21 competitions with 74 seasons
- Almost 3500 matches, spread among the above mentioned competitions and seasons
- Events and lineups for each of these matches, included additional StatsBomb 360 data for selected matches

A fine-grained documentation of the files, their formats and their contents is also included in the GitHub repository. [[2]](#ref2)

#### Competitions and Seasons

The single competition.json file includes an array of competition-season objects with the following properties relevant for this project:
- competition_id, competition_name (unique identifier of the competition and the respective name of this competition)
- season_id, season_name (unique identifier of the season and the respective name of this season)

The competition file was used as starting point for the data import in this project.

The full set of objects can be found [here](https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Competitions%20v2.0.0.pdf).

#### Matches

For every competition-season, there exists a match file which includes 1 to n matches of this season. Thus, there exist 74 match-specific json files.
Each match file contains an array of match objects, the most important objects for further investigation are the following:
- match_id (unique identifier for the match, which is required to request events, as the events files are named by this match_id)
- competition (mixed-type containing the unique ID for the competition and the respective name of the competition)
- season (mixed-type containing the unique ID for the season and the respective name of the season)
- home_team (mixed-type containing the unique ID for the home team and the respective name of this team)
- away_team (mixed-type containing the unique ID for the away team and the respective name of this team)
- home_score (final score of the home team)
- away_score (final score of the away team)

Please note that there is no explicit information on the result (i.e., the winner) of the match - this has to be retrieved manually, as described later on in the Data Preprocessing section.

The full set of objects can be found [here](https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Matches%20v3.0.0.pdf).

#### Events

The events represent the actual complexity of the data set as they involve each recorded event of a match.
Thus, there is one json file per match, summing up to almost 3500 matches. The number of recorded events per match obviously differs (since every game is different from each other), but is typically in the range of 2000 up to 5000 events. The average number of recorded events per match is around 3600 in this open data set, as investigated within this project.

Each events file contains an array containing information for both teams. Some elements have rather child elements (e.g., ID and name of an object) and some have even child arrays, leading to a deeper structure of this file type.

For each event, there are some basic information on the event like the period, minute and second when the event was recorded.

The heart of the complexity consists in the very fine-granular event types used in this file. 
Each event 

A complete list of the 119 event types (which are recorded team-specific, thus doubling this number in practice) can be found in the following figure, just to clarify the complexity of a modern football match.

![Event-types](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Event-Types.PNG?raw=true)

#### Lineups

The lineups represent the players of both teams that participate in a match, represented by an array of players for each team.
They only show up the unique identifiers, names, jersey numbers and countries of each player.

As there is no further information on the players included like their skills, value on the market, fitness or freshness, we do not further regard the lineups in the following.


## Data Visualization

As mentioned above, the event types in the events file contain the most important analysis data on a specific match, so we focus on them here.

In order to find out which event types are most important to become features for the supervised classification algorithms later on, we imported the data of a specific subset of the 3600 matches.
Regarding the immense amount of the dataset and the immense amount and memory consumption for processing the data, we focused on a small subset of matches for our observations, leaving the extension to a larger set for further work. Thus, the data visualizations in the following are created using the following data:
- All 116 provided matches from a specific season of 3 competitions - Spanish 'La Liga' 2019/2020, World Cup 2022 and Copa America 2024
- These matches include 417,221 events of all event types, which can be distinguished to the following numbers of event patterns:

![Pattern-types](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Pattern-types.PNG?raw=true)
- In order to score, a team typically needs to get adequately close to the goal. This is typically performed by 
  - conquering the ball in a dribble, duel, clearance, by a misleading pass of the oppenent or by getting a free kick, penalty kick or throw-in,
  - then (optionally) passing the ball between players to come closer to the door
  - and finally, performing a shot on the goal
  - to overcome the goalkeeper of the opponent team

We consider events that are related to the above mentioned game techniques to be most important for prediction the winner of a match based on analysis date.
Contrary to this, events like the start or end of a half, or an interception where the game is stopped should have a rather minimal impact on the result of a match.

As the event types start with a technique specific prefix (e.g., 'shot.' for all goal shot-related event types), we distinguished between to team for which event was relevant (home or away time) and created multiple heatmaps showing the influence of these team-specific event types on the result columns for the prediction, i.e., the three binary columns for the different match outcomes ('win_home', 'win_away', 'win_none'). [Fig. 2](#Fig2) shows these heatmaps.

![Fig2](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Heatmap_comb_trans.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 2: Heatmaps for specific event-types: A) clearance-, dribble- and duel-related events, B) foul-related events, C) pass-related events, D) shot-related events, E) goalkeeper-releated events
</p>

The heatmaps show at first glance that influence of the specific types on the result columns is widely spreaded: Some of the events have a positive impact on a specific result column, represented by a positive value and in a rather bright color, while other events have a negative impact with values below 0. More important, some events have a rather large impact on the result with values >0.2 or <-0.2, while others have a rather minimal impact on the results columns since their values are close to 0.

To clarify this, let us take a closer look on some specific cells of the heatmap E for some of the goalkeeper-related event types. [Fig. 3](#Fig3) displays them.

![Fig3](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Heatmap_goalkeeper_zoom.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 3: Part of Heatmap E, zoomed-in to some specific goalkeeper-related events
</p>

There are some interesting insights to gain from these cells:
- There is a significant different in terms of the team for which a goalkeeper-specific event was recorded: Taking the home team related goalkeeper events, it can be seen that there is a positive impact on the result 'win_home', while the impact on the result 'win_away' is negative and the impact on the result 'win_none' is rather marginal. Contrary to this, the away team related goalkeeper events' impact on 'win_home' and 'win_away' is switched. Thus, keeping both of these event types ('home_' and 'away_') strongly seems to make sense, as their appearences particularly help to predict the results where a specific team won.
- Moreover, some of the events have significantly higher impact than others. For instance, the 'technique.id' events have impacts with absolute values that are larger than 0.3 on 'win_home' or 'win_away', while the 'shot_saved_off_target' events have rather low impact, as all absolute values are below 0.15.
- As the influences of the features on the third result column that represents a draw with no winner are much lower than those on the home or away win columns, it may be harder to accurately predict draws than to predict wins of the home or the away team

# Methodology

## Data Preprocessing

#### Duplicates and Missing Values
The competitions and matches files could simply be imported using the standard Python3 json import function. They did not include any duplicates or missing rows. While there was no further preprocessing needed for the competitions set, the matches needed to be enriched by the event-specific data of the events files to have the feature set (X) and the output set (y) for the classification algorithms.

#### Feature Set Reduction
First of all, as we already discussed in the data visualizations section, some features seemed to have only a minor impact on the result columns. Thus, we only considered those features from the events dataset where at least one of the absolute cell values was above 0.2, i.e. |value > 0.2|. This leads to significantly reduced calculation and processing costs without a larger impact on the classification results. [Fig. 4](#Fig4) shows the resulting consolidated heatmap which only includes those 60 features (out of the original 238) with the largest influence on the result columns that were taken into consideration for the classification algorithms later on. As one can see, there is still a lot of complexity included in this immense feature set.

![Fig4](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Heatmap_mixed-type.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 4: Heatmap of combined-type, reduced feature set with 60 features
</p>

#### Feature Set - Quantitative Approach
For this project, we decided just to consider the quantitative amounts of the event types, meaning we count the amount of each event type per match and added an additional column for this feature holding the number of counts of this event. For some events, it may be possible to additionally exploit the actual qualitative value of an event, but we leave this for future work, as it would increase the computational complexity again (e.g., the goalkeeper´s body part for an event may be 'head', 'chest', 'both hands', 'left hand', 'right hand', 'left foot' or 'right foot', i.e. 7 possible values for only one event type!) when considering the whole feature set for this technique.

Immanent to this way of calculating and adding up the match statistics is that there are neither missing values nor duplicates in the feature set included, as each cell is calculated (even if an event did not occur, there is still a zero and not a NaN as cell value) exactly once per feature and team.

The feature set is imported event-type-wise from the events file into the matches dataframe within the function 'update_match_stats'.

#### Scaling
What was finally needed was scaling of the feature values: As the events occur in (partially) significantly different magnitudes, they need to be scaled before they can be used by Machine Learning algorithms, since otherwise, features would not be equally weighted by the algorithms. To scale the results in a positive range of floating-point numbers between zero (meaning this event type did not happen at all in this match) and one (meaning this event type happens most often in this match), we used scikit-learn´s standard MinMaxScaler [[5]](#ref5)

Summarizing, there are 179,136 out of the originally 417,221 events remaining for further analysis within this project.

## Implementation

For the implementation of the classification algorithm, a Jupyter Notebook was created relying on Python 3 within the Anaconda Distribution. The most important packages that were used comprise Pandas, Numpy, Matplotlib and Seaborn as well as several scikit-learn packages such as various classifier (e.g., MultiOutputClassifier, DecisionTreeClassifier or RandomForestClassifier) and regressory (e.g., MultiOutputRegressor, Ridge or GradientBoostingRegressor). The complete set of packages to be imported can be found in the initial import section of the provided Jupyter Notebook.

As we need to predict three binary result columns, a multi-output classifier is required which has to provide binary results per output, i.e. a vector [y1, y2, y3] with binary values for y1 (1 for 'win_home', otherwise 0), y2 (1 for 'win_none' i.e. a draw, otherwise 0) and y3 (1 for 'win_away', otherwise 0).

The following algorihms were implemented to predict the classification results:
- RandomClassifier: a simple classifier which randomly chooses among one of the three possible outcomes ([1, 0, 0], [0, 1, 0] or [0, 0, 1]). This accuracy, precision, recall and f1 results of this classifier serve as benchmark for the other classifiers.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with DecisionTrees [[7]](#ref7) estimator: Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with RandomForests [[8]](#ref8) estimator: the RandomForest algorithm represents an averaging algorithms based on randomized Decision Trees. A diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with LogisticRegression [[9]](#ref9) estimator: Logistic Regression is actually implemented as a linear model for classification rather than regression in terms of the scikit-learn/ML nomenclature.

## Refinement

While the three multi-output classifiers represent modern Machine Learning approaches, they face one big problem: As each value of the result column is calculated independently of each other, it may be possible that not always exactly one of the three outcomes is predicted with a 1. Instead, it may be possible that all predicted values are zero (i.e., [0, 0, 0] as result vector) or more than one predicted value is one (i.e., [1, 1, 0], [1, 0, 1], [0, 1, 1] or even [1, 1, 1]. To address this issue, we addditionally implemented three multi-output regressors from scikit-learn which predict not binary, but continuous values for the result vector. As the result values may still only be a 0 or a 1, the correction part of the algorithm takes the column which the largest predicted value and puts this value to 1, while the other two values ar put to 0. For instance, if the regressor´s result is [0.2, 0.6, 0.7], then the result vector is adapted to [0, 0, 1], since the value of the third column was the largest.

The following three regressors were implemented and optizimed using the algorithm described above in the Refinement section:
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with GradientBoostingRegressor [[8]](#ref8) as estimator: This estimator builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with Ridge [[11]](#ref11) as estimator: Ridge represents a Linear least squares with l2 regularization and minimizes the following objective function:
||y - Xw||^2_2 + alpha * ||w||^2_2
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with Stochastic Gradient Descent (SGD) [[12]](#ref12) as estimator: SGD represents a simple, yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression.

In summary, we compare the simple RandomClassifier with three ML multi-output classifiers and three ML multi-output regressor with corrected result vectors.

Finally, after having evaluated the results and decided in favor of a specific classifier or regressor (see Results section below), the corresponding classifier was organized within a ML pipeline. Furthermore, the extract, transform and load data parts were also extracted in an ETL pipeline which is executed prior to the ML pipeline. The pipeline guarantee automized, simple and flexible runs of the algorithms.

# Results

## Model Evaluation and Validation

#### [Evaluation Setup](#evalSetup)
We evaluated the implemented models presented in the last section using a set of automated tests with randomized, but determined generation of the training and test sets (using built-in scikit-learn function train_test_split) with different test ratios (i.e., the size of the test set compared to the total size). We decided to evaluate various test ratios to find out if the size of the training set for the ML classifier is sufficient, considering that only 116 matches were investigated.

After each test run comprising of the fitting each classifier separately to the train set, predicting of the results for each classifier in the test set and storing these results in an array, the random_state was increased by 1 to ensure a different train_test_split at the next run. For each test_ratio, there were 300 tests performed and then, the test_ratio was increased by 0.05 and the next test series were started. While the test series for a single test_ratio took around 132.5sec in average, the complete test series took around 19.9 minutes to be completed.

[[Fig. 5]](#fig5) gives an overview of the evaluation setup and the runtimes of the single test_ratio series.

![Fig5](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Evaluation-setup.PNG?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 5: Evaluation Setup
</p>

#### Classification Reports

The following sections show the evaluated classification metrics of the different classifiers that were gained over the test series described above.
While precision, recall and F1 scores are separated per possible result (home team wins, draw, away team wins), the accuracy is reported as average over all possible results in the classification report.

##### 1. RandomClassifier (RC)

The RandomClassifier (RC) produced accuracies in the anticipated range: The average accuracy is 0.34 which is close to 1/3, the expected when the result is randomly predicted.
The other metrics - precision, recall and F1 - are in the same range, except the precision score for 'win_home' that was around 25% above the accuracy score. 
As already mentioned, the RandomClassifier just serves as benchmark for the ML classifiers and regressors.

![Fig6](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_1_RandomClassifier_testRatios=0.1-0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 6: RandomClassifier evaluation results
</p>

##### 2. Multi-Output Classifier with DecisionTrees estimator (MOC_DecTree)

Decision Trees represent a rather simple ML approach. Furthermore, they suffer from the problem of non-unique classifcations discussed in the Refinement section. Because of this, it is not surprising that the average accuracy is quite poor with only 0.25. The classification matrix also shows that only the win of the home team could be predicted in an acceptable range that is higher than the one of the RandomClassifier. The correct prediction of the outcome 'win_away' was worse than with RC, and the prediction performance of draws was particularly poor.

![Fig7](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_2_MOC_DecTree_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 7: MOC_DecTree evaluation results
</p>


##### 3. Multi-Output Classifier with RandomForests estimator (MOC_RandomForests)


![Fig8](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_3_MOC_RandomForests_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 8: MOC_RandomForests evaluation results
</p>

##### 4. Multi-Output Classifier with LogisticRegression estimator (MOC_LogisticRegression)


![Fig9](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_4_MOC_LogisticRegression_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 9: MOC_LogisticRegression evaluation results
</p>


##### 5. Multi-Output Regressor with GradientBoosting estimator (MOR_GradBoo)


![Fig10](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_5_MOR_GradBoost_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 10: MOC_GradBoost evaluation results
</p>

##### 6. Multi-Output Regressor with Ridge estimator (MOR_Ridge)

![Fig11](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_6_MOR_Ridge_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 11: MOR_Ridge evaluation results
</p>

##### 7. Multi-Output Regressor with SGD estimator (MOR_SGD)

![Fig12](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_7_MOR_SGD_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 12: MOR_SGD evaluation results
</p>

#### Accuracy: Is the training set big enough?
The evaluation results presented in the last section show that the overall accuracy over the complete test series can significantly be increased compared to random choice by using a multi-output regressor with an estimator like Ridge or SGD. However, one question that may arise when looking on the number of investigated matches is: Is this amount of matches and events sufficient for an adequate use of Machine Learning classifiers? To find an answer on this question, we decided to vary the test_ratio and change the amount of training data in terms of matches for the classifier: While the training set is rather large when test_ratio = 0.1 - there still remain 108 out of 120 matches for training (see Evaluation Setup) - this number is significantly lower when test_ratio = 0.5 - then, there are only 60 out of 120 matches left for training. By starting with test_ratio = 0.1 and then taking small increases of 0.05 for the test_ratio up to 0.5, the accuracies per classifier were recorded and are shown in Fig. ...

TODO: Add accuracy graph

## Justification


# Conclusion

## Reflection

## Improvement

While the results presented in the previous section show that optimized Multi-Output Regressors which use advanced algorithms like Ridge or SVD help to significantly improve the prediction accuracy,
it has to be mentioned that there is still a large optimization potential, considering that the best accuracy was only slightly above 50%.

However, regarding that the data set consisted only of 120 out of the 3500 matches from open data (and the whole StatsBomb data comprises even much more than those matches!), it is obvious that there exists a large potential by increasing the training set for the ML predictors.

Futhermore, we only used a quantitative approach by simply counting the number of events per event type. As there is also qualitative information included in the data (i.e, different results of the event types), this data could additionally be used to have an even more fine-grained feature set to further improve prediction accuracies. 

Taking into account the timely manner and sequence in which specific events occur may also help to further increase the performance of the classifiers, but additionally increase the complexity of the model, making it much more time and resource consuming. Thus, one has to consider if maximum accuracy should be aspired under each circumstances without considering consumption costs of the data processings and calculations.

# Acknowledges

Acknowledges go to StatsBomb for providing their open data set.
![StatsBomb](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/StatsBomb-logo_small.png?raw=true)

# References
1. <a name="ref1">[StatsBomb homepage](https://statsbomb.com/)</a>
2. <a name="ref2">[StatsBomb Open Data - public GitHub Repository](https://github.com/statsbomb/open-data)</a>
3. <a name="ref3">[Performance Metrics in Machine Learning | Complete Guide]](https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide)</a>
4. <a name="ref4">[scikit-learn Classification Report Metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)</a>
5. <a name="ref5">[scikit-learn MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)</a>
6. <a name="ref6">[scikit-learn MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)</a>
7. <a name="ref7">[Decision Trees in scikit-learn](https://scikit-learn.org/stable/modules/tree.html)</a>
8. <a name="ref8">[RandomForests, GradientBoosting and others in scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html)</a>
9. <a name="ref9">[Logistic Regression in scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)</a>
10. <a name="ref10">[scikit-learn MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)</a>
11. <a name="ref11">[Ridge in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)</a>
12. <a name="ref12">[StochasticGradientBoost in scikit-learn](https://scikit-learn.org/stable/modules/sgd.html)</a>
