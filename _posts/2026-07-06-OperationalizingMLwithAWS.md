# An Operational End-2-End Approach for Predicting Match Results in Modern Football with AWS SageMaker

![](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2026-07_MLE-Cap/football2.png?raw=true)

# Project Definition

## Project Overview

Football — known as soccer in the United States where currently the World Cup 2026 is ongoing — is one of the most popular sports worldwide. Over the past decade, the volume of analytical data generated during professional football matches has increased dramatically due to advances in technologies such as high-resolution cameras, drones, wearable sensors, and automated tracking systems. One of the leading providers of football analytics data is StatsBomb [[1]](#ref1), which offers extensive event and performance data from matches around the world. In addition to its commercial products, StatsBomb maintains a large open-data repository on GitHub [[2]](#ref2). This repository currently (in July 2026) contains free data from many thousand matches across 190+ competitions, ranging from domestic leagues to major continental and international tournaments, including the 2022 FIFA World Cup as well as the latest continental championships, like the 2025 UEFA Women´s Euro Cup.

The rapid growth in the availability of football analytics data raises an interesting research question:

*“Suppose the detailed event data of a football match is available, but the final result is unknown. Based solely on historical match data for which the outcomes are known, is it possible to accurately predict the result of the match using modern Machine Learning (ML) techniques?“*

This question forms the foundation of the present project. Leveraging the extensive StatsBomb open-data collection, various Machine Learning models—including multi-output classifiers and regressors—are trained to predict the outcome of a football match. Possible outcomes are defined as a win for the home team, a win for the away team, or a draw. The performance of these ML-based approaches is then compared to a simple baseline classifier that selects one of the three possible outcomes at random. The objective is to evaluate whether modern Machine Learning methods can significantly outperform such a naïve approach and achieve meaningful predictive accuracy based solely on match event data.

While previous work from October 2024 [[2b]](#ref2b) focused solely on the fitting of classifiers to predict the match results, we follow a complete end-2-end approach here in order to operationalize the prediction process using AWS SageMaker invocation endpoints and AWS Lambda. Moreover, we optimize the prediction results by extending the data scope for training and testing from originally 150 matches (in 2024) up to 1000 matches (now). As you will see, this further improves the prediction results.

## Problem Statement

The objective of this project is to predict the outcome of a football match without directly using the final score information, i.e., without exploiting the number of goals scored by either team.
In football, each match is contested by two teams: the home team and the away team. A team wins the match if it scores more goals than its opponent during regular playing time (90 minutes). If both teams score the same number of goals, the match ends in a draw. Although some knockout competitions continue with extra time and penalty shootouts when the score is level after regular time, these scenarios are not considered in the scope of this project. Only the result at the end of regular playing time is taken into account. Consequently, every match can result in exactly one of the following three outcomes:

- win_home: The home team scores more goals than the away team and wins the match.
- win_none: Both teams score the same number of goals, resulting in a draw.
- win_away: The away team scores more goals than the home team and wins the match.

To enable efficient processing by Machine Learning algorithms, the match outcomes are encoded using a three-dimensional binary output vector. Each row represents a single match, while each column corresponds to one of the three possible outcomes. A value of 1 indicates the actual outcome of the match, whereas the remaining entries are set to 0.
The resulting target representation is defined as follows:

| Match Outcome | Encoded Vector |
|---------------|---------------|
| Home team wins | [1, 0, 0] |
| Draw | [0, 1, 0] |
| Away team wins | [0, 0, 1] |

Stripping down the encoded vector to single variables would lead to the following representation:

| Match Outcome | win_home | win_none | win_away |
|---------------|----------|----------|----------|
| Home team wins | 1 | 0 | 0 |
| Draw | 0 | 1 | 0 |
| Away team wins | 0 | 0 | 1 |


Fig. 1 represents exemplary score data from five matches with the number of goals of the home team ('home_score') and the number of goals of the away team ('away_score') and the respective (5x3) result matrix including three row vectors [1, 0, 0] where the home team won (match_ids 3939976, 3939971, 3939969), one row vector with values [0, 0, 1] where the away team won (match_id 3939972), and one row with values [0, 1, 0] where the result was a draw (match_id 3939970).

![StatsBomb](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Result_Matrix.PNG?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 1: Result Matrix as output vector for prediction
</p>

The goal of a prediction algorithm is to predict result matrices y_pred which are close to or, in the optimum case, exactly the result matrix y_true of the actual results.
Mathematically set, the sum of |y_pred - y| (row-wise calculated) should be as minimal as possible. If the prediction exactly matches the actual results, the sum of |y_pred - y| is zero.

## Evaluation Metrics

To quantitatively assess the performance of the investigated Machine Learning models, several standard classification metrics are used. These metrics are provided by the scikit-learn framework and summarized in a classification report #ref3, #ref4.

For each target class, the prediction results can be represented by a confusion matrix consisting of the following components:

- **True Positives (TP):** Positive samples that were correctly classified.
- **True Negatives (TN):** Negative samples that were correctly classified.
- **False Positives (FP):** Negative samples that were incorrectly classified as positive.
- **False Negatives (FN):** Positive samples that were incorrectly classified as negative.

Based on these quantities, several evaluation metrics can be defined.

### Precision

Precision measures the proportion of predicted positive samples that are actually positive and is therefore an indicator of how reliable positive predictions are.

$$
Precision = \frac{TP}{TP + FP}
$$

A high precision score indicates that the classifier produces relatively few false positive predictions. In the context of football match prediction, this metric is useful for identifying whether a predicted outcome (e.g., a home-team win) was frequently predicted incorrectly. 

### Recall

Recall, also referred to as sensitivity or true positive rate, measures the proportion of actual positive samples that are correctly identified by the classifier.

$$
Recall = \frac{TP}{TP + FN}
$$

A high recall score indicates that the classifier successfully identifies most of the matches belonging to a specific outcome class. Low recall values suggest that many actual occurrences of a result are missed by the model.

### F1 Score

The F1 score combines precision and recall into a single metric by calculating their harmonic mean:

$$
F_1 = \frac{2 \cdot Precision \cdot Recall}
           {Precision + Recall}
$$

The F1 score is particularly useful when class distributions are imbalanced, as it simultaneously penalizes false positives and false negatives. A high F1 score can only be achieved when both precision and recall are high.

### Accuracy

Accuracy measures the proportion of correctly classified samples among all evaluated samples.

$$
Accuracy = \frac{TP + TN}
                 {TP + TN + FP + FN}
$$

In this project, a sample corresponds to a single football match represented by a target vector of length three. A prediction is considered correct if the predicted outcome vector exactly matches the actual outcome vector.

Since the primary objective of the project is to maximize the number of correctly predicted match outcomes, accuracy is considered the most important evaluation metric. However, accuracy alone may not provide a complete picture, especially when the classes are not equally distributed. Therefore, precision, recall, and F1 score are evaluated as complementary metrics.

### Macro and Weighted Averages

Because the prediction problem consists of three possible match outcomes (`win_home`, `win_none`, and `win_away`), the classification report additionally provides aggregated metrics:

- **Macro Average:** Arithmetic mean of the metric across all classes. Each class contributes equally, regardless of its frequency.

$$
MacroAvg = \frac{1}{C}
            \sum_{i=1}^{C} Metric_i
$$

where \(C\) denotes the number of classes.

- **Weighted Average:** Average of the metric weighted by the number of samples belonging to each class.

$$
WeightedAvg =
\frac{\sum_{i=1}^{C} n_i \cdot Metric_i}
     {\sum_{i=1}^{C} n_i}
$$

where \(n_i\) represents the number of samples of class \(i\).

These aggregated metrics provide additional insights into the model's performance, particularly when one class occurs more frequently than the others.

### Evaluation Objective

Although all previously introduced evaluation metrics are considered during model assessment, the primary objective of this project is to maximize the overall prediction accuracy. Since the task consists of predicting the correct outcome of a football match, the proportion of correctly classified matches represents the most intuitive and practically relevant measure of model performance.

Nevertheless, accuracy alone does not provide a complete assessment of a classifier's quality. In particular, when the distribution of match outcomes is imbalanced, a model may achieve a relatively high overall accuracy while performing poorly on less frequent classes. For this reason, the complementary metrics precision, recall, and F1 score are also evaluated. These metrics provide additional insight into the classifier's behavior with respect to false positive and false negative predictions and enable a more comprehensive assessment of model performance.

Special attention is paid to the individual predictive performance of the three target classes (`win_home`, `win_none`, and `win_away`). While aggregated metrics summarize the overall model quality, class-specific metrics reveal whether certain match outcomes are systematically easier or more difficult to predict. Such analyses are particularly important for identifying potential biases, class imbalance effects, or weaknesses of individual Machine Learning models.

Consequently, the final evaluation of each model is based not only on its overall accuracy, but also on its ability to consistently and reliably identify all three possible match outcomes.

# Analysis

## Dataset Overview

As of July 2026, the StatsBomb Open Data repository represents only a subset of StatsBomb's complete commercial data offering. Nevertheless, it provides a comprehensive and diverse collection of football match data that is well suited for data analytics and Machine Learning applications.

The dataset is publicly available through the StatsBomb Open Data GitHub repository [[2]](#ref2) and is provided as a collection of JSON files exported from the StatsBomb Data API. The repository is organized hierarchically and currently contains:

- **21 football competitions**
- **74 individual seasons**
- **Approximately 3,500 matches** distributed across the available competitions and seasons
- **Event data and lineup information** for each match
- **StatsBomb 360 data** for a subset of matches, providing additional contextual information about player positioning and surrounding match situations

The competitions included in the repository cover a wide range of football tournaments, from domestic league competitions to major continental and international championships. Consequently, the dataset captures a broad spectrum of playing styles, tactical approaches, and competitive levels.

For each match, the event data contains a chronological sequence of football actions such as passes, shots, dribbles, duels, interceptions, fouls, and goalkeeper actions. These events form the primary data source used in this project to derive match-specific features for the Machine Learning models.

The repository additionally provides extensive documentation describing the structure, content, and semantics of the available JSON files. This documentation includes detailed explanations of event types, attributes, nested data structures, and relationships between the different datasets, thereby facilitating efficient data preprocessing and feature engineering.

### Competitions and Seasons

The entry point to the StatsBomb Open Data repository is the `competitions.json` file. This file contains an array of competition-season objects that provide metadata about the available football competitions and seasons.

For the purposes of this project, the most relevant attributes are:

- `competition_id` and `competition_name`, which uniquely identify a competition and provide a human-readable description.
- `season_id` and `season_name`, which uniquely identify a season within a competition.

The competition dataset serves as the starting point of the data ingestion process. It is used to identify all available competition-season combinations and subsequently retrieve the corresponding match data.

A complete description of all available fields can be found [here](https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Competitions%20v2.0.0.pdf) in the official StatsBomb Open Data documentation.

---

### Matches

For each competition-season combination, the repository contains a dedicated match file comprising all matches played during the respective season. Consequently, the dataset includes a total of 74 match files, corresponding to the 74 available seasons.

Each match file consists of an array of match objects. The most relevant attributes used in this project are:

- `match_id`: Unique identifier of the match. This identifier is particularly important as it is used to locate the corresponding events and lineup files.
- `competition`: Nested object containing the competition identifier and competition name.
- `season`: Nested object containing the season identifier and season name.
- `home_team`: Nested object containing information about the home team.
- `away_team`: Nested object containing information about the away team.
- `home_score`: Number of goals scored by the home team during regular playing time.
- `away_score`: Number of goals scored by the away team during regular playing time.

Although the final scores are included in the dataset, no explicit match outcome variable is provided. Consequently, the target variables (`win_home`, `win_none`, and `win_away`) must be derived during the data preprocessing stage by comparing the final scores of both teams.

The match dataset therefore serves two primary purposes within this project:

1. Providing metadata about the participating teams and the competition context.
2. Generating the target variables used for supervised Machine Learning.

A complete description of all available match attributes can be found [here](https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Matches%20v3.0.0.pdf) in the official StatsBomb Open Data documentation.

---

### Events

The event data represents the most important and by far the most complex part of the entire dataset. While the competition and match files primarily contain metadata, the event files contain the actual football actions that occurred during a match.

For every match, a dedicated event file exists. Given that the repository currently contains approximately 3,500 matches, the dataset also contains approximately 3,500 event files.

The number of recorded events differs from match to match depending on the flow and intensity of the game. Within the dataset analyzed in this project, matches typically contain between 2,000 and 5,000 recorded events, with an average of approximately 3,600 events per match.

Each event file consists of an ordered sequence of event objects describing actions performed by both participating teams. In addition to basic event information such as:

- Match period
- Timestamp
- Minute
- Second
- Team
- Player

many event types contain nested attributes and hierarchical substructures. Examples include shot information, pass characteristics, duel outcomes, goalkeeper actions, and positional information. Some attributes are represented as nested objects containing identifiers and descriptive names, while others contain arrays of additional contextual data.

The richness of the event data is primarily driven by the large number of supported event types. StatsBomb records a highly detailed representation of football matches covering offensive, defensive, tactical, and transitional actions. Examples include:

- Passes
- Shots
- Carries
- Dribbles
- Duels
- Ball recoveries
- Interceptions
- Fouls
- Clearances
- Goalkeeper actions

and many more.

Overall, the repository contains 119 distinct event attributes and event-related feature categories. Since these events are recorded separately for both teams, the effective number of potentially relevant match characteristics is substantially larger. Consequently, the event data forms the foundation for the feature engineering process and provides the predictive variables used by the Machine Learning models developed in this project.

A complete overview of all events' files 150 structures including the ~120 event types is shown below. Every single event is attached to one specific teams, i.e. the home team _or_ the away team.

```python
[
 'id',
 'index',
 'period',
 'timestamp',
 'minute',
 'second',
 'possession',
 'duration',
 'type.id',
 'type.name',
 'possession_team.id',
 'possession_team.name',
 'play_pattern.id',
 'play_pattern.name',
 'team.id',
 'team.name',
 'tactics.formation',
 'tactics.lineup',
 'related_events',
 'location',
 'player.id',
 'player.name',
 'position.id',
 'position.name',
 'pass.recipient.id',
 'pass.recipient.name',
 'pass.length',
 'pass.angle',
 'pass.height.id',
 'pass.height.name',
 'pass.end_location',
 'pass.body_part.id',
 'pass.body_part.name',
 'pass.type.id',
 'pass.type.name',
 'carry.end_location',
 'under_pressure',
 'pass.outcome.id',
 'pass.outcome.name',
 'ball_receipt.outcome.id',
 'ball_receipt.outcome.name',
 'counterpress',
 'interception.outcome.id',
 'interception.outcome.name',
 'off_camera',
 'pass.switch',
 'pass.aerial_won',
 'duel.type.id',
 'duel.type.name',
 'duel.outcome.id',
 'duel.outcome.name',
 'dribble.outcome.id',
 'dribble.outcome.name',
 'pass.cross',
 'goalkeeper.type.id',
 'goalkeeper.type.name',
 'goalkeeper.outcome.id',
 'goalkeeper.outcome.name',
 'clearance.left_foot',
 'clearance.body_part.id',
 'clearance.body_part.name',
 'foul_committed.advantage',
 'foul_won.advantage',
 'clearance.head',
 'clearance.aerial_won',
 'foul_won.defensive',
 'clearance.right_foot',
 'ball_recovery.recovery_failure',
 'pass.assisted_shot_id',
 'pass.shot_assist',
 'shot.statsbomb_xg',
 'shot.end_location',
 'shot.key_pass_id',
 'shot.type.id',
 'shot.type.name',
 'shot.outcome.id',
 'shot.outcome.name',
 'shot.technique.id',
 'shot.technique.name',
 'shot.body_part.id',
 'shot.body_part.name',
 'shot.freeze_frame',
 'goalkeeper.end_location',
 'goalkeeper.position.id',
 'goalkeeper.position.name',
 'out',
 'dribble.nutmeg',
 'foul_committed.card.id',
 'foul_committed.card.name',
 'block.offensive',
 'pass.through_ball',
 'pass.technique.id',
 'pass.technique.name',
 'shot.first_time',
 'ball_recovery.offensive',
 'substitution.outcome.id',
 'substitution.outcome.name',
 'substitution.replacement.id',
 'substitution.replacement.name',
 'foul_committed.offensive',
 'pass.inswinging',
 'pass.straight',
 'pass.outswinging',
 'bad_behaviour.card.id',
 'bad_behaviour.card.name',
 'pass.goal_assist',
 'shot.aerial_won',
 'goalkeeper.body_part.id',
 'goalkeeper.body_part.name',
 'goalkeeper.technique.id',
 'goalkeeper.technique.name',
 'clearance.other',
 'shot.deflected',
 'block.deflection',
 'match_id',
 'pass.cut_back',
 'miscontrol.aerial_won',
 'shot.saved_off_target',
 'goalkeeper.shot_saved_off_target',
 'pass.no_touch',
 'dribble.overrun',
 'foul_committed.type.id',
 'foul_committed.type.name',
 'injury_stoppage.in_chain',
 'shot.one_on_one',
 'shot.saved_to_post',
 'goalkeeper.shot_saved_to_post',
 '50_50.outcome.id',
 '50_50.outcome.name',
 'pass.deflected',
 'goalkeeper.punched_out',
 'pass.miscommunication',
 'foul_committed.penalty',
 'foul_won.penalty',
 'shot.open_goal',
 'shot.redirect',
 'pass.backheel',
 'dribble.no_touch',
 'block.save_block',
 'goalkeeper.penalty_saved_to_post',
 'goalkeeper.lost_in_play',
 'goalkeeper.success_in_play',
 'half_start.late_video_start',
 'shot.follows_dribble',
 'goalkeeper.success_out',
 'goalkeeper.lost_out',
 'player_off.permanent',
 'half_end.early_video_end',
 'goalkeeper.saved_to_post',
 'shot.kick_off'
]
```

### Lineups

For each match, the repository additionally provides lineup information describing the participating players of both teams.

The lineup files contain player-specific metadata such as:

- Player identifier
- Player name
- Jersey number
- Nationality

While these files provide information about which players participated in a match, they do not contain additional performance-related attributes such as player ratings, market values, fitness indicators, recent form, or physical condition.

Since the primary objective of this project is to predict match outcomes solely on the basis of event-derived match statistics, the lineup information is not incorporated into the feature engineering process. Consequently, the lineup data is excluded from further analysis.

## Data Visualization

As mentioned above, the event types in the events file contain the most important analysis data on a specific match, so we focus on them here.

In order to find out which event types are most important to become features for the supervised classification algorithms later on, we imported the data of a specific subset of the 3600 matches.
Regarding the immense amount of the dataset and the immense amount and memory consumption for processing the data, we focused on a small subset of matches for our observations, leaving the extension to a larger set for further work. Thus, the data visualizations in the following are created using the following data:
- All 116 provided matches in the open data set from specific seasons of 3 competitions - Spanish 'La Liga' 2019/2020, Euro Cup 2024 and Copa America 2024
- These matches include 417,221 events of all event types, which can be distinguished by the following numbers of event patterns:

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

The heatmaps show at first glance that the influence of the specific types on the result columns is widely spread: 
- Some of the events have a positive impact on a specific result column, represented by a positive value and in a rather bright color, while other events have a negative impact with values below 0.
- Moreover, some events have a rather large impact on the result with values >0.2 or <-0.2, while others have a rather minimal impact on the results columns since their values are close to 0.

To clarify this, let us take a closer look on some specific cells of the heatmap E for some of the goalkeeper-related event types. [Fig. 3](#Fig3) displays them.

![Fig3](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Heatmap_goalkeeper_zoom.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 3: Part of Heatmap E, zoomed-in to some specific goalkeeper-related events
</p>

There are some interesting insights to gain from these cells:
- There is a significant difference in terms of the team for which a goalkeeper-specific event was recorded: Taking the home team related goalkeeper events, it can be seen that there is a positive impact on the result 'win_home', while the impact on the result 'win_away' is negative and the impact on the result 'win_none' is rather marginal. Contrary to this, the away team related goalkeeper events' impact on 'win_home' and 'win_away' is switched. Thus, keeping both of these event types ('home_' and 'away_') strongly seems to make sense, as their appearences particularly help to predict the results where a specific team won.
- Moreover, some of the events have significantly higher impact than others. For instance, the 'technique.id' events have impacts with absolute values that are larger than 0.3 on 'win_home' or 'win_away', while the 'shot_saved_off_target' events have rather low impact, as all absolute values are below 0.15.
- As the influences of the features on the third result column that represents a draw with no winner are much lower than those on the home or away win columns, it may be harder to accurately predict draws than to predict wins of the home or the away team

# Methodology

## Data Preprocessing

#### Duplicates and Missing Values
The competitions and matches files could simply be imported using the standard Python3 json import function. They did not include any duplicates or missing rows. While there was no further preprocessing needed for the competitions set, the matches needed to be enriched by the event-specific data of the events files to have the feature set (X) and the output set (y) for the classification algorithms.

#### Feature Set Reduction
First of all, as we already discussed in the data visualizations section, some features seemed to have only a minor impact on the result columns. Thus, we only considered those features from the events dataset where at least one of the absolute cell values was above 0.2, i.e. |value > 0.2|. This leads to significantly reduced calculation and processing costs without a larger impact on the classification results. [Fig. 4](#Fig4) shows the resulting consolidated heatmap which only includes those 60 features (out of the original 2*119 = 238) with the largest influence on the result columns that were taken into consideration for the classification algorithms later on. As one can see, there is still a lot of complexity included in this immense feature set.

![Fig4](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Heatmap_mixed-type.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 4: Heatmap of combined-type, reduced feature set with 60 features
</p>

#### Feature Set - Quantitative Approach
For this project, we decided just to consider the quantitative amounts of the event types, meaning we counted the amount of each event type per match and added an additional column for this feature holding the number of counts of this event. For some events, it may be possible to additionally exploit the actual qualitative value of an event, but we leave this for future work, as it would increase the computational complexity again (e.g., the goalkeeper´s body part for an event may be 'head', 'chest', 'both hands', 'left hand', 'right hand', 'left foot' or 'right foot', i.e. 7 possible values for only one event type!) when considering the whole feature set for this technique.

Immanent to this way of calculating and adding up the match statistics is that there are neither missing values nor duplicates in the feature set included, as each cell is calculated (even if an event did not occur, there is still a zero and not a NaN as cell value) exactly once per feature and team.

The feature set wass imported event type-wise from the events file into the matches dataframe within the function 'update_match_stats'.

#### Scaling
What was finally needed was scaling of the feature values: As the events occur in (partially) significantly different magnitudes, they need to be scaled before they can be used by Machine Learning algorithms, since otherwise, features would not be equally weighted by the algorithms. To scale the results in a positive range of floating-point numbers between zero (meaning this event type did not happen at all in this match) and one (meaning this event type happened most often in this match), we used scikit-learn´s standard MinMaxScaler [[5]](#ref5)

Summarizing, there are 179,136 out of the originally 417,221 events remaining for further analysis within this project.

## Implementation

For the implementation of the classification algorithm, a Jupyter Notebook was created relying on Python 3 within the Anaconda Distribution. The most important packages that were used comprise Pandas, Numpy, Matplotlib and Seaborn as well as several scikit-learn packages such as various classifier (e.g., MultiOutputClassifier, DecisionTreeClassifier or RandomForestClassifier) and regressors (e.g., MultiOutputRegressor, Ridge or GradientBoostingRegressor). The complete set of packages to be imported can be found in the initial import section of the provided Jupyter Notebook.

As we need to predict three binary result columns, a multi-output classifier is required which has to provide binary results per output, i.e. a vector [y1, y2, y3] with binary values for y1 (1 for 'win_home', otherwise 0), y2 (1 for 'win_none' i.e. a draw, otherwise 0) and y3 (1 for 'win_away', otherwise 0).

The following algorihms were implemented to predict the classification results:
- RandomClassifier: a simple classifier which randomly chooses among one of the three possible outcomes ([1, 0, 0], [0, 1, 0] or [0, 0, 1]). This accuracy, precision, recall and f1 results of this classifier serve as benchmark for the other classifiers.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with DecisionTrees [[7]](#ref7) estimator: Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with RandomForests [[8]](#ref8) estimator: the RandomForest algorithm represents an averaging algorithms based on randomized Decision Trees. A diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with LogisticRegression [[9]](#ref9) estimator: Logistic Regression is actually implemented as a linear model for classification rather than regression in terms of the scikit-learn/ML nomenclature.

## Refinement

While the three multi-output classifiers introduced above represent modern Machine Learning approaches, they face one big problem: As each value of the result column is calculated independently from each other, it may be possible that the outcome not always predicts exactly one of the 3 possible results with a 1. Instead, it may be possible that all predicted values are zero (i.e., [0, 0, 0] as result vector), or that more than one predicted value is one (i.e., [1, 1, 0], [1, 0, 1], [0, 1, 1] or even [1, 1, 1]). To address this issue, we addditionally implemented three multi-output regressors from scikit-learn which predict not binary, but continuous values for the result vector. As the result values may still only be a 0 or a 1, the correction part of the algorithm takes the column which the largest predicted value and puts this value to 1, while the other two values ar put to 0. For instance, if the regressor´s result is [0.2, 0.6, 0.7], then the result vector is adapted to [0, 0, 1], since the value of the third column was the largest.

The following three regressors were implemented and optizimed using the algorithm described above:
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with GradientBoostingRegressor [[8]](#ref8) as estimator: This estimator builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with Ridge [[11]](#ref11) as estimator: Ridge represents a Linear least squares with L2 regularization and minimizes the following objective function:
||y - Xw||^2_2 + alpha * ||w||^2_2
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with Stochastic Gradient Descent (SGD) [[12]](#ref12) as estimator: SGD represents a simple, yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression.

In summary, we compare the naive RandomClassifier with three ML multi-output classifiers and three ML multi-output regressor with corrected result vectors.

Finally, after having evaluated the results and decided in favor of a specific classifier or regressor (see Results section below), the corresponding classifier can easily be organized within an ML pipeline. Furthermore, the extract (-> import from StatsBomb open data), transform (-> data preprocessing) and load (-> store processed data) parts could also be extracted in an ETL pipeline which would be executed prior to the ML pipeline. The pipelines would guarantee automated, simple and flexible runs of the algorithms.

# Results

## Model Evaluation and Validation

#### Evaluation Setup
We evaluated the implemented models presented in the last section using a set of automated tests with randomized, but determined generation of the training and test sets (using built-in scikit-learn function train_test_split) with different test ratios (i.e., the size of the test set compared to the total size). We decided to evaluate various test ratios to find out if the size of the training set for the ML classifier is sufficient, considering that only 116 matches were investigated.

After each test run which comprises of the fitting of each classifier separately on the training set, predicting the results for each classifier on the test set and storing these results in an array, the random_state was increased by 1 to ensure differing train_test_splits and algorithm executions at the next run. For each test_ratio, there were 300 tests performed and then, the test_ratio was increased by 0.05 and the next test series was started. While the test series for a single test_ratio took around 132.5sec in average, the complete test series took around 19.9 minutes to be completed.

[[Fig. 5]](#fig5) gives an overview of the evaluation setup and the runtimes of the single test_ratio series.

![Fig5](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Evaluation-setup.PNG?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 5: Evaluation Setup
</p>

#### Classification Reports

The following sections show the evaluated classification metrics of the different classifiers that were gained over the test series described above.
While precision, recall and F1 scores are separated per possible result outcome ('win_home', 'win_none', 'win_away'), the accuracy is reported as average over all possible result outcomes in the classification report.

##### 1. RandomClassifier (RC)

The RandomClassifier (RC) produced accuracies in the anticipated range: The average accuracy was 0.34 which is close to 1/3, the expected when the result is randomly predicted.
The other metrics - precision, recall and F1 - were in the same range, except the precision score for 'win_home' that was around 25% above the accuracy score. 
As already mentioned, the RandomClassifier just serves as benchmark for the ML classifiers and regressors.

![Fig6](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_1_RandomClassifier_testRatios=0.1-0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 6: RandomClassifier evaluation results
</p>

##### 2. Multi-Output Classifier with DecisionTrees estimator (MOC_DecTree)

Decision Trees represent a rather simple ML approach. Furthermore, they suffer from the problem of non-unique classifcations discussed in the Refinement section. Because of this, it is not surprising that the average accuracy is quite poor with only 0.25. The classification matrix also shows that only the win of the home team could be predicted in an acceptable range that is higher than the one of the RandomClassifier. The correct prediction of the outcome 'win_away' was worse than with RC, and the prediction performance of 'draws' was particularly poor.

![Fig7](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_2_MOC_DecTree_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 7: MOC_DecTree evaluation results
</p>


##### 3. Multi-Output Classifier with RandomForests estimator (MOC_RandomForests)

Using RandomForests as estimator increased the accuracy score slightly to 0.33, which is almost the value of RC. The averaging and randomization of decision trees seems to also improve the results of the other metrics. However, this only holds for the result 'win_home', while the metric scores of the other results 'draw' and 'win_away' were very poor. Like the other two multi-output classifiers, MOC_RandomForests also suffers from the non-unique classication matrix, explaining the rather poor overall results.

![Fig8](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_3_MOC_RandomForests_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 8: MOC_RandomForests evaluation results
</p>

##### 4. Multi-Output Classifier with LogisticRegression estimator (MOC_LogisticRegression)

As the above mentioned multi-output classifier, Logistic Regression as a linear classifier also suffers from the non-uniqueness of some classifications. However, since Logistic Regression typically is advanced compared to Decision Trees or Random Forests, its classification accuracy of 0.37 is slightly higher than theirs or the RC accuracy. While precision, recall and, consequently, F1 scores are fine for 'win_home', the scores of 'win_away' are also better than those of the previously discussed algorithms. However, a 'draw' is still very hard to be predicted by Logistic Regression, especially regarding the very poor score of only 0.07 for recall.

![Fig9](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_4_MOC_LogisticRegression_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 9: MOC_LogisticRegression evaluation results
</p>


##### 5. Multi-Output Regressor with GradientBoosting estimator (MOR_GradBoo)

The Gradient Boosting-based multi-outpot regressor represents an advanced algorithm over the simple classifier, as it predicts non-binary, continuous prediction values whose row maximum can simply be determined and the corresponding predicted vector set to 1 only for this result, while the others become zero. Thus, non-unique classification are not possible here. This leads to a much better accuracy of 0.48 which is around 41% higher than the one of RC. The other metric are also improved compared to the previously discussed classifiers, especially considering that all scores for 'win_home' are at least 0.6, 'win_away' scores are around 0.4 and 'draw' scores are around 0.3.

![Fig10](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_5_MOR_GradBoost_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 10: MOR_GradBoost evaluation results
</p>

##### 6. Multi-Output Regressor with Ridge estimator (MOR_Ridge)

The Ridge estimator based multi-output regressor performs quite similar to the Gradient Boosting variant with an only slightly increased accuracy of 0.49. While the precision, recall and F1-scores for 'win_home' and 'win_away' are slightly better in average than those from Gradient Boosting, the 'draw' prediction scores are a bit lower.

![Fig11](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_6_MOR_Ridge_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 11: MOR_Ridge evaluation results
</p>

##### 7. Multi-Output Regressor with SGD estimator (MOR_SGD)

The multi-output regressor with an stochastic gradient descent estimator produced the best overall scores with an accuracy of 0.51 and all values for 'win_home' and 'win_away' also above 0.5. However, even for this advanced regressor, it was very hard to predict a draw and the precision, recall and F1-score did not exceed 0.3 there. 

Compared to the RC benchmark classifier, the accuracy score was improved by exactly 50%, making this predictor the preferable choice among all of the discussed classifiers and regressors..

![Fig12](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/CR_ClassificationReport_for_7_MOR_SGD_testRatios=0.1..0.5.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 12: MOR_SGD evaluation results
</p>

#### Focus on Accuracy: Is the training set big enough?
The evaluation results presented in the last section show that the accuracy over the complete test series can significantly be increased compared to random choice by using a multi-output regressor with an estimator like Ridge or SGD. However, one question that may arise when looking on the number of investigated matches is: Is this amount of matches and events sufficient for an adequate use of Machine Learning classifiers? To find an answer on this question, we decided to vary the test_ratio and change the amount of training data in terms of matches for the classifier: While the training set is rather large when test_ratio = 0.1 - there still remain 108 out of 120 matches for training and only 12 are used in testing (see Evaluation Setup) - this number is significantly lower when test_ratio = 0.5 - then, there are only 60 out of 120 matches left for training. By starting with test_ratio = 0.1 and then taking small increases of 0.05 for the test_ratio up to 0.5, the accuracies per classifier were recorded and are shown in Fig. 13. There, it can be seen that all of the ML algorithms perform best with the largest training set (i.e., test_ratio = 0.1) and the score decrease with a rising test_ratio value. However, the decrease is only slightly, meaning that even a small training set of 60 matches is sufficient for an acceptable accuracy. Comparing the best accuracies (typically at test_ratio 0.1 or 0.15) with the worst accuracies (typically at test_ratio 0.45 or 0.5), the relative decreases of accuracy are the following:
- 7.4% for MOC_DecTree
- 5.9% for MOC_RandomForests
- 7.7% for MOC_LogisticRegression
- 7.8% for MOR_GradBoo
- 5.9% for MOR_Ridge
- 3.8% for MOR_SGD

These results show once again that all ML algorithms are suited even for small training sets, and that the multi-output regressor with SGD estimator should be preferred over the other classifiers, as it its loss in accuracy is lowest when the training set becomes smaller.

![Fig13](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/Accuracy_vs_test_ratio.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 13: Accuracy scores at various test ratios
</p>

## Justification

Summing up the evaluation results, the following interesting insights could be gained in this project:
- The randomized classifier which serves as benchmark for the ML algorithms produced performance scores in the anticipated range. The overall accuracy score was 0.34.
- As the simple multi-output classifiers suffered from the problem of non-unique classifications, they could not really outperform the randomized classifier, but were in the same range of this simple non-ML classifier.
- However, the advanced multi-output regressors with corrected result vectors showed significantly improved accuracy scores around 0.5 (i.e., around 50% better than the naive randomized approach), as they do not suffer from the non-unique classification problem. The choice of the estimator seems to be of secondary nature, since the differences in the accuracy scores were quite low.
- Nevertheless, if maximum accuracy has to be aspired, we suggest to the multi-output regressor with an SGD estimator, since it produced the highest overall accuracy as well as fine precision, recall and F1-scores. Furthermore, its loss of accuracy was least when the training set was further reduced.
- All classifiers and regressors had problems in accurately determining a draw, which was kind of forecasted in the data visualization section when determining the influence of the features on the results. As football is a very complex sport with thousands of events in a single match, this result is not surprising.


# Conclusion

## Reflection

In this paper, we presented an ML-based approach to predict results of football matches using advanced statistical analysis data from StatsBomb. After exploring the provided data set, we first made some necessary preprocessings, visualized the data and selected a feature set that promises to have the biggest impact on the prediction performance of the ML algorihms. Then, we implemented and evaluated a bunch of modern ML classifiers and regressors and compared their accuracies to a naive approach where the result was selected randomly by the classifier.

The extensive evaluations showed that standard binary multi-output classifiers suffered the problem of non-unique classification results and, thus, led to rather poor prediction accuracies. However, using advanced continuous multi-output regressors with corrected binary results helped to significantly improve accuracies by up to 50% compared to the naive approach. This shows that Data Science and Machine Learning becomes more and more an important field of intereset in modern supervised sports events like football matches.

## Improvement

While the results presented in the previous section show that optimized Multi-Output Regressors which use advanced algorithms like Ridge or SVD help to significantly improve the prediction accuracy,
it has to be mentioned that there is still a large optimization potential, considering that the best accuracy was only slightly above 50% (which is, though, quite fine for complex football matches in our opinion).

However, regarding that the data set consisted only of 120 out of the 3500 matches from open data (and the whole StatsBomb data comprises even much more than those matches!), it is obvious that there exists a large potential by increasing the training set for the ML predictors.

Futhermore, we only used a quantitative approach by simply counting the number of events per event type. As there is also qualitative information included in the data (i.e, different results of the event types), this data could additionally be used to have an even more fine-grained feature set to further improve prediction accuracies. 

Taking into account the timely manner and sequence in which specific events occur may also help to further increase the performance of the classifiers, but additionally increase the complexity of the model, making it much more time and resource consuming. Thus, one has to consider if maximum accuracy should be aspired under each circumstances without considering consumption costs of the data processings and calculations.

# Acknowledges

Thanks go to Udacity´s Data Scientist Nanodegree ([nd025](#ref13)) where this project is part of.

Further acknowledges go to [StatsBomb](#ref1) for providing their [open data set](#ref2).
![StatsBomb](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2014-10_Football/StatsBomb-logo_small.png?raw=true)

# References
1. <a name="ref1">[StatsBomb homepage](https://statsbomb.com/)</a>
2. <a name="ref2">[StatsBomb Open Data - public GitHub Repository](https://github.com/statsbomb/open-data)</a>
2b. <a name="ref2b">[Applied Data Science with Modern Football Data: Predicting Match Results](https://github.com/statsbomb/open-data](https://sschuhmi.github.io/2024/10/06/PredictingFootballResults.html)</a>
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
13. <a name="ref13">[Data Scientist Nanodegree at Udacity](https://www.udacity.com/enrollment/nd025)</a>
