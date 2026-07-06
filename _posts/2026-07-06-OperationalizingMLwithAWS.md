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
- 1000 matches in the open data set from all seasons of all 21 competitions provided via StatsBomb open data
- These matches include 3,508,294 events of all event types

In 2024, we regarded a significantly reduced data amount and only focused on the Data Science part of the problem than Operationalizing the complete E2E process .
- All 116 provided matches in the open data set from specific seasons of 3 competitions - Spanish 'La Liga' 2019/2020, Euro Cup 2024 and Copa America 2024
- These matches included 417,221 events of all event types

From these numbers, one can see that the amount of regarded data is significantly increased by around factor 8.5.


# Methodology

## Data Preprocessing

#### Duplicates and Missing Values

The competitions and matches files could simply be imported using the standard Python3 json import function. They did not include any duplicates or missing rows. While there was no further preprocessing needed for the competitions set, the matches needed to be enriched by the event-specific data of the events files to have the feature set (X) and the output set (y) for the classification algorithms.

#### Feature Set Reduction

In football, scoring a goal typically requires a team to gain possession of the ball, advance into promising attacking areas, create goal-scoring opportunities, and ultimately convert these opportunities into shots on goal. Consequently, successful attacking sequences are often reflected in specific event patterns recorded within the match data.

Typical phases leading to a scoring opportunity include:

- Gaining possession through actions such as dribbles, duels, interceptions, clearances, opponent mistakes, throw-ins, free kicks, or penalty kicks.
- Progressing the ball towards the opponent's goal through carries and combinations of passes.
- Creating goal-scoring opportunities by generating advantageous attacking situations.
- Performing one or more shots on goal.
- Successfully overcoming the opposing goalkeeper and defensive structure.

As a result, many event characteristics recorded in the StatsBomb dataset can be expected to exhibit a relationship to the eventual match outcome. The objective of the feature engineering process is therefore to identify those event-derived variables that provide the most predictive information regarding a team's probability of winning, drawing, or losing a match.

Since all event data is recorded separately for both participating teams, each event-derived feature was transformed into a pair of team-specific match features. For a given event attribute \(x\), two corresponding match-level features were generated:

- `home_<x>`
- `away_<x>`

Depending on the semantic meaning of the attribute, different aggregation strategies were applied. Event frequencies were generally represented as occurrence counts, while quantitative variables such as expected goals (`statsbomb_xg`) and event durations were aggregated by summation. Continuous variables such as pass lengths and pass angles were aggregated using arithmetic means. Binary indicators, for example `under_pressure` or `counterpress`, were represented by the number of positive occurrences within a match.

This aggregation process resulted in a match-level feature matrix containing team-specific statistics for each recorded event type and attribute.

To identify the most relevant predictive variables, correlation analyses were performed between the generated feature set and the three target variables:

- `win_home`
- `win_none`
- `win_away`

The results were visualized using a series of correlation heatmaps, as shown in #Fig2. These visualizations provide an intuitive overview of the strength and direction of the relationship between individual event-derived features and the different match outcomes.

The analysis revealed that many features exhibit only weak correlations with the target variables and therefore contribute little predictive information. To reduce the dimensionality of the feature space and improve computational efficiency, only the features with the largest absolute correlation values were retained for model training.

More specifically, the 30 most influential home-team features and the 30 most influential away-team features were selected, resulting in a reduced feature set of 60 variables. This approach substantially decreases computational and memory requirements while preserving the majority of the predictive information contained in the original event dataset.

The resulting consolidated correlation heatmap is shown in #Fig4. Although the feature space was reduced from 238 team-specific event features (2 × 119 event attributes) to only 60 features, the remaining variables still capture a rich and diverse representation of match dynamics, including offensive actions, defensive behavior, possession-related statistics, goalkeeper actions, and expected-goal metrics. Consequently, the reduced feature set provides a strong foundation for the Machine Learning models developed in this project while avoiding unnecessary model complexity.

![Fig4](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2026-07_MLE-Cap/eval/Heatmap-filtered_mixed-type.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 4: Heatmap of combined-type, full feature set
</p>

The heatmaps show at first glance that the influence of the specific types on the result columns is widely spread: 
- Some of the events have a positive impact on a specific result column, represented by a positive value and in a rather bright color, while other events have a negative impact with values below 0.
- Moreover, some events have a rather large impact on the result, e.g. with values >0.5 or <-0.5, while others have a rather minimal impact on the results columns since their values are close to 0.

![Fig5](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2026-07_MLE-Cap/eval/Heatmap-top30features.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 5: Heatmap of combined-type, filtered to reduced feature set to Top 30 features with highest influence on target columns 'win_home', 'win_none', 'win_away'
</p>

#### Feature Aggregation

The event files contain a large number of event-specific attributes that must be transformed into match-level features before they can be used for Machine Learning. Since the prediction models operate on a single feature vector per match, the event data was aggregated individually for the home and away team, resulting in one row per match in the final dataset.

For many event attributes, the aggregation was performed by counting the number of occurrences within a match. This approach was primarily applied to categorical and event-based attributes, such as passes, shots, duels, interceptions, or goalkeeper actions. The resulting feature values therefore represent the frequency with which specific types of events occurred during a match.

However, not all event attributes were aggregated using simple event counts. Depending on the semantic meaning of the underlying variable, different aggregation strategies were employed:

- **Count aggregation:** Number of occurrences of a particular event or attribute.
- **Sum aggregation:** Cumulative values of quantitative measures such as event duration or expected goals (`statsbomb_xg`).
- **Mean aggregation:** Average values of continuous attributes such as pass length or pass angle.
- **Boolean occurrence counts:** Number of successful occurrences of binary event indicators such as `under_pressure`, `counterpress`, or `pass.cross`.

This aggregation strategy allows quantitative information contained in the event data to be preserved more effectively than a pure counting approach. For example, the total expected goals (`xG`) generated by a team during a match is generally more informative than merely counting the number of shots for which an xG value was recorded.

A potential extension of this work would be to exploit the qualitative values of categorical event attributes more explicitly. For example, a goalkeeper action may contain information regarding the body part involved (`head`, `chest`, `both hands`, `left hand`, `right hand`, `left foot`, or `right foot`). Encoding all possible categories separately could potentially provide additional predictive power. However, doing so would substantially increase the dimensionality of the feature space and the computational complexity of the feature engineering process. Consequently, this project focuses primarily on aggregated numerical representations of the event data.

An important advantage of the chosen aggregation process is that it naturally produces a complete and consistent feature matrix. Each feature value is calculated exactly once for every match and team. If a specific event did not occur during a match, the corresponding feature value is explicitly set to zero. As a result, the generated feature set contains neither duplicate observations nor missing values (`NaN`) resulting from the aggregation process.

The aggregation and transfer of event-derived statistics into the match-level dataset was implemented in the `update_match_stats()` function. This function processes each event attribute according to its predefined aggregation rule and generates corresponding home-team and away-team features, which are subsequently used as input variables for the Machine Learning models.


#### Feature Scaling

A final preprocessing step required prior to model training was the normalization of the feature values. The generated match-level features exhibit substantially different numerical ranges and magnitudes. For example, some event frequencies may occur several hundred times per match, whereas aggregated variables such as expected goals (`statsbomb_xg`) or averaged pass characteristics typically assume much smaller values.

Without appropriate scaling, features with larger numerical ranges could disproportionately influence the learning process of many Machine Learning algorithms. Consequently, the contribution of individual features would no longer solely depend on their predictive relevance but also on their scale.

To ensure comparability between features and to prevent scale-dependent biases, all predictor variables were normalized using the **MinMaxScaler** provided by the scikit-learn framework [[5]5. The transformation maps each feature to the interval \([0,1]\) according to

$$
X_{scaled} =
\frac{X - X_{min}}
     {X_{max} - X_{min}}
$$

where \(X\) denotes the original feature value, \(X_{min}\) the minimum observed value, and \(X_{max}\) the maximum observed value of the respective feature.

As a consequence, a transformed value of 0 indicates the minimum observed value of a feature within the dataset, whereas a value of 1 corresponds to the maximum observed value. All intermediate values are mapped proportionally to this range.

Feature scaling is particularly important in the present study because the feature set contains a heterogeneous mixture of aggregated statistics, including event frequencies, cumulative measures (e.g., total expected goals or event durations), average values (e.g., pass lengths and pass angles), and counts of binary event occurrences. Normalization ensures that these different feature types contribute on a comparable scale during model training and evaluation.

The resulting normalized feature matrix is subsequently used as input for all investigated Machine Learning models.

To perform scaling, we used scikit-learn´s standard MinMaxScaler [[5]](#ref5)


## Model Implementation

### Model Implementation

The Machine Learning pipeline was implemented in a Jupyter Notebook using **Python 3** within the **Anaconda Distribution**. The implementation relies on several widely used data science and machine learning libraries.

The most important packages include:

- **Pandas** for data manipulation and preprocessing.
- **NumPy** for numerical computations.
- **Matplotlib** and **Seaborn** for data visualization and exploratory data analysis.
- **scikit-learn** for feature scaling, model training, and model evaluation.

Within scikit-learn, a variety of supervised learning algorithms were investigated, including both classification and regression approaches. Examples include:

- **Classification models:** `MultiOutputClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `SGDClassifier`.
- **Regression models:** `MultiOutputRegressor`, `Ridge`, and `GradientBoostingRegressor`.

The complete set of imported packages and dependencies can be found in the initialization section of the accompanying Jupyter Notebook.

### Multi-Output Prediction Formulation

The prediction task considered in this project differs from a conventional binary classification problem because a football match can result in one of three possible outcomes:

- `win_home`
- `win_none`
- `win_away`

These outcomes are represented using a one-hot encoded target vector

$$
y =
[y_1, y_2, y_3]
$$

where

$$
y_1 =
\begin{cases}
1 & \text{if the home team wins} \\
0 & \text{otherwise}
\end{cases}
$$

$$
y_2 =
\begin{cases}
1 & \text{if the match ends in a draw} \\
0 & \text{otherwise}
\end{cases}
$$

$$
y_3 =
\begin{cases}
1 & \text{if the away team wins} \\
0 & \text{otherwise}
\end{cases}
$$

As exactly one outcome can occur for a given match, the target vectors are constrained to the following set:

$$
[1,0,0],\;
[0,1,0],\;
[0,0,1]
$$

Consequently, the prediction problem can be formulated as a multi-output classification task with three binary target variables. The goal of the learning algorithm is to estimate the most likely outcome vector based on the engineered match features.

To support this formulation, scikit-learn's multi-output framework was employed. Multi-output classifiers and regressors internally train a separate estimator for each target variable while allowing predictions for all three outputs to be generated simultaneously. This approach enables a direct representation of the one-hot encoded match outcomes and allows a consistent comparison between different Machine Learning algorithms.

The following algorihms were implemented to predict the classification results:
- RandomClassifier: a simple classifier which randomly chooses among one of the three possible outcomes ([1, 0, 0], [0, 1, 0] or [0, 0, 1]). This accuracy, precision, recall and f1 results of this classifier serve as benchmark for the other classifiers.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with DecisionTrees [[7]](#ref7) estimator: Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with RandomForests [[8]](#ref8) estimator: the RandomForest algorithm represents an averaging algorithms based on randomized Decision Trees. A diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.
- scikit-learn´s MultiOutputClassifier [[6]](#ref6) with LogisticRegression [[9]](#ref9) estimator: Logistic Regression is actually implemented as a linear model for classification rather than regression in terms of the scikit-learn/ML nomenclature.

## Multi-Output Regression Refinement

### Multi-Output Regression Refinement

While the multi-output classification models introduced previously represent powerful and widely used Machine Learning approaches, they exhibit an inherent limitation in the present prediction task. Since each output variable is learned independently, the resulting predictions do not necessarily satisfy the problem-specific constraint that exactly one match outcome must occur.

More specifically, the three target variables

- `win_home`
- `win_none`
- `win_away`

are predicted separately. Consequently, a classifier may produce invalid outcome vectors such as

$$
[0,0,0]
$$

indicating that none of the possible outcomes is predicted, or

$$
[1,1,0], \quad [1,0,1], \quad [0,1,1], \quad [1,1,1]
$$

indicating that multiple mutually exclusive outcomes are predicted simultaneously.

From a football perspective, such predictions are not meaningful, as every match must result in exactly one of the three possible outcomes.

To address this issue, three additional multi-output regression models were investigated:

- `GradientBoostingRegressor`
- `Ridge`
- `SGDRegressor`

using scikit-learn's `MultiOutputRegressor` framework.

Unlike classification models, the regressors predict continuous scores rather than binary outcome values. For a given match, the model therefore produces an output vector

$$
\hat{y} =
[\hat{y}_1,\hat{y}_2,\hat{y}_3]
$$

where each component corresponds to one of the three possible outcomes.

A post-processing step is subsequently applied to transform the continuous prediction into a valid one-hot encoded result vector. This correction procedure assigns the value 1 to the outcome with the highest predicted score while assigning the value 0 to all remaining outcomes.

Formally, let

$$
k =
\operatorname{argmax}
(\hat{y}_1,\hat{y}_2,\hat{y}_3)
$$

denote the index of the largest predicted value. The corrected prediction vector is then defined as

$$
\hat{y}_{corr,i} =
\begin{cases}
1 & \text{if } i = k \\
0 & \text{otherwise}
\end{cases}
$$

As an example, consider the continuous prediction

$$
[0.2,\;0.6,\;0.7]
$$

generated by a regressor. Since the third component is the largest value, the corrected prediction becomes

$$
[0,\;0,\;1]
$$

corresponding to an away-team victory (`win_away`).

This refinement guarantees that every prediction satisfies the fundamental constraint of the problem formulation, namely that exactly one match outcome is predicted for every match. Furthermore, it allows the regression models to express different degrees of confidence through their continuous output values while still producing valid classification results after the correction step.

The following three regressors were implemented and optizimed using the algorithm described above:
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with GradientBoostingRegressor [[8]](#ref8) as estimator: This estimator builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with Ridge [[11]](#ref11) as estimator: Ridge represents a Linear least squares with L2 regularization and minimizes the following objective function:
||y - Xw||^2_2 + alpha * ||w||^2_2
- scikit-learn´s MultiOutputRegressor [[10]](#ref10) with Stochastic Gradient Descent (SGD) [[12]](#ref12) as estimator: SGD represents a simple, yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression.

In summary, we compare the naive RandomClassifier with three ML multi-output classifiers and three ML multi-output regressor with corrected result vectors.

Finally, after having evaluated the results and decided in favor of a specific classifier or regressor (see Results section below), the corresponding classifier can easily be organized within an ML pipeline. Furthermore, the extract (-> import from StatsBomb open data), transform (-> data preprocessing) and load (-> store processed data) parts could also be extracted in an ETL pipeline which would be executed prior to the ML pipeline. The pipelines would guarantee automated, simple and flexible runs of the algorithms.

# Results

## Model Evaluation and Validation

#### Experimental Evaluation Setup

The models introduced in the previous section were evaluated using an automated experimental framework based on repeated randomized train-test splits. The data was partitioned into training and test sets using scikit-learn's `train_test_split()` function, ensuring reproducible yet diverse evaluation scenarios through the controlled use of varying random seeds.

The primary objective of these experiments was to investigate the robustness and generalization performance of the different Machine Learning models under varying amounts of training data. Since the dataset analyzed in this project is relatively small compared to many modern Machine Learning applications, the size of the training set may have a considerable influence on the predictive performance of the investigated algorithms.

To analyze this effect, several train-test split ratios were evaluated. The test set proportion was varied from **10% to 20%** in increments of **5 percentage points**, resulting in the following test ratios:

- 10%
- 15%
- 20%

For each test ratio, a series of **300 independent experiments** was performed. In every experiment, the following steps were executed:

1. Random partitioning of the dataset into a training set and a test set.
2. Training (fitting) of each Machine Learning model using the generated training set.
3. Prediction of the match outcomes for all samples in the corresponding test set.
4. Computation and storage of the evaluation metrics.
5. Increment of the `random_state` parameter to generate a different train-test split for the subsequent iteration.

By repeatedly varying the train-test split while maintaining reproducibility through deterministic random seeds, the influence of a particular dataset partition on the evaluation results is significantly reduced. Consequently, the reported performance measures represent average model behavior across many independent train-test configurations rather than a single potentially favorable or unfavorable split.

The repeated evaluation procedure also allows the variance of the performance metrics to be assessed and provides a more reliable estimate of the true predictive capabilities of each algorithm.

From a computational perspective, a complete test series comprising 300 evaluation runs for a single test ratio required approximately 12.5 minutes on the utilized hardware platform. Since three different test ratios were investigated, the overall experimental evaluation required approximately 37.5 minutes to complete.

The resulting performance metrics were subsequently aggregated and compared in order to identify the most suitable Machine Learning approach for football match outcome prediction.

#### 1. Overall Accuracy Comparison

Before taking a look at the classification report, the investigated Machine Learning models are compared using their overall prediction accuracy. Since accuracy represents the primary evaluation metric of this study, it provides a concise overview of the relative performance of the different approaches.

Figure 12 shows the average accuracy obtained by each classifier and regressor for test ratios between 10% and 20%. It can be observed that the accuracy remains largely stable across all investigated train-test splits, indicating that the models are relatively insensitive to the exact partitioning of the available data.

The RandomClassifier serves as the baseline and achieves an average accuracy of approximately 0.33, which corresponds closely to the theoretical expectation of randomly selecting one of three possible match outcomes. The DecisionTreeClassifier improves only moderately upon this baseline and achieves an average accuracy of approximately 0.39.

Among the classification-based approaches, the RandomForestClassifier and LogisticRegressionClassifier perform substantially better, reaching accuracies of approximately 0.53 and 0.52, respectively.

The strongest performance is achieved by the regression-based approaches. The MultiOutputRegressor using Ridge regression obtains the highest overall accuracy of approximately 0.67, followed closely by the GradientBoostingRegressor with approximately 0.65 and the SGDRegressor with approximately 0.63.

These results indicate that the regression-based formulation combined with the subsequent one-hot correction procedure provides a more effective solution to the match outcome prediction problem than the investigated classification-based approaches. Consequently, the following sections focus on a more detailed analysis of the best-performing models.

![Fig12](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2026-07_MLE-Cap/eval/Accuracy_vs_test_ratio.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 12: Accuracy scores at various test ratios
</p>

### 2. MultiOutputRegressor using Ridge Regression (MOR-Ridge)

The MultiOutputRegressor based on Ridge regression achieved the highest overall accuracy of all investigated models, reaching an average accuracy of approximately 0.67. Compared to the random baseline (0.33), this corresponds to an improvement of more than 100%, demonstrating that the event-derived match features contain substantial predictive information regarding match outcomes.

A more detailed view of the model performance is provided by the classification report shown in Figure X.

The Ridge model performs particularly well for the `win_home` class. A recall score of approximately 0.90 indicates that the vast majority of home-team victories are correctly identified. The corresponding F1 score of approximately 0.72 further confirms that home wins can be predicted with comparatively high reliability.

The model also achieves balanced performance for the `win_none` class (draws), with precision, recall, and F1 scores ranging between approximately 0.63 and 0.71. This suggests that the model is reasonably capable of recognizing matches that end without a winner.

However, the performance for the `win_away` class differs substantially. While the precision of approximately 0.36 remains moderate, the recall drops to only 0.04, resulting in an F1 score of approximately 0.08. This indicates that the model rarely predicts away-team victories, even when they actually occur.

This behavior can largely be explained by the class distribution of the dataset. Home-team victories represent the largest outcome category, whereas away-team victories are the least frequent. Consequently, the Ridge model appears to be biased towards predicting the more common classes (`win_home` and `win_none`), thereby maximizing overall accuracy at the expense of detecting away-team victories.

While this class-specific weakness limits the practical applicability of the model in its current form, the overall accuracy results nevertheless demonstrate that Ridge regression is the most effective approach among all investigated algorithms. The classification report further reveals that future improvements should primarily focus on enhancing the predictive performance for the `win_away` class, for example through improved feature engineering, class balancing techniques, or alternative model architectures.

![Fig13](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2026-07_MLE-Cap/eval/CR_Classification_report_for_MOR_Ridge.png?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 13: Classification Report for MOR Ridge (Regressor with highest accuracy)
</p>

# Operationalization and Deployment

While the previous chapters focused on data analysis, feature engineering, model development, and model evaluation, an additional objective of this project was the operationalization of the best-performing Machine Learning model. To achieve this goal, the complete prediction pipeline was implemented and deployed on the Amazon Web Services (AWS) cloud platform, enabling automated prediction of football match outcomes based on StatsBomb event data.

The deployed solution consists of three main components:

1. Model training in an AWS SageMaker Jupyter Notebook.
2. Deployment of the trained model as a SageMaker inference endpoint.
3. Implementation of an AWS Lambda function that performs feature generation and invokes the deployed endpoint.

The resulting architecture is illustrated below.

```text
StatsBomb Open Data (S3)
           │
           ▼
  SageMaker Notebook
  - Data Import
  - Feature Engineering
  - Model Training
  - Model Evaluation
           │
           ▼
  Best Model (MOR Ridge)
           │
           ▼
 SageMaker Endpoint
           ▲
           │
 AWS Lambda Function
  - Load Match Data
  - Load Event Data
  - Build Features
  - Invoke Endpoint
           │
           ▼
      Prediction
```

## Model Training in SageMaker

The complete data preparation, feature engineering, model training, and evaluation process was implemented within an AWS SageMaker Jupyter Notebook environment. The notebook imports the StatsBomb Open Data repository, aggregates event statistics at match level, performs feature selection, applies feature scaling, and trains multiple Machine Learning models.

Following the experimental evaluation presented in the previous chapter, the **MultiOutputRegressor using Ridge Regression (MOR Ridge)** achieved the highest average prediction accuracy and was therefore selected for operational deployment.

After training, the resulting model was serialized and prepared for deployment using SageMaker's built-in model hosting capabilities.

## Endpoint Deployment

To enable online predictions, the trained MOR Ridge model was deployed as a real-time SageMaker inference endpoint.

The endpoint provides a REST-like prediction interface that accepts a feature vector containing the selected match statistics and returns the corresponding prediction result. Internally, the endpoint loads the serialized Ridge regression model and performs inference on incoming requests.

For a given match, the endpoint returns:

- The predicted match outcome.
- The corresponding one-hot encoded prediction vector.
- The raw regression scores generated by the model before the final outcome correction step.

An example prediction response is shown below:

```json
{
  "prediction": "win_none",
  "prediction_vector": {
    "win_home": 0,
    "win_away": 0,
    "win_none": 1
  }
}
```

The SageMaker endpoint therefore represents the central prediction component of the deployed architecture.

Figure X shows a screenshot of the Endpoint "up & running".

![Fig14](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2026-07_MLE-Cap/10-endpoint_InService.jpg?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 14: AWS SageMaker Inference Endpoint in Service
</p>

## Feature Engineering Service using AWS Lambda

While the SageMaker endpoint expects a feature vector as input, the original StatsBomb data is stored as JSON documents containing match metadata and detailed event information. Consequently, an additional preprocessing step is required before a prediction can be generated.

To automate this process, an AWS Lambda function was implemented. The Lambda function acts as an orchestration layer between the raw data source and the deployed Machine Learning model.

For each prediction request, the Lambda function performs the following steps:

1. Load the corresponding match file from Amazon S3.
2. Determine the participating home and away teams.
3. Load the event file of the requested match.
4. Aggregate the event statistics according to the feature engineering rules used during model training.
5. Construct the feature vector containing the selected model features.
6. Invoke the SageMaker endpoint.
7. Return the prediction result.

A key design objective was to ensure that the feature engineering logic executed within the Lambda function exactly matches the preprocessing logic used during model training. This guarantees consistency between training and inference and prevents discrepancies caused by differing feature calculations.

## End-to-End Prediction Workflow

The resulting operational workflow can be summarized as follows:

1. A match identifier is provided to the Lambda function.
2. The corresponding match and event data are loaded from S3.
3. Match-level statistics are generated using the same aggregation procedures applied during model training.
4. The generated feature vector is forwarded to the MOR Ridge inference endpoint.
5. The endpoint predicts the match outcome.
6. The prediction result is returned to the caller.

As a result, the complete Machine Learning workflow can be executed automatically without requiring any manual intervention after deployment.

Figure X shows an exemplary successful test call on the Lambda function in AWS.

![Fig16](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2026-07_MLE-Cap/04-LambdaSuccess.jpg?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 16: Successful Test Call on AWS Lambda function
</p>

Moreover, the next figure shows the taken concurrency settings to be able to react on temporarily high-throughput, low-latency situations where many parallel calls on the Lambda function are performed.

In AWS Lambda, concurrency refers to the number of function instances that can execute simultaneously. By default, Lambda automatically creates execution environments when requests arrive. If no initialized execution environment is available, Lambda must create a new one, resulting in a so-called cold start. During a cold start, AWS has to provision resources, initialize the runtime environment, load the function code, and execute all initialization logic before the actual request can be processed. This introduces additional latency for the affected invocation.

To reduce this latency, AWS provides the Provisioned Concurrency feature. Provisioned Concurrency maintains a predefined number of pre-initialized Lambda execution environments that are ready to process requests immediately. As a result, requests can be served without experiencing cold-start delays.

The value 30 was chosen primarily as a demonstration of the operational deployment capabilities of AWS Lambda rather than because a workload analysis indicated a need for 30 concurrent requests. For a real production system, the Provisioned Concurrency value would typically be determined based on expected traffic levels, response-time requirements, and cost considerations, which is beyond the scope of this work.

![Fig17](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/2026-07_MLE-Cap/05-LambdaConcurrency.jpg?raw=true)
<p align="center" style="text-align:center, text-style:italic">
Fig. 17: Lambda Concurrency Settings
</p>


## Discussion

The implemented AWS architecture demonstrates how a Machine Learning model developed in an experimental environment can be transformed into an operational prediction service. By combining SageMaker for model hosting with AWS Lambda for serverless preprocessing and orchestration, the solution provides a scalable and reusable framework for football match outcome prediction.

Furthermore, the architecture separates feature engineering, model inference, and data storage into independent components, enabling future extensions such as automated retraining, batch predictions, API integration, or deployment of alternative Machine Learning models without fundamental changes to the overall system design.

# Acknowledges

Thanks go to Udacity´s AWS Machine Learning Engineer Nanodegree ([nd189](#ref13)) where this project is part of.

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
13. <a name="ref13">[AWS Machine Learning Engineer Nanodegree at Udacity](https://www.udacity.com/enrollment/nd189)</a>
