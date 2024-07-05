# This will make you rethink of developers' infamous reputation as passionated night-working coders

![Coding](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/coding-pic.jpg?raw=true)

# Introduction	

There are many circulating prejudices about software developers, like being “sociophobic nerds, neglecting their real lifes in favor of hanging in front of the computer for writing code, discussing hacking and eating pizza” [[1]](#ref1). Moreover, there exists the persistent opinion that many software developers start their work day quite late, while loving to code at night, being a key to creativity and productivity [[2]](#ref2) in a distraction-free zone [[6]](#ref6).

But, do those statements defy a critical analysis or are they just urban legends? To find out, I analyzed data from the 2017 Annual Developer Survey of Stack Overflow [[3]](#ref3) covering more than 64,000 reviews from more than 200 countries, particularly concentrating on developers' preferred  time of day to start work. The goal is to find out if there are effects on satisfaction and the expected salary, obviously having in mind that the time of day where developers would preferably work is only one of several factors which has an influence on these issues.

# Analyzing the time of day developers would prefer to start their work

When analyzing the time of day the developers would like to start their work day according to the survey, I particularly focused on three questions which came to my mind:
1.	Do developers really have “special times” for work, or would they prefer to be the typical “9-to-5" workers?
2.	Is there an influence on the developers’ job and career satisfactions based on the time of day they wanted to start their work?
3.	Can you recognize a trend on the predicted salary for a developer solely based on the time they wanted to start to work? 

## Question 1: Are software developers the typical "9-to-5" workers?

One of the raised questions in Stack Overflow´s survey was:

*“Suppose you could choose your own working hours for an 8-hour day. What time would you start work for the day? Please adjust the slider to the hour nearest your ideal start time. The box next to the slider will display your selection using a 24-hour clock”*

Regarding Figure 1 which shows the histogram of the survey results for this question, one can see that developers seem not to confirm the prejudice that they rather would like to have a significant amount of their work time at night:
-	Most developers would prefer to start at 09:00 AM (3,570, i.e., 26.4%), followed by 10:00 AM (3,380, i.e., 25.0%), meaning more than half of the participants would prefer working completely over the typical daylight times.
-	In contrast, only 7.6% in sum decided in favor of rather unconventional work start times between 01:00 PM and 05:00 AM

![Figure 1](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/WorkStart24_Distribution.png?raw=true)

*Figure 1 – Histogram of survey results to the question concerning preferred work start*

Furthermore, having a closer look on Figure 2 showing the respective cumulative histogram, it becomes obvious that around 90% of the survey participants would start an 8h-work day between 06:00 AM and 11:00 AM, meaning they would finish work before 08:00 PM and, thus, would clearly not prefer to work at night.

![image2\label=image2](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/WorkStart24_Aggregation.png?raw=true)

*Figure 2 – Cumulative histogram of preferred work starts*

## Question 2: Is there a influence between the time of day a developer would prefer to start to work and their satisfaction?

To discover the answer to this question, let us take a look on Figure 3, which represents a heatmap illustrating the dependency between preferred week start (on the x-axis) and the denoted satisfactions and the salary the respective persons gain. While positive values (> 0.00) represent a positive effect on satisfaction or salary, negative values (< 0.00) represent a negative effect and zero values depict no effect.

![image3\label=image3](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/CorrelationHeatMap_cropped.png?raw=true)

*Figure 3 – Heatmap showing the influence of preferred work starts on different satisfaction values and salary*

Figure 4 summarizes the effects of the values shown in Figure 3 by summing up the 4 values per work start column and showing the correlation between the preferred work start and the cumulated satisfaction/salary values. As there may be many other factors which obviously influence satisfactions and salary in a much larger way than the preferred work start of the developers, it still can be seen there is a trend that the typical “daylight developers” preferring to start between 05:00 AM and 10:00 AM have positive aggregated values, while the most other developers which prefer rather unconventional work start times display mostly negative values in this figure.

![image4\label=image4](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/AggregatedCorrelations.png?raw=true)

*Figure 4 – Correlation between preferred Work Start and aggregated satisfactions and salary*

## Question 3: Is there a difference in the salary that is predicted based on the time of day a developer prefers to start work?

Finally, let us try to find an answer on the question if the predicted annual salary for developers which would prefer to work at traditional daylight times differs from those which rather would work at unconventional times, involving evening hours, night hours or very early morning hours. 

Therefore, a Linear Regression model [[4]](#ref4), [[5]](#ref5) was used to predict the expected salary based on the given answers in the survey. While this may be an oversimplification as it only regards one specific question concerning the preferred work start, it is interesting to see that the respective Figure 5 shows a clear trend that people which preferred traditional work start hours seem to have the highest predicted salaries, while “night workers” which would prefer to start their work day between 10:00 PM and 02:00 AM have by far the lowest predicted salaries: The highest predicted annual salary (almost 80k $ for the “early birds” that would prefer to start at 05:00 AM) is more than 7 times as much as the lowest predicted salary (around 11k $ for those that would prefer to start at midnight).

![image5\label=image5](https://github.com/sschuhmi/sschuhmi.github.io/blob/main/_posts/img/WorkStart_PredictedSalaries.png?raw=true)

*Figure 5 - Predicted Annual Salary (in $) using SciKit Learn´s LinearRegression Model [[4](#ref4), [5](#ref5)]*

## Conclusion 

In this article, we focused on questions concerning the preferred working times of software developers, trying to find out if there is something special about them (i.e., being passionated “coders at night”) or if they are rather traditional workers that prefer to work at daylight times.

Analyzing the results of the regarded survey showed that …
-	… the vast majority of developers prefers not to work at night
-	… developers tend to be happier and/or earn more if they prefer to work at traditional daylight times
-	… ”early birds” which would like to start an 8h working day in the early morning are supposed to earn the highest salaries

These results seem to unmask the established hypotheses (e.g., in [[2]](#ref2), [[6]](#ref6)) to be rather prejudices up to urban legends than representing the truth.

## References

1. <a name="ref1">[Are you afraid of software developers?](https://ilnumerics.net/blog/are-you-afraid-of-software-developers/)</a>
2. <a name="ref2">[Why developers like to code at night](https://opensource.com/article/20/2/why-developers-code-night)</a>
3. <a name="ref3">[Stack Overflow Developer Survey 2017](https://survey.stackoverflow.co/2017)</a>
4. <a name="ref4">[LinearRegression — scikit-learn 1.5.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)</a>
5. <a name="ref5">[Linear Regression in Python with Scikit-Learn](https://medium.com/analytics-vidhya/linear-regression-in-python-with-scikit-learn-e1bb8a059cd2#:~:text=With%20Scikit-Learn%20it%20is%20extremely%20straight%20forward%20to,machine%20learning%20library%20to%20train%20on%20your%20data.)</a>
6. <a name="ref6">[The Advantages of Coding at Night: The Distraction-Free Zone](https://www.lucidcoder.dev/the-advantages-of-coding-at-night)</a>
