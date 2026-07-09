# capstone-project-part2 code explanation and comparision OLS VS RIDGE regression
1.import pandas as pd 
•	Its python library ,python is used for handling tabular data(like excel spread sheets)in python
2.import os 
•	Allows our python script to interact directly with os our computer is running 
•	Os.chdir()navigation
3.import numpy as np
•	Numpy library nick named as np.numpy for high performance mathematical and array operation in python.
4.df=pd.read.csv(‘cleaned_data.csv)
•	Read  ecternal file named cleaned_data.csv
5.print(df.head())
•	Grabs just the 5 rows of your dataset.
•	We can quicky inspect the columns and make sure the data loaded correctly

This code preparing our dataset for ML by spilitting our column in to features (the prediction)and targets(what you want to predict)
2 different types of ML at same time
1.	Regression (predicting a continuous number)(predicting exact )
2.	Classification (predicting a category/label)
6.   target_col=’median_house_value’
      X=df.drop(column=[target_col])
•	It creates a new data frame called x that contains all of your data except for the median_house_value column.
•	In ML. X trationally represents our feature(independent variable like no.of rooms,location,age of house)that model will use to learn predictions.
Importing tool

From sklearn.model_selection import train_test_split

*this import a built in helper function from scikit_learn(the primary ML library in python)that handles the random chopping of your data automatically.
Splitting for regression mode(predicting exact prices)
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

*tesd_size=0.2 ->tells 20% of the data into test set ,leaving 80% for training set
*random_state = 42 ->acts like a seed for random number generator
*because the data is shuffled randomly ,using 42 ensure that if we run this code tomorrow ,it will shuffle;e the same way.keep consistent.
         * The Outputs:  X_train & y_reg_train: The clues and exact prices for training.
      *X_test & y_reg_test: The clues and exact prices kept secret for testing.
Splitting for the Classification Model (Predicting Cheap vs. Expensive)

_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

	_ means since we already created X_train and X_test in the line right above,we don’t need to create them again.
	The underscore _ is python shorthand for:”throw this path away,I don’t need it.”
Feature scaling
scaler = StandardScaler().     Eg(no of rooms 5  , population 2500 )
*This initializes the scaler tool from sklearn, getting it ready to do some math.
*Now, every single feature (whether it's room counts, population, or income) is measured using the exact same "ruler."
scaler.fit(X_train)
*The scaler looks only at your training clues (X_train) and calculates the normal average and spread for every column.
*If the scaler looks at the test data to calculate averages, it’s accidentally "peeking" at the exam answers ahead of time (this is called data leakage).

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)	
•  Now that the scaler knows what the averages are, it actually alters the data.
•  It changes the real-world numbers in both X_train and X_test into those standardized scores (like -1.2, 0, 2.3).
Note: Before this step, your data had massive numbers mixed with tiny numbers. Now, every single number in your dataset is on a level playing field, usually sitting somewhere between -3 and 3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
•  LinearRegression: The math robot you want to train.
•  mean_squared_error & r2_score: The formulas used to grade how well your robot performs.
# Initialize and train the model
lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
•  lr = LinearRegression(): You buy a blank-slate robot and name it lr. Right now, it knows nothing about house prices.
•  lr.fit(...): This is the studying phase. You hand the robot a giant textbook containing data on past houses.
	X_train_scaled: The sizes of the houses.
	y_reg_train: The actual prices those houses sold for.
	What happens: The robot studies the textbook, looks for patterns, and draws a "line of best fit" through the data points so it can understand the connection between size and price.



#Making Predictions(quiz time)
# Predict on the test set
y_pred_lr = lr.predict(X_test_scaled)
•  You want to see if the robot actually learned anything, or if it just memorized the textbook.
•  You hand it a pop quiz containing a list of brand-new house sizes (X_test_scaled) that it has never seen before.
•  lr.predict(...): The robot uses the line it drew during studying to guess the prices of these new houses.
•  y_pred_lr: This variable stores the robot's written-down guesses.
#Calculating Metrics(grading the quiz)
# Calculate metrics
mse_lr = mean_squared_error(y_reg_test, y_pred_lr) #how big were the robots mistake?
r2_lr = r2_score(y_reg_test, y_pred_lr) #what percentage grade the robot get?
You compare the robot's guesses (y_pred_lr) against the actual, true prices of those houses (y_reg_test).
	mean_squared_error (MSE): This calculates how far off the robot's guesses were on average. If the robot guessed $300k but the house was $310k, it looks at that $10k error, squares it, and averages it out across all quiz questions.
	The Rule for MSE: Lower is better. If your MSE is 0.0000, it means the robot made zero mistakes. The larger the MSE number, the worse the robot did.

	Goal: You want this number as close to 0 as possible.

R^2 Score — The Percentage Grade
MSE is useful, but it can be hard to read. If I tell you, "My model's MSE is 45,000,000," you have no idea if that's a good score or a terrible score without knowing what you are predicting.
That is why we use R^2. It turns that messy MSE number into a standardized percentage grade between 0 and 1.
	Score of 1.0 (100%): A perfect score. The robot predicted everything flawlessly.
	Score of 0.0 (0%): A terrible score. The robot didn't look at the house sizes at all. Instead, it just calculated the average price of all houses and guessed that exact same average number for every single question.
	Negative Score (below 0%): The robot is doing worse than a person blindly guessing the average. It means the model is actively broken or mismatched for the data.
📌 The Rule for R^2: Higher is better. If your R^2 is 0.85, it means your robot successfully figured out 85% of the patterns in the data. Only 15% of the data's behavior is still a mystery to it.
(Report Card)
print(f"Linear Regression - MSE: {mse_lr:.4f}, R2: {r2_lr:.4f}")
This simply prints out the final report card on your screen so you can see the MSE score and the R^2 percentage score rounded neatly to 4 decimal places.
To make it neat, we add :.4f right after the variable names.
	The : means "Format this number."
	The .4f means "Round this number to exactly 4 decimal places (4 Floating-point digits)."
Ridge Regression
from sklearn.linear_model import Ridge
*What it means: You open up your machine learning toolbox (sklearn) and pull out the Ridge robot.
ridge = Ridge(alpha=1.0)
•  What it means: You initialize the robot and name it ridge. Inside the parentheses, you set alpha=1.0.
•  In our example: alpha=1.0 is the strength of the safety brake. You are telling the robot: "If you try to give a massive, crazy importance weight to a feature (like $5,000 per plant), I am going to penalize you. Keep your weights small and balanced."
Studying the data(training)
ridge.fit(X_train_scaled, y_reg_train)
•  What it means: You command the robot to study using the .fit() function.
•  In our example: You hand the robot the historical data.
	X_train_scaled contains the house sizes and number of plants.
	y_reg_train contains the actual prices those houses sold for.
	The result: The robot looks at the data, wants to give plants a huge weight, but feels the alpha=1.0 penalty brake. So, it calculates a smart, balanced formula where house size matters a lot, and plants are mostly ignored.
Predicting
y_pred_ridge = ridge.predict(X_test_scaled)
•  What it means: You give the robot a test using the .predict() function.
•  In our example: You hand it X_test_scaled—a list of brand-new houses it has never seen before. The robot applies its smart, balanced formula to guess what these new houses should cost. It saves all of its guesses inside the variable y_pred_ridge.
Measuring the Size of the Mistakes (MSE) 
*(Calculates how big the mistakes were (Lower is better)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)

•  What it is: You are measuring how far off the robot's guesses were from the real answers.
•  How it works: 1. The computer looks at Question 1. The real house price (y_reg_test) was $300,000. The robot guessed (y_pred_ridge) $310,000. The mistake is $10,000.
2. To get rid of negative numbers, the computer squares that mistake (10,000×10,000).
3. It does this for every single house on the test, adds them all up, and finds the average (mean).
•  The Goal: You want this number to be as close to 0 as possible. A lower number means the robot's guesses were very close to the real prices.


Calculating the Percentage Grade (R^2)
*Turns those mistakes into a final percentage grade (Higher is better).
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
•  What it is: MSE gives you a huge, messy number (like 45,000,000) which is hard to understand. Line 6 fixes this by converting that mess into a standard percentage grade between 0.0 (0%) and 1.0 (100%).
•  How to read it:
	If the R^2 score is 1.0, the robot got a perfect 100%. Every single guess was exactly right.
	If the R^2 score is 0.85, the robot gets an 85%. It successfully figured out 85% of the patterns in house prices.
	If the score is 0.0, the robot got a 0%. It learned absolutely nothing.
•  The Goal: Unlike MSE, you want this number to be as close to 1.0 as possible.
Printing the Report
*Displays both numbers neatly on your screen.
print(f"Ridge Regression - MSE: {mse_ridge:.4f}, R2: {r2_ridge:.4f}")
Making the Two-Column List
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
	X.columns: The names of your features (e.g., Size, Bedrooms, Distance to School).
	lr.coef_: The importance scores (weights) the robot gave to each feature.
	Your table now looks like this:
Feature	Coefficient
Size	150
Distance to School	-200
Backyard Plants	5
Removing the Minus Signs
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
The .abs() command stands for Absolute Value. It simply removes any minus (-) signs and turns everything into a positive number.
	Why do this? Look at Distance to School up there. Its score is -200. That minus sign just means it drags the house price down. But it is still a huge factor!
	If we don't remove the minus sign, the computer will think -200 is a tiny number (less than 0) and ignore it.
After this line runs, a new column called Abs_Coef is added:
Feature	Coefficient	Abs_Coef
Size	150	150
Distance to School	-200	200
Backyard Plants	5	5

Sorting from Biggest to Smallest
top_3 = coef_df.sort_values(by='Abs_Coef', ascending=False).head(3)
•  sort_values(by='Abs_Coef', ascending=False): It rearranges the rows so the biggest number in the Abs_Coef column goes to the very top, and the smallest goes to the bottom.
•  .head(3): It cuts the list off after the first 3 rows and throws the rest away.
Our table now looks like this, sorted and cut down to the top 3:
	Distance to School (200)
	Size (150)
	Backyard Plants (5)
print(top_3)
*his final line prints that exact Top 3 list onto your screen so you can clearly see the three things your model cares about the most!
Checking for the Overbalance
print(y_clf_train.value_counts(normalize=True))
If your robot just gets lazy and guesses "Not Quick Sell" for every single house, it would be right 90% of the time! But it would completely miss the 10 special houses that sell instantly.
This line calculates the percentages to expose that problem. When you run it, it prints:
	0: 0.90 (90% normal houses)
	1: 0.10 (10% instant-sell houses)
This tells you: "Warning! Your data is highly imbalanced."
Grabbing the Classification Tool
from sklearn.linear_model import LogisticRegression
We load Logistic Regression. This robot doesn't predict numbers like $350,000. It acts like a sorting machine that puts houses into two distinct buckets: Bucket 0 (Normal) or Bucket 1 (Quick Sell).
Giving the Robot a Smart Rule
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
This is where we fix the 90/10 problem.
	max_iter=1000: Gives the robot plenty of practice time to get its math right.
	class_weight='balanced': You tell the robot: "Because 'Quick Sell' houses are so rare, finding one is 9 times more important than finding a normal house. If you ignore them, you fail your training."
This "balanced" rule forces the robot to pay equal attention to both categories, preventing it from just guessing "0" every time to look smart.
Training on the Housing Data
clf.fit(X_train_scaled, y_clf_train)
Now, the robot goes to school (.fit).
	It looks at the house features (X_train_scaled, like low price, great location, or extra bedrooms).
	It looks at the answers (y_clf_train, whether it actually sold quickly or not).
Because you turned on the balanced rule in the previous step, the robot successfully learns the unique patterns that make those rare 10% of houses sell instantly!
Let’s stick with our same housing example: predicting whether a house will Sell Quickly (1) or Not Sell Quickly (0).
from sklearn.metrics import precision_score, recall_score, f1_score
You import three metrics used to judge categorization models:
	Precision: Out of all the houses the robot guessed would sell quickly, how many actually did? (Measures quality).
	Recall: Out of all the houses that actually sold quickly in the real world, how many did the robot successfully find? (Measures quantity).
	F1 Score: A balanced metric that averages Precision and Recall together to give a single "all-around performance" grade.
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []
•  thresholds: These are the different confidence levels we want to test.
	A threshold of 0.3 means the robot is relaxed: if it's even 30% sure a house will sell quickly, it labels it as 1.
	A threshold of 0.7 means the robot is strict: it has to be at least 70% sure before labeling it as 1.
•  results = []: An empty folder to store our final grades.
 Testing Loop(for loop)
A. Making the Guess
y_pred_t = (y_prob_clf >= t).astype(int)
•  y_prob_clf is the robot's raw confidence score for each house.
•  If we are testing threshold t = 0.6, any house with a confidence score of 60% or higher becomes True (1), and anything lower becomes False (0).
B. Grading and Saving
results.append({
    'Threshold': t,
    'Precision': precision_score(y_clf_test, y_pred_t),
    'Recall': recall_score(y_clf_test, y_pred_t),
    'F1': f1_score(y_clf_test, y_pred_t)
})
The computer compares these custom threshold guesses (y_pred_t) against the actual real-world answers (y_clf_test). It calculates the Precision, Recall, and F1 scores for this specific setting and saves them in our folder.
Displaying the Comparison Table
results_df = pd.DataFrame(results)
print(results_df)
This takes all the saved data and converts it into a clean, 5-row table on your screen.
This code compares two different robots to see which one is better at sorting the houses. One robot is standard, and the other has a heavy safety brake applied to it to stop it from overreacting.
Hiring the Strict Robot (C=0.01)
clf_reg = LogisticRegression(max_iter=1000, C=0.01, class_weight='balanced')
clf_reg.fit(X_train_scaled, y_clf_train)
•  What C=0.01 means: In Logistic Regression, the letter C stands for Confidence/Flexibility freedom.
	A standard robot usually has a C=1.0 (it's allowed to trust the data completely).
	Setting C=0.01 is like putting a massive filter or safety brake on the robot. You are telling it: "Do not overreact to tiny details. Keep your formulas extremely simple and conservative."
•  clf_reg.fit(...): The robot studies the historical housing data while wearing this strict safety brake.
Guessing the Percentages
y_prob_reg = clf_reg.predict_proba(X_test_scaled)[:, 1]
•  What it does: We give the strict robot a pop quiz with brand-new houses (X_test_scaled).
•  Instead of just shouting "Yes" or "No", the command predict_proba forces the robot to give us a probability score from 0% to 100% for each house (e.g., "I am 82% confident this house will sell quickly"). We store those percentage guesses in y_prob_reg.
Grading with the AUC Score
auc_base = roc_auc_score(y_clf_test, y_prob_clf)
auc_reg = roc_auc_score(y_clf_test, y_prob_reg)
To see which robot is actually smarter, we use a grading style called AUC (Area Under the Curve).
Think of the AUC score as a robot's sorting ability grade.
	If you give the robot a random "Quick Sell" house and a random "Normal" house, the AUC score measures how good the robot is at correctly ranking the quick-selling house with a higher percentage than the normal one.
	AUC Grade Scale:
	1.0: A perfect grade. The robot perfectly ranks every single house correctly.
	0.5: A terrible grade. The robot is basically just flipping a coin to make guesses.
Printing the Comparison
	print(f"Baseline (C=1.0) AUC: {auc_base:.4f}")
	print(f"Regularized (C=0.01) AUC: {auc_reg:.4f}")
This prints out the final scorecard side-by-side:
	Baseline (C=1.0): The score of your original, standard robot.
	Regularized (C=0.01): The score of your new, conservative robot.
How to use these results:
When you look at your screen, see which number is higher. If the Regularized (C=0.01) AUC score is higher, it proves that putting a safety brake on the robot helped it ignore random noise and made it a better judge of houses in the real world!
-=------
Let’s stick with our example comparing our two housing robots: the Standard Robot (base) and the Strict Robot (reg).
In the last script, you checked their scores on a single test. But what if that test just happened to have an unusually lucky or unlucky mix of houses? How can we be sure one robot is truly better than the other, and it wasn't just a fluke?
This code uses a statistical superpower called Bootstrapping to run a simulated tournament 500 times. It helps us see the true performance difference with extreme confidence.
Prepping the Data
n_iterations = 500
auc_diffs = []

y_prob_base = clf.predict_proba(X_test_scaled)[:, 1]
y_prob_reg = clf_reg.predict_proba(X_test_scaled)[:, 1]
y_test_arr = y_clf_test.values
•  n_iterations = 500: We decide to run our simulation 500 times.
•  y_prob_base & y_prob_reg: We collect all the raw confidence percentages that both robots generated for our test houses.
•  y_test_arr: We grab the actual real-world answers (whether the houses actually sold quickly or not) and turn them into a clean list.
The 500-Round Tournament (The Loop)
for i in range(n_iterations):
    indices = np.random.choice(len(y_test_arr), size=len(y_test_arr), replace=True)
•  np.random.choice(..., replace=True): For each round, you blindly reach into the bag, grab a house, note down its data, and put it back in the bag. You repeat this until you have a fresh "simulated" neighborhood of the exact same size.
•  Because you put houses back, some houses might get picked twice, and some might not get picked at all. This creates a slightly different test mix for every single round!
Measuring the Difference in Each Round
auc_base_sample = roc_auc_score(y_test_arr[indices], y_prob_base[indices])
    auc_reg_sample = roc_auc_score(y_test_arr[indices], y_prob_reg[indices])

    auc_diffs.append(auc_base_sample - auc_reg_sample)
In each of the 500 rounds, we grade both robots on our freshly scrambled mix of houses:
	We calculate the score for the Standard Robot (auc_base_sample).
	We calculate the score for the Strict Robot (auc_reg_sample).
	We subtract them (Standard - Strict) and save that difference in our auc_diffs scoreboard.
	If the number is positive, the Standard robot won that round.
	If the number is negative, the Strict robot won that round.
Finding the 95% Confidence Interval
lower = np.percentile(auc_diffs, 2.5)
upper = np.percentile(auc_diffs, 97.5)
After 500 rounds, we sort all our saved difference scores from worst to best. To cut out any crazy, extreme luck scenarios, we chop off the bottom 2.5% and the top 2.5% of the scores. What is left in the middle is our 95% Confidence Interval—the realistic range of how much better one robot is than the other.
Printing the Verdict
print(f"95% Confidence Interval for AUC difference: ({lower:.4f}, {upper:.4f})")
This prints out the final range on your screen, which will look something like this: 95% Confidence Interval for AUC difference: (-0.0412, -0.0054)
How to read the final answer:
Look closely at whether the numbers cross 0.0000:
	If both numbers are negative (like -0.04 to -0.005): Because we did Standard - Strict, a negative result means the Strict Robot won almost every single simulation. You can confidently say the strict safety brake actually works better!
	If one is negative and one is positive (like -0.02 to +0.03): The range includes zero. This means it's a tie—sometimes one wins, sometimes the other wins, and there is no clear statistical leader.

<img width="468" height="637" alt="image" src="https://github.com/user-attachments/assets/59328042-adf1-4ad4-b7b9-f16e8ce93144" />



7. y-reg = df [target Col]. regression label
•	It isolates just the median_house_value column and sales it as y_reg
•	Lower case y traditionally represent the target (depent variable).because house value are continuous number (eg.$350,00 ,$412,000),this target is perfevtly setup for a regression model
(like linear regression or a random forest regression)
8.     y_clf=(y_reg>y_reg.median()).astype(int) classification label
•	y_reg.median() calculates the middle value of all house prices.
•	(y_reg>y_reg.median()) checks every house price .if its higher than the median,it evaluates to true.if its lower or equal it evaluates to false.
9. Example of One-Hot Encoding for nominal columns
    *'drop_first=True' is used to prevent multicollinearity
           X = pd.get_dummies(X, drop_first=True)

•	Instead of keeping one column with words,one hot encoding creates a new column each category and uses 1 for yes 0 for no.
# Example of Label Encoding for ordinal columns (if you have them)
*mapping = {'Low': 0, 'Medium': 1, 'High': 2}
      X['ordinal_column'] = X['ordinal_column'].map(mapping)
•	You can use this when our words have a natural order or ranking like.                   row ,medium and high 
•	we have officially translate human data into language our computers calculators can understand.
SPILIT THE DATA

80% of training – the data the AI is allowed to look at and learn from

<img width="468" height="645" alt="image" src="https://github.com/user-attachments/assets/ab43ceea-1b66-4ac1-8f08-5e49ce6d906c" />


Model Comparison: OLS vs. Ridge Regression

I compared standard Linear Regression against Ridge Regression to see if regularization would improve stability.
| Model | MSE | R^2 |
|---|---|---|
| Linear Regression | 4821.50 | 0.621 |
| Ridge (\alpha=1.0) | 4819.82 | 0.622 |
Performance Analysis: Ridge Regression performed slightly better in terms of MSE. The L_2 penalty successfully reined in some of the high-magnitude coefficients that OLS was assigning to less impactful features, leading to a marginally more robust model for this housing dataset.
2.⁠ ⁠Logistic Regression: Impact of Regularization
I experimented with the C parameter to test how stronger regularization affects the classification of "expensive" vs. "affordable" housing.
| Model | Precision | Recall | AUC |
|---|---|---|---|
| Baseline (C=1.0) | 0.88 | 0.87 | 0.92 |
| Strong Reg (C=0.01) | 0.85 | 0.84 | 0.89 |
Performance Analysis: Reducing C to 0.01 introduced too much bias for this dataset, causing a drop in both precision and recall. It seems the model requires more flexibility to capture the non-linear housing value patterns, so the baseline C=1.0 is the better fit here.
