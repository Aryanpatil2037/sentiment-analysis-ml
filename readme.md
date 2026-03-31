🧠 Sentiment Analysis in Prolog (SWISH Compatible)

This project implements a Sentiment Analysis system using Prolog, designed to mirror a typical machine learning pipeline (like Python’s sklearn) using logical rules and symbolic processing.

It classifies text into three categories:

✅ Positive

❌ Negative

⚪ Neutral


🚀 Features
🔹 Custom dataset of labeled sentences

🔹 Train/Test split (80/20)

🔹 Keyword-based scoring (TF-IDF inspired)

🔹 Rule-based classification (logistic-style decision)

🔹 Evaluation metrics:


Precision
Recall
F1-score

🔹 Confusion Matrix

🔹 Predict sentiment of new sentences


📂 Project Structure
sentiment-prolog/
│── sentiment.pl     % Main Prolog code
│── README.md        % Project documentation
⚙️ Requirements

You can run this project using:

🧩 SWI-Prolog (recommended)

🌐 SWISH (Online Prolog IDE)

▶️ How to Run
🔹 Option 1: Run in SWI-Prolog (Local)
Install SWI-Prolog: https://www.swi-prolog.org/
Open terminal and navigate to project folder
Run:
swipl
Load the file:
?- [sentiment].
Execute the program:
?- run.


🔹 Option 2: Run in SWISH (Online)
Open: https://swish.swi-prolog.org/
Copy and paste the code into the editor
Click Run
Execute:
?- run.


📊 Sample Output
✔ Predictions
ID   True        Predicted
1    positive    positive  OK
2    negative    negative  OK


...
📈 Classification Report
Class       Precision   Recall   F1-Score
positive    1.0000      1.0000   1.0000
negative    1.0000      1.0000   1.0000
neutral     1.0000      1.0000   1.0000
🔢 Confusion Matrix
              Predicted
            pos   neg   neu
Actual pos   4     0     0
Actual neg   0     3     0
Actual neu   0     0     3
🧪 Test Your Own Sentence


Use the predict/2 predicate:

?- predict("This is absolutely amazing!", Label).
Example Output:
Sentence : This is absolutely amazing!
Sentiment: positive
🧠 How It Works
1. Tokenization
Converts sentence into lowercase words
2. Keyword Matching
Matches words with predefined sentiment lexicon
3. Scoring
Assigns weights (1–3) to words (similar to TF-IDF importance)
4. Classification Logic
Chooses sentiment with highest score
📌 Example Rules
keyword(amazing, positive, 3).
keyword(worst, negative, 3).
keyword(okay, neutral, 2).
📉 Evaluation Metrics
Precision = Correct positive predictions / Total predicted positives
Recall = Correct positive predictions / Actual positives
F1 Score = Harmonic mean of precision & recall
🎯 Learning Outcomes

This project helps understand:

Logic Programming with Prolog
Rule-based NLP systems
How ML concepts can be simulated without ML libraries
Evaluation metrics used in classification
🔮 Future Improvements
Add larger dataset
Implement real TF-IDF calculation
Use NLP preprocessing (stopwords, stemming)
Integrate with Python (hybrid system)
👤 Author

Aryan Patil
B.Tech CSE Student
