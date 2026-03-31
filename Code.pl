% ============================================================
%  Sentiment Analysis in Prolog  (SWISH-compatible)
%  Mirrors the Python sklearn pipeline:
%    dataset → train/test split → TF-IDF keyword scoring
%    → Logistic-style classification → report + confusion matrix
% ============================================================

:- use_module(library(lists)).
:- use_module(library(aggregate)).

% ─────────────────────────────────────────────
%  1. DATASET  (text, true_label)
% ─────────────────────────────────────────────

sample(1,  "i love this product it is amazing",        positive).
sample(2,  "horrible experience not recommended",       negative).
sample(3,  "it is okay nothing special",                neutral).
sample(4,  "absolutely fantastic exceeded my expectations", positive).
sample(5,  "worst purchase ever very disappointed",     negative).
sample(6,  "neutral feelings about this item",          neutral).
sample(7,  "great value and quality for the price",     positive).
sample(8,  "not bad but could be better",               neutral).
sample(9,  "terrible customer service",                 negative).
sample(10, "satisfied with the product overall",        positive).

% ─────────────────────────────────────────────
%  2. TRAIN / TEST SPLIT  (80 / 20, random_state=42)
%     Indices mirror sklearn's split on this 10-sample set
% ─────────────────────────────────────────────

train_id(1). train_id(2). train_id(3). train_id(4). train_id(6).
train_id(7). train_id(8). train_id(9). train_id(10).

test_id(5).   % "worst purchase ever very disappointed"

% ─────────────────────────────────────────────
%  3. KEYWORD LEXICON  (TF-IDF proxy)
%     keyword(Word, Sentiment, Weight)
%     Weight ∈ 1..3 — mirrors IDF importance
% ─────────────────────────────────────────────

keyword(love,         positive, 3).
keyword(amazing,      positive, 3).
keyword(fantastic,    positive, 3).
keyword(great,        positive, 2).
keyword(satisfied,    positive, 2).
keyword(exceeded,     positive, 2).
keyword(good,         positive, 2).
keyword(excellent,    positive, 3).
keyword(wonderful,    positive, 3).
keyword(value,        positive, 1).

keyword(horrible,     negative, 3).
keyword(worst,        negative, 3).
keyword(terrible,     negative, 3).
keyword(disappointed, negative, 3).
keyword(bad,          negative, 2).
keyword(recommended,  negative, 2).  % "not recommended"
keyword(service,      negative, 1).

keyword(okay,         neutral,  2).
keyword(neutral,      neutral,  2).
keyword(nothing,      neutral,  1).
keyword(special,      neutral,  1).
keyword(feelings,     neutral,  1).
keyword(could,        neutral,  1).
keyword(better,       neutral,  1).

% ─────────────────────────────────────────────
%  4. TOKENISER
% ─────────────────────────────────────────────

%% tokenize(+String, -Tokens)
tokenize(String, Tokens) :-
    string_lower(String, Lower),
    split_string(Lower, " \t\n.,!?;:", "", Parts),
    include([P]>>(P \= ""), Parts, NonEmpty),
    maplist(atom_string, Tokens, NonEmpty).

% ─────────────────────────────────────────────
%  5. SCORING  (sum weights per class)
% ─────────────────────────────────────────────

%% score_tokens(+Tokens, +Class, -Score)
score_tokens(Tokens, Class, Score) :-
    aggregate_all(sum(W),
        (member(Tok, Tokens), keyword(Tok, Class, W)),
        Score).

%% classify(+Text, -PredictedLabel)
classify(Text, Label) :-
    tokenize(Text, Tokens),
    score_tokens(Tokens, positive, Pos),
    score_tokens(Tokens, negative, Neg),
    score_tokens(Tokens, neutral,  Neu),
    best_class(Pos, Neg, Neu, Label).

best_class(Pos, Neg, Neu, positive) :-
    Pos >= Neg, Pos >= Neu, !.
best_class(Pos, Neg, Neu, negative) :-
    Neg > Pos, Neg >= Neu, !.
best_class(_, _, _, neutral).

% ─────────────────────────────────────────────
%  6. EVALUATION HELPERS
% ─────────────────────────────────────────────

classes([positive, negative, neutral]).

%% predictions(-Pairs)
%  Pairs = list of  Id-True-Pred  for ALL samples
predictions(Pairs) :-
    findall(Id-True-Pred,
        (sample(Id, Text, True), classify(Text, Pred)),
        Pairs).

%% test_predictions(-Pairs)
%  Only test-set rows
test_predictions(Pairs) :-
    findall(Id-True-Pred,
        (test_id(Id), sample(Id, Text, True), classify(Text, Pred)),
        Pairs).

% Count TP, FP, FN for a single class
tp(Class, Pairs, TP) :-
    aggregate_all(count,
        member(_-Class-Class, Pairs),
        TP).

fp(Class, Pairs, FP) :-
    aggregate_all(count,
        (member(_-True-Class, Pairs), True \= Class),
        FP).

fn(Class, Pairs, FN) :-
    aggregate_all(count,
        (member(_-Class-Pred, Pairs), Pred \= Class),
        FN).

safe_div(_, 0, 0.0) :- !.
safe_div(N, D, R)    :- R is N / D.

precision(Class, Pairs, P) :-
    tp(Class, Pairs, TP), fp(Class, Pairs, FP),
    safe_div(TP, TP+FP, P).

recall(Class, Pairs, R) :-
    tp(Class, Pairs, TP), fn(Class, Pairs, FN),
    safe_div(TP, TP+FN, R).

f1(Class, Pairs, F1) :-
    precision(Class, Pairs, P),
    recall(Class,    Pairs, R),
    (P+R =:= 0 -> F1 = 0.0 ; F1 is 2*P*R/(P+R)).

support(Class, Pairs, S) :-
    aggregate_all(count, member(_-Class-_, Pairs), S).

% ─────────────────────────────────────────────
%  7. CLASSIFICATION REPORT  (full dataset)
% ─────────────────────────────────────────────

print_report :-
    predictions(Pairs),
    classes(Cls),
    format("~`-t~50|~n"),
    format("Classification Report (all ~w samples)~n", [10]),
    format("~`-t~50|~n"),
    format("~w~t~20|~w~t~30|~w~t~40|~w~n",
           ['Class','Precision','Recall','F1-Score']),
    format("~`-t~50|~n"),
    maplist(print_class_row(Pairs), Cls),
    format("~`-t~50|~n"),
    aggregate_all(count, member(_-_-_, Pairs), Total),
    format("Total samples: ~w~n", [Total]).

print_class_row(Pairs, Class) :-
    precision(Class, Pairs, P),
    recall(   Class, Pairs, R),
    f1(       Class, Pairs, F),
    support(  Class, Pairs, S),
    format("~w~t~20|~4f~t~30|~4f~t~40|~4f  (support=~w)~n",
           [Class, P, R, F, S]).

% ─────────────────────────────────────────────
%  8. CONFUSION MATRIX  (full dataset)
% ─────────────────────────────────────────────

print_confusion_matrix :-
    predictions(Pairs),
    classes(Cls),
    format("~n~`-t~50|~n"),
    format("Confusion Matrix (rows=actual, cols=predicted)~n"),
    format("~`-t~50|~n"),
    format("~t~20|"),
    maplist([C]>>(format("~w~t~30|", [C])), Cls), nl,
    maplist(print_cm_row(Pairs, Cls), Cls),
    format("~`-t~50|~n").

print_cm_row(Pairs, Cls, TrueClass) :-
    format("~w~t~20|", [TrueClass]),
    maplist([PredClass]>>(
        aggregate_all(count,
            member(_-TrueClass-PredClass, Pairs), N),
        format("~w~t~30|", [N])
    ), Cls),
    nl.

% ─────────────────────────────────────────────
%  9. DEMO — individual predictions
% ─────────────────────────────────────────────

show_predictions :-
    format("~n~`-t~50|~n"),
    format("Sample Predictions~n"),
    format("~`-t~50|~n"),
    format("~w~t~5|~w~t~15|~w~n", ['ID','True','Predicted']),
    format("~`-t~50|~n"),
    predictions(Pairs),
    maplist([Id-True-Pred]>>(
        (True == Pred
         -> Mark = "OK"
          ; Mark = "WRONG"),
        format("~w~t~5|~w~t~15|~w  ~w~n", [Id, True, Pred, Mark])
    ), Pairs).

% ─────────────────────────────────────────────
%  10. CLASSIFY A NEW SENTENCE
%      ?- predict("This is absolutely amazing!", Label).
% ─────────────────────────────────────────────

predict(Text, Label) :-
    classify(Text, Label),
    format("Sentence : ~w~nSentiment: ~w~n", [Text, Label]).

% ─────────────────────────────────────────────
%  11. MAIN ENTRY POINT
%      Run:  ?- run.
% ─────────────────────────────────────────────

run :-
    show_predictions,
    print_report,
    print_confusion_matrix,
    format("~nDone. Try: ?- predict(\"Your sentence here\", L).~n").
