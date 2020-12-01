#Importing data anlysis libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") #To ignore harmless warnings
from datetime import datetime, timedelta
sns.set() #Seaborn style background of plots

#Import all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)
sns.set(style='darkgrid')
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from pylab import rcParams
rcParams['figure.dpi'] = 200 #For higher definition plots

# import all libraries and dependencies for machine learning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, plot_confusion_matrix, confusion_matrix
from sklearn.metrics import recall_score, precision_score, precision_recall_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier




#Helper functions to print scores or plot basic curves
def print_scores(y_test, y_pred): #Prints different classification scores
    print("F1 Score = {}".format(f1_score(y_test, y_pred)))
    print("Precision Score = {}".format(precision_score(y_test, y_pred)))
    print("Recall Score = {}".format(recall_score(y_test, y_pred)))
    print(" ")
    print("AUC = {}".format(roc_auc_score(y_test, y_pred)))


def plot_roc(y_test, y_pred, name): #Plots the ROC Curve along with named legend
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'g--')
    plt.plot(fpr, tpr, label="{}, AUC={:.3f}".format(name, roc_auc_score(y_test, y_pred)))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve analysis")
    plt.legend(loc='lower right')
    plt.show()


def plot_pr_curve(y_test, y_pred_proba, name): #Plots the Precision-Recall Curve for classification
    sns.set()
    pr, rc, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 8))
    plt.axhline(y=len(y_test[y_test == 1]) / len(y_test), color='k', linestyle='--',
                label='No skill value = {:.3f}'.format(len(y_test[y_test == 1]) / len(y_test)))
    plt.plot(rc, pr, color='orange', label="{}, AUC={:.3f}".format(name, auc(rc, pr)))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(ticks=[len(y_test[y_test == 1]) / len(y_test), 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    plt.title("Precison-Recall Curve")
    plt.legend(loc='lower right')
    plt.show()


def find_priors(df): #Finding priors for teamwise wins, for Naive Bayes. Returns a dictionary with keys as the team names
    priors = {}
    teams = df['Team'].unique()
    df['win'] = df['win'].astype('int')
    priors = {}
    for t in teams:
        a = df[df['Team'] == t].copy()
        priors[t] = a['win'].sum()/len(a)
    return priors


#Importing the data
df = pd.read_csv('ml_final.csv')
df.drop('Unnamed: 0', axis=1, inplace=True) #Unnecessary extra column of indices appears when loading the data.

#Drop the rows with draws, then resets the index
idx = df[df['win'] == 'draw'].index
df.drop(idx, axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)


def get_last(name):
    _temp1 = df.loc[df["Team"] == name]
    _temp2 = df.loc[df["Team"] == name]
    _temp1 = pd.concat([_temp1, _temp2], axis=0, sort=False)
    return _temp1

team_1=get_last("U MUMBA")
team_2=get_last("DABANG DELHI K.C")
team_3=get_last("PUNERI PALTAN")
team_4=get_last("TELUGU TITANS")
team_5=get_last("PATNA PIRATES")
team_6=get_last("BENGAL WARRIORS")
team_7=get_last("JAIPUR PINK PANTHERS")
team_8=get_last("BENGALURU BULLS")
team_9=get_last("TAMIL THALAIVAS")
team_10=get_last("HARYANA STEELERS")
team_11=get_last("U.P. YODDHA")
team_12=get_last("GUJARAT FORTUNEGIANTS")

#Get the scores of the match between team_1 & team_2, for the Season as season
def get_head_on(team_1, team_2, season):
    _temp1 = df.loc[
        ((df["Team"] == team_1) & (df["Op Team"] == team_2)) | ((df["Team"] == team_2) & (df["Op Team"] == team_1))]
    res = _temp1.loc[_temp1["season"] < season]
    score = 0
    op_score = 0
    for i in res.index:
        if (res['Team'][i] == team_1):
            score += res['score'][i]
            op_score = res['Opscore'][i]
        else:
            score += res['Opscore'][i]
            op_score += res['score'][i]
            # print(score,op_score)
    return score, op_score

#Calculating the current scores from starting as columns 'prev_score' & 'prev_op_score'
df['prev_score'] = np.zeros(len(df), int)
df['prev_op_score'] = np.zeros(len(df), int)
for i in df.index:
    score = 0
    op_score = 0

    score, op_score = get_head_on(df["Team"][i], df["Op Team"][i], df["season"][i])
    df['prev_score'][i] = score
    df['prev_op_score'][i] = op_score

#Assigning teamwise numerical IDs
B=df['Op Team']
A=df['Team']
B=B.drop_duplicates()
#print(B)
id_=1
team_id={}
for i in B:
    team_id[i]=id_
    id_+=1
print(team_id)

#Assigning the winner of the game, by names
df['Winner'] = np.where((df['score'] >df['Opscore']), df['Team'], df['Op Team'])
df['Win_id'] = np.where((df['score'] >df['Opscore']), df['Team'], df['Op Team'])
df=df[df['score'] != df['Opscore']]
for i in df.index:
    df['Win_id'][i]=team_id[df['Winner'][i]]

#Creating a backup file for predicted winners by names
backup1 = df[df['season'] ==6] #validation file
backup2 = df[df['season'] ==7] #test file
backup1.reset_index(drop=True, inplace=True)
backup2.reset_index(drop=True, inplace=True)

#One Hot encoding the teams and dropping the winner id column
final = pd.get_dummies(df, prefix=['Team1', 'Team2'], columns=['Team', 'Op Team'])
df=df.drop(['Win_id'],axis=1)


###Exploratory Data Analysis
print("Plots for EDA: ")
#Heatmap/Correlation plot
def corr_graph(match_df):
    corr = match_df.corr()
    plt.figure(figsize = (10, 8))
    sns.heatmap(corr)
    plt.show()
    return corr
corr_graph(df)

#Bar Plot
df.groupby("Winner")["Winner"].count().plot(kind="bar")
plt.show()

#Violin plots
df2 = pd.read_csv('ml_final.csv') #But first reading in another copy of the same data, so as to not mess up the original one.
df2.drop('Unnamed: 0', axis=1, inplace=True)
idx = df2[df2['win'] == 'draw'].index
df2.drop(idx, axis=0, inplace=True)
plt.show()

#Violin Plot for tackle points
plt.figure(figsize=(16,8))
sns.catplot(x='TacklePoints', y='Team', data=df2, orient="h", height=10, aspect=1, palette="Set3",
                kind="violin", dodge=True, cut=0, bw=.2)
plt.title('Voilin plot of Tackle Points')
plt.show()

#Violin plot for raid points
plt.figure(figsize=(16,8))
sns.catplot(x='RaidPoints', y='Team', data=df2, orient="h", height=10, aspect=1, palette="Set3",
                kind="violin", dodge=True, cut=0, bw=.2)
plt.title('Voilin Plot of Raid Points')
plt.show()

#Violin plot for scores
plt.figure(figsize=(16,8))
sns.catplot(x='score', y='Team', data=df2, orient="h", height=10, aspect=1, palette="Set3",
                kind="violin", dodge=True, cut=0, bw=.2)
plt.title('Voilin Plot of Scores')
plt.show()


######Splitting the data into train, validation and test sets######
df = pd.get_dummies(df, prefix=['Team1', 'Team2'], columns=['Team', 'Op Team'])
final=df.loc[df['season']<6] #The training set. All matches for seasons 1-5
val = df.loc[df['season'] == 6] #The validation set. All matches for season-6
test=df.loc[df['season'] == 7] #The test set. All matches for season 7

#Dropping winner and season columns
final=final.drop(['Winner','season'],axis=1)
val=val.drop(['Winner','season'],axis=1)
test=test.drop(['Winner','season'],axis=1)

#Splitting into features and targets
X_train = final.drop(['win'], axis=1)
y_train = final["win"]

X_test=test.drop(['win'], axis=1)
y_test=test['win']

X_val = val.drop(['win'], axis=1)
y_val = val["win"].astype('int')

y_train = y_train.astype('int') #Just incase they are not of string types
y_test = y_test.astype('int')

print("\n")
print("\n")
print("\n")

#####Supervised Learning Models######

########Decision Tree########
##First trying out an untuned Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("Decision Tree Untuned accuracy = ", accuracy_score(y_test, y_pred)) # = 0.912

#Grid search parameters
params = {'criterion':['gini', 'entropy'], 'max_depth':[2,3,4,5,6], 'min_samples_split':[2,3,4,5,6,7,8], 'max_features':['sqrt', 'log2', None], 'max_leaf_nodes':[2,3,5,6,7,8]}
dt = DecisionTreeClassifier(random_state=0) #Another fresh, unfitted model

#Grid search
gs = GridSearchCV(dt, params, scoring='accuracy', cv=3)
gs.fit(X_train, y_train)

best = gs.best_estimator_ #Best estimator according to grid search
print("Best grid search paramaters {}".format(gs.best_params_)) #Checking out the best parameters

#Predictions on validation set
print("Validation Results for Decision Tree")
yp = best.predict(X_val)
yp_proba = best.predict_proba(X_val)
plot_pr_curve(y_val, yp_proba[:, 1], "Decision Tree (Val)") #Precision recall score
plot_roc(y_val, yp, 'Decision Tree(val)') #ROC curve along with AUC score
print_scores(y_val, yp) #Print different scores
print(classification_report(y_val, best.predict(X_val))) #Another way to get the scores

#Predictions on test set
print("Test set results for Decision Tree")
print(classification_report(y_test, best.predict(X_test)))
plot_roc(y_test, best.predict(X_test), 'Decision tree (test)')
plot_confusion_matrix(best, X_test, y_test)

print("Example predictions for Decision Tree: ")
Y_pred = best.predict(X_test)
for i in backup2.index:
    print(backup2['Team'][i]+" vs "+backup2["Op Team"][i])
    print(backup2['Winner'][i])
    if(Y_pred[i]==1):
        print(backup2["Op Team"][i])
    else:
        print(backup2["Team"][i])

print("\n")
print("\n")
print("\n")


########Logistic Regression########
##Finding the best solver for L2 penalty
params = {'penalty':['l2'], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'max_iter':[5000]} #Kept a high max_iter just to not run into convergence issues
lr = LogisticRegression(random_state=0)
gs = GridSearchCV(lr, param_grid=params, scoring='accuracy', cv=4)
gs.fit(X_train, y_train)
best = gs.best_estimator_
#Val results
yp = best.predict(X_val)
print(confusion_matrix(y_val, yp))
score = best.score(X_train, y_train)
score2 = best.score(X_val, y_val)
print("Logistic Regression with L2 penalty")
print("Training set accuracy: ", '%.3f'%(score))
print("Val set accuracy: ", '%.3f'%(score2))

##Finding best params for L1 penalty
params = {'penalty':['l1'], 'solver':['liblinear', 'saga'], 'max_iter':[2000, 5000, 1000, 3000]}
lr = LogisticRegression(random_state=0)
gs = GridSearchCV(lr, param_grid=params, scoring='accuracy', cv=4)
gs.fit(X_train, y_train)
best = gs.best_estimator_
score = best.score(X_train, y_train)
score2 = best.score(X_val, y_val)
print("Logistic Regression with L1 penalty")
print("Training set accuracy: ", '%.3f'%(score))
print("Val set accuracy: ", '%.3f'%(score2))
print(classification_report(y_val, best.predict(X_val)))
print("Test predictions: {}".format(accuracy_score(y_test, best.predict(X_test))))


##Adding regularization to L1 penalty
params = {'penalty':['l1'], 'solver':['liblinear', 'saga'], 'max_iter':[2000, 5000, 1000, 3000], 'C':[0.2, 0.3, 0.6, 0.7, 0.8, 1.0]}
lr = LogisticRegression(random_state=0)
gs = GridSearchCV(lr, param_grid=params, scoring='accuracy', cv=4)
gs.fit(X_train, y_train)
best = gs.best_estimator_
score = best.score(X_train, y_train)
score2 = best.score(X_val, y_val)
print("Regularized Logistic Regression with L1 penalty")
print("Training set accuracy: ", '%.3f'%(score))
print("Val set accuracy: ", '%.3f'%(score2))
plot_roc(y_val, best.predict(X_val), "Logistic Regression (Val)")
plot_roc(y_test, best.predict(X_test), "Logistic Regression (Test)")
print("Confusion matrix for Logistic regression")
plot_confusion_matrix(best, X_test, y_test)

print("Example predictions for Logistic Regression: ")
Y_pred = best.predict(X_test)
for i in backup2.index:
    print(backup2['Team'][i]+" vs "+backup2["Op Team"][i])
    print(backup2['Winner'][i])
    if(Y_pred[i]==1):
        print(backup2["Op Team"][i])
    else:
        print(backup2["Team"][i])

print("\n")
print("\n")
print("\n")


########Voting Classifier########
##Ensembling the best version of the above two models (Decision Tree + Logistic Regression)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=None, max_leaf_nodes=8, min_samples_split=2, random_state=0)
lr = LogisticRegression(C=0.2, max_iter=2000, penalty='l1', random_state=0, solver='liblinear')

voting_clf = VotingClassifier([('lr', LogisticRegression(C=0.2, max_iter=2000, penalty='l1', random_state=0, solver='liblinear')),
                              ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=None, max_leaf_nodes=8, min_samples_split=2, random_state=0))], voting='soft')
voting_clf.fit(X_train, y_train)
score = voting_clf.score(X_train, y_train)
score2 = voting_clf.score(X_val, y_val)
print("Voting Classifier")
print("Training set accuracy: ", '%.3f'%(score))
print("Val set accuracy: ", '%.3f'%(score2))
score3 = voting_clf.score(X_test, y_test)
print("Test set accuracy: ", '%.3f'%(score3))
print("Classification Report for Voting Classifier")
print(classification_report(y_val, voting_clf.predict(X_val)))
plot_pr_curve(y_val, voting_clf.predict_proba(X_val)[:, 1], 'Voting Classifier (Val)')
plot_roc(y_test, voting_clf.predict(X_test), 'Voting Classifier')
print(classification_report(y_test, voting_clf.predict(X_test)))
plot_confusion_matrix(voting_clf, X_test, y_test)
plt.show()

print("Predicting winners")

Y_pred = voting_clf.predict(X_test)
for i in backup2.index:
    print(backup2['Team'][i]+" vs "+backup2["Op Team"][i])
    print(backup2['Winner'][i])
    if(Y_pred[i]==1):
        print(backup2["Op Team"][i])
    else:
        print(backup2["Team"][i])


########Support Vector Machines########
##Fitting an untuned SVM with RBF Kernel
svm = SVC(random_state=0)
svm.fit(X_train, y_train)
yp_svm = svm.predict(X_val)
print("Accuracy score for SVM with RBF kernal = {}".format(accuracy_score(y_val, yp_svm)))
print(classification_report(y_val, yp_svm))

##Fitting an untuned SVM with Linear Kernel
linear_svm = LinearSVC(random_state=0)
linear_svm.fit(X_train, y_train)
yp_linsvm = linear_svm.predict(X_val)
print("Accuracy score for SVM with RBF kernal = {}".format(accuracy_score(y_val, yp_linsvm)))
yt_linsvm = linear_svm.predict(X_test)
print(classification_report(y_test, yt_linsvm))
plot_roc(y_val, linear_svm.predict(X_val), 'Linear SVM (Val)')
plot_roc(y_test, linear_svm.predict(X_test), 'Linear SVM (Test)')
plot_confusion_matrix(linear_svm, X_test, y_test)
plt.show()


##Regularizing the Linear SVM
svm_params = {'C':[0.2, 0.4, 0.6, 0.7, 0.8, 1]}
lsvc = LinearSVC(random_state=0)
gs = GridSearchCV(lsvc, svm_params, scoring='recall', cv=4)
gs.fit(X_train, y_train)
linsvm = gs.best_estimator_
yp_val = linsvm.predict(X_val)
print("Regularized Linear SVM results")
print("Validation resuts:")
print(classification_report(y_val, yp_val))
plot_roc(y_val, yp_val, 'Regularized Linear SVM (Val)')

print("Test set resuts:")
yp_test = linsvm.predict(X_test)
plot_roc(y_test, yp_test, 'Regularized Linear SVM (test)')
plot_confusion_matrix(linsvm, X_test, y_test)
plt.show()