import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

rs  = 7810


with open ("data/transform_gd.pkl", 'rb') as fff:
    transform_gd =  pickle.load(file=fff)
with open ("data/target.pkl", 'rb') as fff:
    target = pickle.load(file=fff)

#parameters to tune
parameters = {
 'max_iter': [150, 175, 200, 225, 250, 300, 350, 400],
 'learning_rate': [0.1],
 'min_samples_leaf': [1, 3, 5, 7],
 'l2_regularization': [0.05, 0.03, 0.07],
 'max_leaf_nodes': [7, 10, 13, 15],
 'max_bins': [190, 200, 210],
 'scoring': ['roc_auc'],
 'random_state' : [rs+20],
 }
#instantiate the gridsearch
hgb_grid = GridSearchCV(HistGradientBoostingClassifier(), parameters, n_jobs=-1, 
 cv=5, scoring='roc_auc',
 verbose=3, refit=True)
#fit on the grid 
hgb_grid.fit(transform_gd,target)
import pickle
with open ("hgb_grid.pkl", 'wb') as fff:
    pickle.dump(hgb_grid, file=fff)