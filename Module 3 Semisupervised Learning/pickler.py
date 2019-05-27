import pickle
from model import KickstarterModel

my_model = KickstarterModel()
X_train, y_train = my_model.preprocess_training_data("data/train.csv")
my_model.fit(X_train, y_train)

# Pickle model
with open('model.pickle', 'wb') as f:
    pickle.dump(my_model, f)
