from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

class ModelTrainer:
    def __init__(self, df, target="price"):
        self.X = df.drop(columns=[target])
        self.y = df[target]

    def split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        return self

    def build_and_fit(self, model):
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median"),   # imputation — learned on train fold
            StandardScaler(),                   # rescaling — learned on train fold
            model
        )
        self.pipeline.fit(self.X_train, self.y_train)
        return self

    def evaluate(self):
        from sklearn.metrics import mean_absolute_error, r2_score
        preds = self.pipeline.predict(self.X_test)
        print("R²:", r2_score(self.y_test, preds))
        print("MAE:", mean_absolute_error(self.y_test, preds))
        return self

    def predict(self, X_new):
        return self.pipeline.predict(X_new)