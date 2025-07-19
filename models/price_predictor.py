
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class VastgoedPrijsPredictor:
    """
    ML model voor het voorspellen van vastgoedprijzen
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.label_encoders = {}
        self.is_trained = False
    
    def prepare_features(self, df):
        """
        Bereidt data voor op ML training
        """
        # Maak een copy om originele data niet te wijzigen
        df_prepared = df.copy()
        
        # Feature engineering
        df_prepared['leeftijd_woning'] = 2024 - df_prepared['bouwjaar']
        df_prepared['prijs_per_m2'] = df_prepared['prijs'] / df_prepared['oppervlakte']
        
        # Encode categorische variabelen
        categorical_columns = ['stad', 'wijk', 'type_woning', 'energielabel']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_prepared[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_prepared[col])
            else:
                df_prepared[f'{col}_encoded'] = self.label_encoders[col].transform(df_prepared[col])
        
        return df_prepared
    
    def train(self, df):
        """
        Train het model op historische data
        """
        print("🚀 Start training van ML model...")
        
        # Prepare features
        df_prepared = self.prepare_features(df)
        
        # Selecteer features voor training
        feature_columns = [
            'oppervlakte', 'kamers', 'leeftijd_woning',
            'stad_encoded', 'wijk_encoded', 'type_woning_encoded', 
            'energielabel_encoded'
        ]
        
        X = df_prepared[feature_columns]
        y = df_prepared['prijs']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evalueer model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model getraind!")
        print(f"📊 Mean Absolute Error: €{mae:,.0f}")
        print(f"📊 R² Score: {r2:.3f}")
        
        self.is_trained = True
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'mae': mae,
            'r2': r2,
            'feature_importance': feature_importance
        }
    
    def predict(self, property_data):
        """
        Voorspel prijs voor een specifiek pand
        """
        if not self.is_trained:
            raise ValueError("Model moet eerst getraind worden!")
        
        # Prepare input data
        df_input = pd.DataFrame([property_data])
        df_prepared = self.prepare_features(df_input)
        
        # Selecteer features
        feature_columns = [
            'oppervlakte', 'kamers', 'leeftijd_woning',
            'stad_encoded', 'wijk_encoded', 'type_woning_encoded', 
            'energielabel_encoded'
        ]
        
        X = df_prepared[feature_columns]
        
        # Voorspelling
        predicted_price = self.model.predict(X)[0]
        
        # Bereken confidence interval (simplified)
        tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
        confidence_lower = np.percentile(tree_predictions, 5)
        confidence_upper = np.percentile(tree_predictions, 95)
        
        return {
            'predicted_price': predicted_price,
            'confidence_interval': (confidence_lower, confidence_upper),
            'price_per_m2': predicted_price / property_data['oppervlakte']
        }
    
    def save_model(self, filepath='models/price_predictor.pkl'):
        """
        Sla het getrainde model op
        """
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'is_trained': self.is_trained
        }, filepath)
        print(f"✅ Model opgeslagen naar {filepath}")
    
    def load_model(self, filepath='models/price_predictor.pkl'):
        """
        Laad een eerder getraind model
        """
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.label_encoders = saved_data['label_encoders']
        self.is_trained = saved_data['is_trained']
        print(f"✅ Model geladen van {filepath}")