import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from matplotlib.colors import ListedColormap

train = pd.read_csv("/Users/tomragus/Library/CloudStorage/OneDrive-UCSanDiego/CMI-PIU-Model/data/train.csv")
actigraphy = pd.read_csv("/Users/tomragus/Library/CloudStorage/OneDrive-UCSanDiego/CMI-PIU-Model/data/stats.csv")
test = pd.read_csv("/Users/tomragus/Library/CloudStorage/OneDrive-UCSanDiego/CMI-PIU-Model/data/test.csv")
data_dict = pd.read_csv("/Users/tomragus/Library/CloudStorage/OneDrive-UCSanDiego/CMI-PIU-Model/data/data_dictionary.csv")
sample_sub = pd.read_csv("/Users/tomragus/Library/CloudStorage/OneDrive-UCSanDiego/CMI-PIU-Model/data/sample_submission.csv")

sns.set_theme(style="whitegrid")


display(train.head())
print(f"Train shape: {train.shape}")

display(actigraphy.head())
print(f"Actigraphy shape: {actigraphy.shape}")

display(test.head())
print(f"Test shape: {test.shape}")

display(data_dict.head())

class CMIPreprocessor:
    """
    Comprehensive preprocessing pipeline for CMI Problematic Internet Use dataset
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.engineered_features = []
        
    def load_and_initial_clean(self, train_path, test_path):
        """Load data and perform initial cleaning"""
        self.train_df = pd.read_csv("/Users/tomragus/Library/CloudStorage/OneDrive-UCSanDiego/CMI-PIU-Model/data/train.csv")
        self.test_df = pd.read_csv("/Users/tomragus/Library/CloudStorage/OneDrive-UCSanDiego/CMI-PIU-Model/data/test.csv")
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
        # Remove completely empty rows
        self.train_df = self.train_df.dropna(how='all')
        self.test_df = self.test_df.dropna(how='all')

        # ️Drop rows where the target is missing
        self.train_df = self.train_df.dropna(subset=['sii'])
        
        # Identify target variable
        self.target = 'sii'
        
        # Separate features and target
        self.y_train = self.train_df[self.target].copy()

        assert self.y_train.isnull().sum() == 0, "Target contains missing values!"
        
        self.X_train = self.train_df.drop(columns=[self.target]).copy()
        self.X_test = self.test_df.copy()
        
        print(f"After cleaning - Training: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self
    
    def analyze_missing_data(self):
        """Analyze missing data patterns"""
        print("\n=== MISSING DATA ANALYSIS ===")
        
        # Calculate missing percentages
        train_missing = (self.X_train.isnull().sum() / len(self.X_train)) * 100
        test_missing = (self.X_test.isnull().sum() / len(self.X_test)) * 100
        
        missing_df = pd.DataFrame({
            'Train_Missing_%': train_missing,
            'Test_Missing_%': test_missing
        }).sort_values('Train_Missing_%', ascending=False)
        
        print("Top 15 features with most missing data:")
        print(missing_df.head(15))
        
        # Features with >80% missing
        high_missing = missing_df[missing_df['Train_Missing_%'] > 80].index.tolist()
        print(f"\nFeatures with >80% missing data ({len(high_missing)}): {high_missing}")
        
        return missing_df
    
    def create_feature_groups(self):
        """Group related features for better handling"""
        self.feature_groups = {
            'demographics': ['Basic_Demos-Age', 'Basic_Demos-Sex', 'Basic_Demos-Enroll_Season'],
            'physical_basic': ['Physical-BMI', 'Physical-Height', 'Physical-Weight', 
                             'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 
                             'Physical-HeartRate', 'Physical-Systolic_BP'],
            'fitness': ['Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins', 
                       'Fitness_Endurance-Time_Sec'],
            'functional_tests': [col for col in self.X_train.columns if 'FGC-' in col],
            'body_composition': [col for col in self.X_train.columns if 'BIA-' in col],
            'activity_questionnaires': [col for col in self.X_train.columns if 'PAQ_' in col],
            'psychological': [col for col in self.X_train.columns if 'PCIAT-' in col],
            'other_assessments': [col for col in self.X_train.columns if 'SDS-' in col or 'CGAS-' in col],
            'education': [col for col in self.X_train.columns if 'PreInt_EduHx-' in col]
        }
        
        print("\n=== FEATURE GROUPS ===")
        for group, features in self.feature_groups.items():
            print(f"{group}: {len(features)} features")
        
        return self
    
    def engineer_features(self):
        """Create engineered features"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Combine train and test for consistent feature engineering
        combined_df = pd.concat([self.X_train, self.X_test], ignore_index=True)
        
        # 1. Physical Health Composite Features
        if 'Physical-BMI' in combined_df.columns:
            # BMI categories
            combined_df['BMI_category'] = pd.cut(combined_df['Physical-BMI'], 
                                               bins=[0, 18.5, 25, 30, 100], 
                                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            
            # Waist-to-height ratio (if both available)
            if 'Physical-Waist_Circumference' in combined_df.columns and 'Physical-Height' in combined_df.columns:
                combined_df['Waist_Height_Ratio'] = combined_df['Physical-Waist_Circumference'] / combined_df['Physical-Height']
        
        # 2. Blood Pressure Categories
        if 'Physical-Systolic_BP' in combined_df.columns and 'Physical-Diastolic_BP' in combined_df.columns:
            combined_df['BP_Category'] = 'Normal'
            combined_df.loc[(combined_df['Physical-Systolic_BP'] >= 140) | 
                           (combined_df['Physical-Diastolic_BP'] >= 90), 'BP_Category'] = 'High'
            combined_df.loc[(combined_df['Physical-Systolic_BP'] >= 120) & 
                           (combined_df['Physical-Systolic_BP'] < 140), 'BP_Category'] = 'Elevated'
        
        # 3. Fitness Performance Index
        fitness_cols = ['Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins']
        if all(col in combined_df.columns for col in fitness_cols):
            # Normalize and combine fitness metrics
            combined_df['Fitness_Index'] = (
                combined_df['Fitness_Endurance-Max_Stage'].fillna(0) * 0.6 +
                combined_df['Fitness_Endurance-Time_Mins'].fillna(0) * 0.4
            )
        
        # 4. PCIAT Subscales (if PCIAT items available)
        pciat_cols = [col for col in combined_df.columns if 'PCIAT-PCIAT_' in col and col != 'PCIAT-PCIAT_Total']
        if len(pciat_cols) >= 10:
            # Create subscales based on common PCIAT factor structure
            # Compulsive Use (items 1, 2, 3, 4, 5)
            compulsive_items = [f'PCIAT-PCIAT_{i:02d}' for i in range(1, 6) if f'PCIAT-PCIAT_{i:02d}' in combined_df.columns]
            if compulsive_items:
                combined_df['PCIAT_Compulsive'] = combined_df[compulsive_items].mean(axis=1)
            
            # Neglect (items 6, 7, 8, 9, 10)
            neglect_items = [f'PCIAT-PCIAT_{i:02d}' for i in range(6, 11) if f'PCIAT-PCIAT_{i:02d}' in combined_df.columns]
            if neglect_items:
                combined_df['PCIAT_Neglect'] = combined_df[neglect_items].mean(axis=1)
            
            # Control (items 11-20)
            control_items = [f'PCIAT-PCIAT_{i:02d}' for i in range(11, 21) if f'PCIAT-PCIAT_{i:02d}' in combined_df.columns]
            if control_items:
                combined_df['PCIAT_Control'] = combined_df[control_items].mean(axis=1)
        
        # 5. Age-adjusted features
        if 'Basic_Demos-Age' in combined_df.columns:
            # Age groups
            combined_df['Age_Group'] = pd.cut(combined_df['Basic_Demos-Age'], 
                                           bins=[0, 8, 12, 16, 25], 
                                           labels=['Child', 'PreTeen', 'Teen', 'Young_Adult'])
            
            # Age-adjusted BMI percentiles (simplified)
            if 'Physical-BMI' in combined_df.columns:
                combined_df['BMI_Age_Adjusted'] = combined_df['Physical-BMI'] / (combined_df['Basic_Demos-Age'] / 10)
        
        # 6. Body Composition Ratios
        if 'BIA-BIA_Fat' in combined_df.columns and 'BIA-BIA_FFM' in combined_df.columns:
            combined_df['Fat_to_FFM_Ratio'] = combined_df['BIA-BIA_Fat'] / (combined_df['BIA-BIA_FFM'] + 1e-6)
        
        # 7. Activity vs Screen Time Balance
        if 'PAQ_A-PAQ_A_Total' in combined_df.columns and 'PreInt_EduHx-computerinternet_hoursday' in combined_df.columns:
            combined_df['Activity_Screen_Balance'] = combined_df['PAQ_A-PAQ_A_Total'] / (combined_df['PreInt_EduHx-computerinternet_hoursday'] + 1e-6)
        
        # 8. Missingness indicators for important features
        important_features = ['Physical-BMI', 'PCIAT-PCIAT_Total', 'PAQ_A-PAQ_A_Total']
        for feature in important_features:
            if feature in combined_df.columns:
                combined_df[f'{feature}_missing'] = combined_df[feature].isnull().astype(int)
        
        # Split back to train and test
        self.X_train_engineered = combined_df.iloc[:len(self.X_train)].copy()
        self.X_test_engineered = combined_df.iloc[len(self.X_train):].copy()
        
        # Track engineered features
        original_features = set(self.X_train.columns)
        new_features = set(self.X_train_engineered.columns) - original_features
        self.engineered_features = list(new_features)
        
        print(f"Created {len(self.engineered_features)} engineered features:")
        print(self.engineered_features)
        
        return self
    
    def handle_missing_values(self):
        """Smart missing value imputation"""
        print("\n=== MISSING VALUE IMPUTATION ===")
        
        # Separate numerical and categorical features
        numerical_features = self.X_train_engineered.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.X_train_engineered.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Handle categorical features
        for col in categorical_features:
            if col in self.X_train_engineered.columns:
                # Use mode for categorical
                mode_val = self.X_train_engineered[col].mode()[0] if not self.X_train_engineered[col].mode().empty else 'Unknown'
                self.X_train_engineered[col] = self.X_train_engineered[col].fillna(mode_val)
                self.X_test_engineered[col] = self.X_test_engineered[col].fillna(mode_val)
                
                # Label encode categorical features
                le = LabelEncoder()
                combined_values = pd.concat([self.X_train_engineered[col], self.X_test_engineered[col]])
                le.fit(combined_values)
                self.X_train_engineered[col] = le.transform(self.X_train_engineered[col])
                self.X_test_engineered[col] = le.transform(self.X_test_engineered[col])
                self.label_encoders[col] = le
        
        # Handle numerical features with different strategies
        # 1. Simple imputation for basic features
        basic_features = ['Basic_Demos-Age', 'Physical-BMI', 'Physical-Height', 'Physical-Weight']
        basic_numerical = [col for col in basic_features if col in numerical_features]
        
        if basic_numerical:
            imputer_basic = SimpleImputer(strategy='median')
            self.X_train_engineered[basic_numerical] = imputer_basic.fit_transform(self.X_train_engineered[basic_numerical])
            self.X_test_engineered[basic_numerical] = imputer_basic.transform(self.X_test_engineered[basic_numerical])
            self.imputers['basic'] = imputer_basic
        
        # 2. KNN imputation for correlated features (body composition)
        bia_features = [col for col in numerical_features if 'BIA-' in col]
        if len(bia_features) > 3:
            imputer_knn = KNNImputer(n_neighbors=5)
            self.X_train_engineered[bia_features] = imputer_knn.fit_transform(self.X_train_engineered[bia_features])
            self.X_test_engineered[bia_features] = imputer_knn.transform(self.X_test_engineered[bia_features])
            self.imputers['bia'] = imputer_knn
        
        # 3. Zero imputation for questionnaire items (assuming missing = not endorsed)
        questionnaire_features = [col for col in numerical_features if any(x in col for x in ['PCIAT-PCIAT_', 'PAQ_', 'SDS-'])]
        if questionnaire_features:
            self.X_train_engineered[questionnaire_features] = self.X_train_engineered[questionnaire_features].fillna(0)
            self.X_test_engineered[questionnaire_features] = self.X_test_engineered[questionnaire_features].fillna(0)
        
        # 4. Median imputation for remaining features
        remaining_features = [col for col in numerical_features 
                            if col not in basic_numerical + bia_features + questionnaire_features]
        if remaining_features:
            imputer_remaining = SimpleImputer(strategy='median')
            self.X_train_engineered[remaining_features] = imputer_remaining.fit_transform(self.X_train_engineered[remaining_features])
            self.X_test_engineered[remaining_features] = imputer_remaining.transform(self.X_test_engineered[remaining_features])
            self.imputers['remaining'] = imputer_remaining
        
        print("Missing value imputation completed!")
        print(f"Remaining missing values in train: {self.X_train_engineered.isnull().sum().sum()}")
        print(f"Remaining missing values in test: {self.X_test_engineered.isnull().sum().sum()}")
        
        return self
    
    def feature_selection(self, method='mutual_info', k=40):
        """Select most relevant features"""
        print(f"\n=== FEATURE SELECTION ({method}) ===")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)
        
        self.X_train_selected = selector.fit_transform(self.X_train_engineered, self.y_train)
        self.X_test_selected = selector.transform(self.X_test_engineered)
        
        # Get selected feature names
        selected_features = self.X_train_engineered.columns[selector.get_support()].tolist()
        self.selected_features = selected_features
        
        print(f"Selected {len(selected_features)} features out of {self.X_train_engineered.shape[1]}")
        print("Top 10 selected features:")
        feature_scores = selector.scores_[selector.get_support()]
        top_features = sorted(zip(selected_features, feature_scores), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in top_features:
            print(f"  {feature}: {score:.3f}")
        
        return self
    
    def scale_features(self, method='robust'):
        """Scale features for modeling"""
        print(f"\n=== FEATURE SCALING ({method}) ===")
        
        if method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        self.X_train_scaled = scaler.fit_transform(self.X_train_selected)
        self.X_test_scaled = scaler.transform(self.X_test_selected)
        self.scaler = scaler
        
        print("Feature scaling completed!")
        
        return self
    
    def get_processed_data(self):
        """Return processed data for modeling"""
        return {
            'X_train': self.X_train_scaled,
            'X_test': self.X_test_scaled,
            'y_train': self.y_train,
            'feature_names': self.selected_features,
            'original_train_df': self.train_df,
            'original_test_df': self.test_df
        }

# Model Training and Evaluation Class
class CMIModelTrainer:
    """
    Model training and evaluation for CMI dataset
    """
    
    def __init__(self, X_train, y_train, X_test, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
        
    def train_baseline_models(self):
        """Train baseline models"""
        print("\n=== TRAINING BASELINE MODELS ===")
        
        # Define models
        models = {
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Fit model
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            
            # Store results
            self.results[name] = {
                'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
                'cv_rmse_std': np.sqrt(cv_scores.std()),
                'cv_scores': cv_scores
            }
            
            print(f"  CV RMSE: {self.results[name]['cv_rmse_mean']:.4f} (+/- {self.results[name]['cv_rmse_std']:.4f})")
    
    def train_advanced_models(self):
        """Train advanced models (XGBoost, LightGBM)"""
        print("\n=== TRAINING ADVANCED MODELS ===")
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model
        
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
        self.results['XGBoost'] = {
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std()),
            'cv_scores': cv_scores
        }
        
        print(f"XGBoost CV RMSE: {self.results['XGBoost']['cv_rmse_mean']:.4f} (+/- {self.results['XGBoost']['cv_rmse_std']:.4f})")
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        
        lgb_model.fit(self.X_train, self.y_train)
        self.models['LightGBM'] = lgb_model
        
        cv_scores = cross_val_score(lgb_model, self.X_train, self.y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
        self.results['LightGBM'] = {
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std()),
            'cv_scores': cv_scores
        }
        
        print(f"LightGBM CV RMSE: {self.results['LightGBM']['cv_rmse_mean']:.4f} (+/- {self.results['LightGBM']['cv_rmse_std']:.4f})")
    
    def get_best_model(self):
        """Get the best performing model"""
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['cv_rmse_mean'])
        return best_model_name, self.models[best_model_name]
    
    def generate_predictions(self, model_name=None):
        """Generate predictions for test set"""
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        model = self.models[model_name]
        predictions = model.predict(self.X_test)
        
        return predictions, model_name

# Usage Example
def main():
    """
    Main preprocessing and modeling pipeline
    """
    # Initialize preprocessor
    preprocessor = CMIPreprocessor()
    
    # Load and preprocess data
    preprocessor.load_and_initial_clean('train.csv', 'test.csv')
    preprocessor.analyze_missing_data()
    preprocessor.create_feature_groups()
    preprocessor.engineer_features()
    preprocessor.handle_missing_values()
    preprocessor.feature_selection(method='mutual_info', k=40)
    preprocessor.scale_features(method='robust')
    
    # Get processed data
    data = preprocessor.get_processed_data()
    
    # Initialize model trainer
    trainer = CMIModelTrainer(data['X_train'], data['y_train'], 
                            data['X_test'], data['feature_names'])
    
    # Train models
    trainer.train_baseline_models()
    trainer.train_advanced_models()
    
    # Get best model and predictions
    best_model_name, best_model = trainer.get_best_model()
    predictions, model_name = trainer.generate_predictions()
    
    print(f"\nBest model: {best_model_name}")
    print(f"Predictions shape: {predictions.shape}")
    
    print("Preprocessing pipeline ready!")
    print("Uncomment the lines in main() to run the full pipeline")

if __name__ == "__main__":
   main()