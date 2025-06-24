import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self):
        self.cleaning_history = []
        self.encoders = {}
    
    def clean_concrete_dataset(self, input_file='cleaning/SCM-concrete-global.csv', 
                              output_file='SCM-based-concrete-formated.csv'):
        """
        Clean and format the concrete dataset for regression analysis
        """
        try:
            print(f"Loading dataset from: {input_file}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(input_file, encoding=encoding)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not load file with any encoding")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            print(f"Original shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Create column mapping for standardization
            column_mapping = {
                'Cement(kg/m3)': 'cement_opc',
                'Cement (kg/m3)': 'cement_opc',
                'cement': 'cement_opc',
                'FA (kg/m3)': 'scm_flyash',
                'flyash': 'scm_flyash',
                'Fly ash': 'scm_flyash',
                'GGBFS (kg/m3)': 'scm_ggbs',
                'GGBS': 'scm_ggbs',
                'slag': 'scm_ggbs',
                'Fine aggregate(kg/m3)': 'locally_avail_sand',
                'Fine aggregate (kg/m3)': 'locally_avail_sand',
                'sand': 'locally_avail_sand',
                'Splitting tensile strength (MPa)': 'tensile_strength',
                'Tensile strength': 'tensile_strength',
                'tensile': 'tensile_strength',
                'Cylinder compressive strength (MPa)': 'compressive_strength',
                'Compressive strength': 'compressive_strength',
                'compressive': 'compressive_strength',
                'Elastic modulus (GPa)': 'youngs_modulus',
                'Elastic modulus': 'youngs_modulus',
                'modulus': 'youngs_modulus',
                'SP (kg/m3)': 'superplasticizer',
                'Superplasticizer': 'superplasticizer',
                'sp': 'superplasticizer',
                'Water(kg/m3)': 'water',
                'Water (kg/m3)': 'water',
                'water': 'water',
                'Coarse aggregate(kg/m3)': 'coarse_agg',
                'Coarse aggregate (kg/m3)': 'coarse_agg',
                'coarse': 'coarse_agg',
                'aggregate': 'coarse_agg'
            }
            
            # Apply column mapping
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
                    print(f"Renamed '{old_name}' to '{new_name}'")
            
            # Add missing columns with default values
            required_columns = [
                'cement_opc', 'scm_flyash', 'scm_ggbs', 'silica_sand', 
                'locally_avail_sand', 'water', 'superplasticizer', 'coarse_agg',
                'tensile_strength', 'compressive_strength', 'youngs_modulus'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    if col in ['silica_sand', 'perc_of_fibre', 'aspect_ratio', 'elongation']:
                        df[col] = 0
                        print(f"Added missing column '{col}' with default value 0")
                    else:
                        print(f"Warning: Required column '{col}' not found in dataset")
            
            # Ensure numeric types for all relevant columns
            numeric_columns = [
                'cement_opc', 'scm_flyash', 'scm_ggbs', 'locally_avail_sand',
                'water', 'superplasticizer', 'coarse_agg', 'tensile_strength',
                'compressive_strength', 'youngs_modulus'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values before calculations
            print("Handling missing values...")
            original_shape = df.shape
            
            # Drop rows where essential components are completely missing
            essential_cols = ['cement_opc', 'water']
            available_essential = [col for col in essential_cols if col in df.columns]
            
            if available_essential:
                df = df.dropna(subset=available_essential)
                print(f"Dropped rows with missing essential values. Shape: {original_shape} -> {df.shape}")
            
            # Fill missing values for other columns
            for col in numeric_columns:
                if col in df.columns and df[col].isnull().any():
                    if col in ['scm_flyash', 'scm_ggbs', 'superplasticizer']:
                        # These can be zero
                        df[col].fillna(0, inplace=True)
                    else:
                        # Use median for other columns
                        df[col].fillna(df[col].median(), inplace=True)
                    print(f"Filled missing values in '{col}'")
            
            # Calculate derived features
            print("Calculating derived features...")
            
            # Water-binder ratio
            if all(col in df.columns for col in ['cement_opc', 'water']):
                binder_cols = ['cement_opc']
                if 'scm_flyash' in df.columns:
                    binder_cols.append('scm_flyash')
                if 'scm_ggbs' in df.columns:
                    binder_cols.append('scm_ggbs')
                
                binder = df[binder_cols].sum(axis=1)
                df['w_b'] = df['water'] / binder.replace(0, np.nan)
                print("Calculated water-binder ratio")
            
            # HRWR/binder ratio
            if all(col in df.columns for col in ['superplasticizer']) and 'w_b' in df.columns:
                binder = df['water'] / df['w_b'].replace(0, np.nan)
                df['hrwr_b'] = df['superplasticizer'] / binder.replace(0, np.nan)
                print("Calculated HRWR-binder ratio")
            
            # Density calculation
            density_cols = []
            for col in ['cement_opc', 'water', 'coarse_agg', 'locally_avail_sand', 'scm_flyash', 'scm_ggbs']:
                if col in df.columns:
                    density_cols.append(col)
            
            if density_cols:
                df['density'] = df[density_cols].sum(axis=1)
                print("Calculated density")
            
            # Add default values for missing engineered features
            engineered_features = ['perc_of_fibre', 'aspect_ratio', 'elongation']
            for col in engineered_features:
                if col not in df.columns:
                    df[col] = 0
            
            # Estimate missing target values using regression
            if 'tensile_strength' in df.columns and 'compressive_strength' in df.columns:
                self._estimate_missing_tensile_strength(df)
            
            # Define final column set
            final_columns = [
                'cement_opc', 'scm_flyash', 'scm_ggbs', 'silica_sand', 'locally_avail_sand',
                'w_b', 'hrwr_b', 'perc_of_fibre', 'aspect_ratio', 'tensile_strength',
                'density', 'youngs_modulus', 'elongation', 'compressive_strength'
            ]
            
            # Keep only available columns
            available_columns = [col for col in final_columns if col in df.columns]
            df_final = df[available_columns].copy()
            
            # Final cleanup
            df_final = df_final.dropna()
            
            # Validate data quality
            self._validate_data_quality(df_final)
            
            # Save cleaned dataset
            df_final.to_csv(output_file, index=False)
            print(f"✅ Cleaning complete. Saved {df_final.shape[0]} rows and {df_final.shape[1]} columns to '{output_file}'")
            print(f"Final columns: {list(df_final.columns)}")
            
            return df_final
            
        except Exception as e:
            print(f"Error during cleaning: {str(e)}")
            raise
    
    def _estimate_missing_tensile_strength(self, df):
        """Estimate missing tensile strength values using regression"""
        if 'tensile_strength' not in df.columns or 'compressive_strength' not in df.columns:
            return
        
        mask = df['tensile_strength'].isna()
        if not mask.any():
            return
        
        # Use available data to train regression model
        train_data = df.dropna(subset=['compressive_strength', 'tensile_strength'])
        
        if len(train_data) > 5:  # Need minimum data for regression
            try:
                model = LinearRegression()
                X_train = train_data[['compressive_strength']]
                y_train = train_data['tensile_strength']
                
                model.fit(X_train, y_train)
                
                # Predict missing values
                X_pred = df.loc[mask, ['compressive_strength']].dropna()
                if len(X_pred) > 0:
                    predictions = model.predict(X_pred)
                    df.loc[X_pred.index, 'tensile_strength'] = predictions
                    print(f"Estimated {len(predictions)} missing tensile strength values using regression")
            
            except Exception as e:
                print(f"Could not estimate tensile strength: {str(e)}")
    
    def _validate_data_quality(self, df):
        """Validate the cleaned dataset quality"""
        print("\n📊 Data Quality Report:")
        print(f"Shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Check for outliers (simple IQR method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
        
        if outlier_counts:
            print(f"Potential outliers detected: {outlier_counts}")
        else:
            print("No obvious outliers detected")
        
        # Check data ranges
        print("\nData ranges:")
        for col in numeric_cols[:5]:  # Show first 5 columns
            print(f"  {col}: {df[col].min():.2f} to {df[col].max():.2f}")
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """Detect outliers in the dataset"""
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
            
            if len(outliers) > 0:
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'indices': outliers.index.tolist()
                }
        
        return outliers_info
    
    def handle_missing_values(self, df, strategies):
        """Handle missing values based on specified strategies"""
        handled_columns = {}
        
        for col, strategy_info in strategies.items():
            if col not in df.columns:
                continue
            
            strategy = strategy_info.get('method', 'drop')
            custom_value = strategy_info.get('value', None)
            
            original_missing = df[col].isnull().sum()
            
            if strategy == 'drop':
                df = df.dropna(subset=[col])
                handled_columns[col] = f"Dropped {original_missing} rows with missing values"
                
            elif strategy == 'mean' and df[col].dtype in ['int64', 'float64']:
                fill_value = df[col].mean()
                df[col].fillna(fill_value, inplace=True)
                handled_columns[col] = f"Filled {original_missing} missing values with mean ({fill_value:.2f})"
                
            elif strategy == 'median' and df[col].dtype in ['int64', 'float64']:
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                handled_columns[col] = f"Filled {original_missing} missing values with median ({fill_value:.2f})"
                
            elif strategy == 'mode':
                if not df[col].mode().empty:
                    fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                    handled_columns[col] = f"Filled {original_missing} missing values with mode ({fill_value})"
                
            elif strategy == 'forward_fill':
                df[col].fillna(method='ffill', inplace=True)
                handled_columns[col] = f"Forward filled {original_missing} missing values"
                
            elif strategy == 'backward_fill':
                df[col].fillna(method='bfill', inplace=True)
                handled_columns[col] = f"Backward filled {original_missing} missing values"
                
            elif strategy == 'custom_value' and custom_value is not None:
                df[col].fillna(custom_value, inplace=True)
                handled_columns[col] = f"Filled {original_missing} missing values with custom value ({custom_value})"
            
            self.cleaning_history.append(f"Handled missing values in {col}: {handled_columns.get(col, 'No action taken')}")
        
        return df, handled_columns
    
    def encode_categorical_data(self, df, encoding_strategies):
        """Encode categorical columns"""
        encoded_columns = {}
        
        for col, strategy in encoding_strategies.items():
            if col not in df.columns or df[col].dtype != 'object':
                continue
            
            if strategy == 'label':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                encoded_columns[col] = f"Label encoded with {len(le.classes_)} categories"
                
            elif strategy == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col)
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
                encoded_columns[col] = f"One-hot encoded into {len(dummies.columns)} columns"
            
            self.cleaning_history.append(f"Encoded {col}: {encoded_columns.get(col, 'No encoding applied')}")
        
        return df, encoded_columns
    
    def validate_dataset(self, df, target_column=None):
        """Validate dataset for regression training"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check if dataframe is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check target column
        if target_column and target_column not in df.columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Target column '{target_column}' not found")
        
        # Check for sufficient data
        if len(df) < 10:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Insufficient data (less than 10 rows)")
        
        # Check for features
        feature_cols = [col for col in df.columns if col != target_column] if target_column else df.columns[:-1]
        if len(feature_cols) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No feature columns found")
        
        # Check data types
        non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            validation_results['warnings'].append(f"Non-numeric columns detected: {non_numeric_cols}")
            validation_results['recommendations'].append("Consider encoding categorical variables")
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            validation_results['warnings'].append(f"Missing values in columns: {missing_cols}")
            validation_results['recommendations'].append("Handle missing values before training")
        
        # Check for constant columns
        constant_cols = [col for col in feature_cols if df[col].nunique() <= 1]
        if constant_cols:
            validation_results['warnings'].append(f"Constant columns detected: {constant_cols}")
            validation_results['recommendations'].append("Remove constant columns")
        
        # Check target variable for regression
        if target_column:
            target_series = df[target_column]
            if target_series.dtype == 'object':
                validation_results['warnings'].append("Target variable is categorical")
                validation_results['recommendations'].append("Ensure target encoding for regression tasks")
            elif target_series.nunique() < 3:
                validation_results['warnings'].append("Target variable has very few unique values")
        
        return validation_results

class DataValidator:
    """Validate data before training"""
    
    @staticmethod
    def validate_for_training(df, target_column=None):
        """Validate if data is ready for machine learning training"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check if dataframe is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check for target column
        if target_column and target_column not in df.columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Target column '{target_column}' not found")
        
        # Check for sufficient data
        if len(df) < 10:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Insufficient data (less than 10 rows)")
        
        # Check for features
        feature_cols = [col for col in df.columns if col != target_column] if target_column else df.columns[:-1]
        if len(feature_cols) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No feature columns found")
        
        # Check data types
        non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            validation_results['warnings'].append(f"Non-numeric columns detected: {non_numeric_cols}")
            validation_results['recommendations'].append("Consider encoding categorical variables")
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            validation_results['warnings'].append(f"Missing values in columns: {missing_cols}")
            validation_results['recommendations'].append("Handle missing values before training")
        
        # Check for constant columns
        constant_cols = [col for col in feature_cols if df[col].nunique() <= 1]
        if constant_cols:
            validation_results['warnings'].append(f"Constant columns detected: {constant_cols}")
            validation_results['recommendations'].append("Remove constant columns")
        
        # Check target variable for regression
        if target_column:
            target_series = df[target_column]
            if target_series.dtype == 'object':
                validation_results['warnings'].append("Target variable is categorical")
                validation_results['recommendations'].append("Ensure target encoding for regression tasks")
            elif target_series.nunique() < 3:
                validation_results['warnings'].append("Target variable has very few unique values")
        
        return validation_results