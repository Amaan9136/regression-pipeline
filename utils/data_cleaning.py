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
        self.cleaning_summary = {}
    
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
                'Fly Ash (kg/m3)': 'scm_flyash',
                'BFS (kg/m3)': 'scm_slag',
                'slag': 'scm_slag',
                'Blast Furnace Slag (kg/m3)': 'scm_slag',
                'SF (kg/m3)': 'scm_silica_fume',
                'silica_fume': 'scm_silica_fume',
                'Silica Fume (kg/m3)': 'scm_silica_fume',
                'Water (kg/m3)': 'water',
                'water': 'water',
                'SP (kg/m3)': 'superplasticizer',
                'superplasticizer': 'superplasticizer',
                'Superplasticizer (kg/m3)': 'superplasticizer',
                'Coarse Aggregate (kg/m3)': 'coarse_aggregate',
                'coarse_aggregate': 'coarse_aggregate',
                'Fine Aggregate (kg/m3)': 'fine_aggregate',
                'fine_aggregate': 'fine_aggregate',
                'Age (days)': 'age_days',
                'age': 'age_days',
                'Compressive Strength (MPa)': 'compressive_strength',
                'compressive_strength': 'compressive_strength',
                'Tensile Strength (MPa)': 'tensile_strength',
                'tensile_strength': 'tensile_strength'
            }
            
            # Apply column mapping
            df = df.rename(columns=column_mapping)
            
            # Handle missing values
            print("\nHandling missing values...")
            missing_info = df.isnull().sum()
            print(f"Missing values per column: {missing_info[missing_info > 0]}")
            
            # Fill missing values with appropriate strategies
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        # Use median for numeric columns
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        # Use mode for categorical columns
                        mode_value = df[col].mode()
                        if len(mode_value) > 0:
                            df[col].fillna(mode_value[0], inplace=True)
                        else:
                            df[col].fillna('Unknown', inplace=True)
            
            # Estimate missing tensile strength if needed
            self._estimate_missing_tensile_strength(df)
            
            # Remove outliers using IQR method
            print("\nRemoving outliers...")
            df_final = self._remove_outliers_iqr(df)
            
            # Validate data quality
            self._validate_data_quality(df_final)
            
            # Save the cleaned dataset
            df_final.to_csv(output_file, index=False)
            print(f"\nSaved {df_final.shape[0]} rows and {df_final.shape[1]} columns to '{output_file}'")
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
    
    def _remove_outliers_iqr(self, df, multiplier=1.5):
        """Remove outliers using IQR method"""
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        outliers_removed = 0
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound))
            outliers_removed += outliers.sum()
            
            # Remove outliers
            df_cleaned = df_cleaned[~outliers]
        
        print(f"Removed {outliers_removed} outliers total")
        return df_cleaned
    
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
            outlier_counts[col] = outliers
        
        total_outliers = sum(outlier_counts.values())
        print(f"Remaining outliers: {total_outliers}")
        
        # Check data types
        print(f"Data types: {df.dtypes.value_counts().to_dict()}")
    
    def get_column_info(self, df):
        """Get detailed column information"""
        column_info = {}
        
        for col in df.columns:
            col_data = df[col]
            info = {
                'dtype': str(col_data.dtype),
                'non_null_count': int(col_data.count()),
                'null_count': int(col_data.isnull().sum()),
                'unique_count': int(col_data.nunique()),
                'memory_usage': int(col_data.memory_usage(deep=True))
            }
            
            if col_data.dtype in ['int64', 'float64']:
                info.update({
                    'mean': float(col_data.mean()) if not col_data.empty else 0,
                    'std': float(col_data.std()) if not col_data.empty else 0,
                    'min': float(col_data.min()) if not col_data.empty else 0,
                    'max': float(col_data.max()) if not col_data.empty else 0,
                    'median': float(col_data.median()) if not col_data.empty else 0
                })
            else:
                # For categorical columns
                if not col_data.empty:
                    top_values = col_data.value_counts().head(5).to_dict()
                    info['top_values'] = {str(k): int(v) for k, v in top_values.items()}
                else:
                    info['top_values'] = {}
            
            column_info[col] = info
        
        return column_info
    
    def validate_data_quality(self, df):
        """Validate data quality and return report"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values_total': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns)
        }
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_info[col] = int(outliers)
        
        quality_report['outliers'] = outlier_info
        return quality_report
    
    def drop_columns(self, df, columns_to_drop):
        """Drop specified columns from dataframe"""
        dropped_columns = []
        df_copy = df.copy()
        
        for col in columns_to_drop:
            if col in df_copy.columns:
                df_copy = df_copy.drop(columns=[col])
                dropped_columns.append(col)
                self.cleaning_history.append(f"Dropped column: {col}")
        
        return df_copy, dropped_columns
    
    def rename_columns(self, df, column_mapping):
        """Rename columns based on mapping"""
        df_copy = df.copy()
        renamed_columns = {}
        
        for old_name, new_name in column_mapping.items():
            if old_name in df_copy.columns:
                df_copy = df_copy.rename(columns={old_name: new_name})
                renamed_columns[old_name] = new_name
                self.cleaning_history.append(f"Renamed column: {old_name} -> {new_name}")
        
        return df_copy, renamed_columns
    
    def handle_missing_values(self, df, strategies):
        """Handle missing values with different strategies"""
        df_copy = df.copy()
        handled_columns = {}
        
        for col, strategy in strategies.items():
            if col not in df_copy.columns:
                continue
            
            missing_count = df_copy[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
                handled_columns[col] = f"Dropped {missing_count} rows with missing values"
                
            elif strategy == 'mean' and df_copy[col].dtype in ['int64', 'float64']:
                mean_value = df_copy[col].mean()
                df_copy[col].fillna(mean_value, inplace=True)
                handled_columns[col] = f"Filled {missing_count} missing values with mean ({mean_value:.2f})"
                
            elif strategy == 'median' and df_copy[col].dtype in ['int64', 'float64']:
                median_value = df_copy[col].median()
                df_copy[col].fillna(median_value, inplace=True)
                handled_columns[col] = f"Filled {missing_count} missing values with median ({median_value:.2f})"
                
            elif strategy == 'mode':
                mode_value = df_copy[col].mode()
                if len(mode_value) > 0:
                    df_copy[col].fillna(mode_value[0], inplace=True)
                    handled_columns[col] = f"Filled {missing_count} missing values with mode ({mode_value[0]})"
                
            elif strategy == 'constant':
                fill_value = strategies.get(f'{col}_fill_value', 0)
                df_copy[col].fillna(fill_value, inplace=True)
                handled_columns[col] = f"Filled {missing_count} missing values with constant ({fill_value})"
            
            self.cleaning_history.append(f"Handled missing values in {col}: {handled_columns.get(col, 'No action')}")
        
        return df_copy, handled_columns
    
    def encode_categorical_data(self, df, encoding_strategies):
        """Encode categorical columns"""
        df_copy = df.copy()
        encoded_columns = {}
        
        for col, strategy in encoding_strategies.items():
            if col not in df_copy.columns:
                continue
            
            if df_copy[col].dtype == 'object' or df_copy[col].dtype.name == 'category':
                if strategy == 'label':
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.encoders[col] = le
                    encoded_columns[col] = f"Label encoded with {len(le.classes_)} categories"
                    
                elif strategy == 'onehot':
                    dummies = pd.get_dummies(df_copy[col], prefix=col)
                    df_copy = df_copy.drop(columns=[col])
                    df_copy = pd.concat([df_copy, dummies], axis=1)
                    encoded_columns[col] = f"One-hot encoded into {len(dummies.columns)} columns"
                
                self.cleaning_history.append(f"Encoded {col}: {encoded_columns.get(col, 'No encoding applied')}")
        
        return df_copy, encoded_columns
    
    def apply_transformations(self, df, transformations):
        """Apply mathematical transformations to columns"""
        df_copy = df.copy()
        transformed_columns = {}
        
        for col, transformation in transformations.items():
            if col not in df_copy.columns:
                continue
            
            if df_copy[col].dtype not in ['int64', 'float64']:
                continue
            
            try:
                if transformation == 'log':
                    # Add small constant to avoid log(0)
                    df_copy[col] = np.log(df_copy[col] + 1e-8)
                    transformed_columns[col] = "Log transformation applied"
                    
                elif transformation == 'sqrt':
                    # Handle negative values
                    df_copy[col] = np.sqrt(np.abs(df_copy[col]))
                    transformed_columns[col] = "Square root transformation applied"
                    
                elif transformation == 'square':
                    df_copy[col] = df_copy[col] ** 2
                    transformed_columns[col] = "Square transformation applied"
                    
                elif transformation == 'reciprocal':
                    # Add small constant to avoid division by zero
                    df_copy[col] = 1 / (df_copy[col] + 1e-8)
                    transformed_columns[col] = "Reciprocal transformation applied"
                
                self.cleaning_history.append(f"Transformed {col}: {transformed_columns.get(col, 'No transformation')}")
                
            except Exception as e:
                print(f"Error transforming column {col}: {str(e)}")
        
        return df_copy, transformed_columns
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """Detect outliers in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            outliers_indices = []
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_indices = df[outliers_mask].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_mask = z_scores > threshold
                outliers_indices = df[outliers_mask].index.tolist()
            
            outliers_info[col] = {
                'count': len(outliers_indices),
                'indices': outliers_indices[:100],  # Limit to first 100 for performance
                'percentage': (len(outliers_indices) / len(df)) * 100 if len(df) > 0 else 0
            }
        
        return outliers_info
    
    def remove_outliers(self, df, outlier_indices):
        """Remove specified outlier rows"""
        df_copy = df.copy()
        initial_shape = df_copy.shape
        
        # Remove outliers by index
        df_copy = df_copy.drop(index=outlier_indices, errors='ignore')
        
        removed_count = initial_shape[0] - df_copy.shape[0]
        self.cleaning_history.append(f"Removed {removed_count} outlier rows")
        
        return df_copy, removed_count
    
    def get_cleaning_summary(self):
        """Return summary of all cleaning operations performed"""
        return {
            'operations_performed': len(self.cleaning_history),
            'history': self.cleaning_history.copy(),
            'encoders_used': list(self.encoders.keys()),
            'summary': self.cleaning_summary.copy()
        }

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