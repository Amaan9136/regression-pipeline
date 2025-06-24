import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler

class DataCleaner:
    def __init__(self):
        self.encoding_mappings = {}
        self.column_mappings = {}
        self.original_columns = []
        self.cleaning_history = []
    
    def preview_data(self, file_path, nrows=100):
        """Preview the first few rows of the dataset"""
        try:
            df = pd.read_csv(file_path, nrows=nrows)
            
            preview_info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'head': df.head().to_dict('records'),
                'missing_values': df.isnull().sum().to_dict(),
                'summary_stats': df.describe().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'duplicate_rows': df.duplicated().sum()
            }
            
            # Identify potential issues
            issues = []
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    issues.append(f"Missing values in '{col}': {df[col].isnull().sum()}")
                
                if df[col].dtype == 'object':
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio > 0.9:
                        issues.append(f"High cardinality in '{col}': {df[col].nunique()} unique values")
            
            preview_info['potential_issues'] = issues
            
            return preview_info
            
        except Exception as e:
            raise Exception(f"Error previewing data: {str(e)}")
    
    def get_column_info(self, df):
        """Get detailed information about each column"""
        column_info = {}
        
        for col in df.columns:
            info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'unique_values': int(df[col].nunique()),
                'sample_values': df[col].dropna().head(5).tolist()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                info.update({
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'skewness': float(df[col].skew()) if not df[col].isnull().all() else None
                })
            else:
                info.update({
                    'top_values': df[col].value_counts().head(5).to_dict()
                })
            
            column_info[col] = info
        
        return column_info
    
    def drop_columns(self, df, columns_to_drop):
        """Drop specified columns from the dataframe"""
        dropped_columns = []
        
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                dropped_columns.append(col)
                self.cleaning_history.append(f"Dropped column: {col}")
        
        return df, dropped_columns
    
    def rename_columns(self, df, rename_mapping):
        """Rename columns according to the mapping"""
        renamed_columns = {}
        
        for old_name, new_name in rename_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                renamed_columns[old_name] = new_name
                self.column_mappings[old_name] = new_name
                self.cleaning_history.append(f"Renamed column: {old_name} -> {new_name}")
        
        return df, renamed_columns
    
    def handle_missing_values(self, df, strategies):
        """
        Handle missing values according to specified strategies
        
        strategies: dict with column names as keys and strategy as values
        Possible strategies: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'custom_value'
        """
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
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                handled_columns[col] = f"Filled {original_missing} missing values with mean ({mean_val:.2f})"
                
            elif strategy == 'median' and df[col].dtype in ['int64', 'float64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                handled_columns[col] = f"Filled {original_missing} missing values with median ({median_val:.2f})"
                
            elif strategy == 'mode':
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    handled_columns[col] = f"Filled {original_missing} missing values with mode ({mode_val})"
                
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
        """
        Encode categorical data according to specified strategies
        
        encoding_strategies: dict with column names as keys and encoding info as values
        Possible encodings: 'label', 'onehot', 'ordinal'
        """
        encoded_columns = {}
        
        for col, strategy_info in encoding_strategies.items():
            if col not in df.columns:
                continue
                
            encoding_type = strategy_info.get('method', 'label')
            
            if encoding_type == 'label':
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoding_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                    encoded_columns[col] = f"Label encoded with {len(le.classes_)} categories"
                    
            elif encoding_type == 'onehot':
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    # Create dummy variables
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = df.drop(columns=[col])
                    df = pd.concat([df, dummies], axis=1)
                    encoded_columns[col] = f"One-hot encoded into {len(dummies.columns)} columns"
                    
            elif encoding_type == 'ordinal':
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    # Custom ordinal mapping if provided
                    ordinal_mapping = strategy_info.get('mapping', {})
                    if ordinal_mapping:
                        df[col] = df[col].map(ordinal_mapping)
                        encoded_columns[col] = f"Ordinal encoded with custom mapping"
                    else:
                        # Default ordinal encoding
                        unique_vals = sorted(df[col].unique())
                        ordinal_map = {val: idx for idx, val in enumerate(unique_vals)}
                        df[col] = df[col].map(ordinal_map)
                        encoded_columns[col] = f"Ordinal encoded with {len(unique_vals)} levels"
            
            self.cleaning_history.append(f"Encoded {col}: {encoded_columns.get(col, 'No encoding applied')}")
        
        return df, encoded_columns
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """Detect outliers in numerical columns"""
        outliers_info = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            outliers = []
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold].index.tolist()
            
            outliers_info[col] = {
                'count': len(outliers),
                'indices': outliers[:100],  # Limit to first 100 for display
                'percentage': len(outliers) / len(df) * 100
            }
        
        return outliers_info
    
    def remove_outliers(self, df, outlier_indices):
        """Remove specified outlier indices"""
        original_shape = df.shape
        df_clean = df.drop(index=outlier_indices)
        removed_count = original_shape[0] - df_clean.shape[0]
        
        self.cleaning_history.append(f"Removed {removed_count} outlier rows")
        
        return df_clean, removed_count
    
    def apply_transformations(self, df, transformations):
        """
        Apply mathematical transformations to columns
        
        transformations: dict with column names and transformation types
        Possible transformations: 'log', 'sqrt', 'square', 'reciprocal', 'normalize'
        """
        transformed_columns = {}
        
        for col, transform_type in transformations.items():
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            original_col = df[col].copy()
            
            try:
                if transform_type == 'log':
                    # Add small constant to handle zeros
                    df[col] = np.log1p(df[col].clip(lower=0))
                    transformed_columns[col] = "Applied log transformation"
                    
                elif transform_type == 'sqrt':
                    df[col] = np.sqrt(df[col].clip(lower=0))
                    transformed_columns[col] = "Applied square root transformation"
                    
                elif transform_type == 'square':
                    df[col] = df[col] ** 2
                    transformed_columns[col] = "Applied square transformation"
                    
                elif transform_type == 'reciprocal':
                    df[col] = 1 / df[col].replace(0, np.nan)
                    transformed_columns[col] = "Applied reciprocal transformation"
                    
                elif transform_type == 'normalize':
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    transformed_columns[col] = "Applied min-max normalization"
                
                self.cleaning_history.append(f"Transformed {col}: {transformed_columns[col]}")
                
            except Exception as e:
                df[col] = original_col  # Restore original if transformation fails
                transformed_columns[col] = f"Transformation failed: {str(e)}"
        
        return df, transformed_columns
    
    def get_cleaning_summary(self):
        """Get summary of all cleaning operations performed"""
        return {
            'operations_count': len(self.cleaning_history),
            'operations': self.cleaning_history,
            'column_mappings': self.column_mappings,
            'encoding_mappings': self.encoding_mappings
        }
    
    def save_cleaned_data(self, df, output_path):
        """Save the cleaned dataframe to a CSV file"""
        try:
            df.to_csv(output_path, index=False)
            self.cleaning_history.append(f"Saved cleaned data to {output_path}")
            return True
        except Exception as e:
            return False, str(e)
    
    def validate_data_quality(self, df):
        """Perform data quality validation"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        
        # Check for potential issues
        issues = []
        
        # High missing value percentage
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing_cols = missing_pct[missing_pct > 30].index.tolist()
        if high_missing_cols:
            issues.append(f"High missing values (>30%) in columns: {high_missing_cols}")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant/single-value columns: {constant_cols}")
        
        # Check for highly correlated numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                issues.append(f"Highly correlated columns (>0.95): {high_corr_pairs}")
        
        quality_report['potential_issues'] = issues
        quality_report['quality_score'] = self._calculate_quality_score(df)
        
        return quality_report
    
    def _calculate_quality_score(self, df):
        """Calculate an overall data quality score (0-100)"""
        score = 100
        
        # Deduct for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        score -= missing_pct * 0.5
        
        # Deduct for duplicates
        duplicate_pct = df.duplicated().sum() / len(df) * 100
        score -= duplicate_pct * 0.3
        
        # Deduct for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            score -= len(constant_cols) / len(df.columns) * 20
        
        return max(0, min(100, score))

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