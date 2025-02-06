import streamlit as st
import pandas as pd
import numpy as np
from semopy import Model
import google.generativeai as genai

st.set_page_config(page_title="SEM Analysis App", layout="wide")

def initialize_gemini():
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = None
    if 'interpretation_requested' not in st.session_state:
        st.session_state.interpretation_requested = False
    if 'sem_results' not in st.session_state:
        st.session_state.sem_results = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'fit_df' not in st.session_state:
        st.session_state.fit_df = None
    if 'model_spec' not in st.session_state:
        st.session_state.model_spec = None
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        api_key = st.text_input(
            "Enter your Gemini API key:",
            type="password",
            value=st.session_state.gemini_api_key if st.session_state.gemini_api_key else "",
            help="Get your API key from https://makersuite.google.com/app/apikey",
            key="gemini_api_input"
        )
    
    with col2:
        if st.button("Generate Interpretation", key="generate_button"):
            st.session_state.interpretation_requested = True
    
    if api_key:
        if api_key != st.session_state.gemini_api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                _ = model.generate_content("Test")
                st.session_state.gemini_api_key = api_key
                return True
            except Exception as e:
                st.error(f"Error initializing Gemini API: {str(e)}")
                st.session_state.gemini_api_key = None
                return False
        return True
    return False

def generate_results_interpretation(results_df, fit_df, model_spec):
    try:
        prompt = f"""
        As a statistical expert, analyze these Structural Equation Modeling (SEM) results and provide a clear, concise interpretation:

        Model Specification:
        {model_spec}

        Parameter Estimates:
        {results_df.to_string()}

        Model Fit Measures:
        {fit_df.to_string()}

        Please provide:
        1. A summary of significant relationships found (p < 0.05)
        2. The strength and direction of these relationships
        3. An interpretation of the model fit
        4. Key insights and implications
        
        Keep the response concise and focused on the most important findings.
        """

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating interpretation: {str(e)}"

st.title("Structural Equation Modeling (SEM) Analysis")
st.markdown("""
Perform Structural Equation Modeling analysis on your data.
Upload your data and specify the relationships between variables to analyze their structural paths.
""")

def prepare_time_series_data(df, selected_metrics, date_column, frequency='W', include_lags=False, include_time_features=False):
    try:
        analysis_df = df[[date_column] + selected_metrics].copy()
        
        if analysis_df[date_column].str.contains('/').any():
            analysis_df[date_column] = analysis_df[date_column].str.split('/').str[0].str.strip()
        
        analysis_df[date_column] = pd.to_datetime(analysis_df[date_column])
        
        if frequency == 'M':
            analysis_df['period_date'] = analysis_df[date_column].dt.to_period('M').astype(str) + '-01'
        else:
            analysis_df['period_date'] = analysis_df[date_column]
        
        analysis_df['period_date'] = pd.to_datetime(analysis_df['period_date'])
        
        agg_df = (analysis_df.groupby('period_date')[selected_metrics]
                 .agg('sum')
                 .reset_index()
                 .set_index('period_date'))
        
        if include_lags:
            lag_periods = 3 if frequency == 'M' else 4
            for metric in selected_metrics:
                for lag in range(1, lag_periods + 1):
                    agg_df[f'{metric}_lag{lag}'] = agg_df[metric].shift(lag)
        
        if include_time_features:
            agg_df['month'] = agg_df.index.month
            agg_df['quarter'] = agg_df.index.quarter
            agg_df['trend'] = np.arange(len(agg_df))
        
        agg_df = agg_df.dropna()
        
        if len(agg_df) == 0:
            st.error("No data remaining after preprocessing. Check for missing values or invalid dates.")
            return None
            
        return agg_df
        
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None

def format_sem_results(sem_model):
    """
    Format SEM results into readable tables
    """
    try:
        # Get parameter estimates with labels
        params = sem_model.inspect(std_est=True)
        param_df = pd.DataFrame(params)
        
        # Create proper column names based on semopy output structure
        param_df.columns = ['Path', 'Operator', 'Variable', 'Estimate', 'Std.Est', 'Std.Err', 'z-value', 'P-value']
        
        # Round numeric columns
        numeric_cols = ['Estimate', 'Std.Est', 'Std.Err', 'z-value', 'P-value']
        param_df[numeric_cols] = param_df[numeric_cols].round(4)
        
        # Format P-values
        param_df['P-value'] = param_df['P-value'].apply(lambda x: f"{x:.4f}")
        
        return param_df
        
    except Exception as e:
        st.error(f"Error formatting results: {str(e)}")
        return pd.DataFrame(sem_model.inspect(std_est=True))

def get_model_fit_measures(sem_model):
    """
    Get and format model fit measures safely
    """
    try:
        # Get basic fit measures from model inspection
        basic_measures = sem_model.inspect()
        
        # Get additional fit measures
        fit_measures = sem_model.inspect(mode='fit')
        
        # Combine all measures
        all_measures = {}
        
        # Add basic measures if available
        if basic_measures is not None:
            for key in ['chi2', 'df', 'pvalue']:
                if key in basic_measures:
                    all_measures[key] = basic_measures[key]
        
        # Add fit measures if available
        if fit_measures is not None:
            all_measures.update(fit_measures)
        
        # Convert to DataFrame
        fit_data = []
        for metric, value in all_measures.items():
            if value is not None:  # Only include available measures
                fit_data.append({
                    'Metric': metric,
                    'Value': value
                })
        
        fit_df = pd.DataFrame(fit_data)
        
        if len(fit_df) == 0:
            st.info("Note: This appears to be a just-identified model, so traditional fit measures are not applicable.")
            return pd.DataFrame({
                'Metric': ['Model Status'],
                'Value': ['Just-identified model (df = 0)']
            })
        
        # Convert values to numeric where possible and round
        fit_df['Value'] = pd.to_numeric(fit_df['Value'], errors='ignore')
        numeric_mask = fit_df['Value'].apply(lambda x: isinstance(x, (int, float)))
        fit_df.loc[numeric_mask, 'Value'] = fit_df.loc[numeric_mask, 'Value'].round(4)
        
        # Sort measures in a logical order
        measure_order = ['chi2', 'df', 'pvalue', 'CFI', 'TLI', 'RMSEA', 'SRMR', 'AIC', 'BIC']
        fit_df['order'] = fit_df['Metric'].map({m: i for i, m in enumerate(measure_order)})
        fit_df = fit_df.sort_values('order').drop('order', axis=1)
        
        return fit_df
        
    except Exception as e:
        st.error(f"Error getting fit measures: {str(e)}")
        return pd.DataFrame({
            'Metric': ['Model Status'],
            'Value': ['Unable to calculate fit measures']
        })

def run_structural_equation_modeling(df, model_spec):
    """
    Run SEM analysis with custom model specification
    """
    try:
        # Fix model specification format (ensure * is present between coefficient and variable)
        fixed_model_spec = []
        for line in model_spec.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                fixed_model_spec.append(line)
                continue
            
            if '~' in line and '~~' not in line:
                # Handle regression paths
                parts = line.split('~')
                left = parts[0].strip()
                right = parts[1].strip()
                # Add * if missing between coefficient and variable
                if not '*' in right and right[0] == 'b':
                    coef = right.split()[0]
                    var = right.split()[1]
                    right = f"{coef}*{var}"
                fixed_model_spec.append(f"{left} ~ {right}")
            else:
                fixed_model_spec.append(line)
        
        model_spec = '\n'.join(fixed_model_spec)
        
        # Select metrics used in the model
        used_metrics = set()
        for line in model_spec.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '~~' in line:
                var_name = line.split('~~')[0].strip()
                if var_name:
                    used_metrics.add(var_name)
            elif '~' in line:
                parts = line.split('~')
                dependent = parts[0].strip()
                if dependent:
                    used_metrics.add(dependent)
                
                independent_part = parts[1].strip()
                for term in independent_part.split('+'):
                    if '*' in term:
                        var_name = term.split('*')[1].strip()
                    else:
                        var_name = term.strip()
                    if var_name:
                        used_metrics.add(var_name)
        
        used_metrics = list(used_metrics)
        if not used_metrics:
            st.error("No valid metrics found in model specification")
            return None
        
        # Verify metrics exist in dataframe
        missing_metrics = [m for m in used_metrics if m not in df.columns]
        if missing_metrics:
            st.error(f"The following metrics are not in your data: {', '.join(missing_metrics)}")
            return None
        
        # Select and standardize data
        sem_data = df[used_metrics].copy()
        for col in sem_data.columns:
            sem_data[col] = (sem_data[col] - sem_data[col].mean()) / sem_data[col].std()
        
        # Fit model
        sem_model = Model(model_spec)
        sem_model.fit(sem_data)
        
        return sem_model
        
    except Exception as e:
        st.error(f"Error in SEM analysis: {str(e)}")
        st.error(f"Data columns available: {list(df.columns)}")
        st.error(f"Metrics needed by model: {used_metrics}")
        return None

def generate_default_model(metrics):
    """
    Generate a default model specification based on selected metrics
    """
    if len(metrics) < 2:
        return "# Please select at least 2 metrics to generate a model"
    
    model_lines = []
    # Create relationships between consecutive metrics
    for i in range(len(metrics)-1):
        model_lines.append(f"{metrics[i+1]} ~ b{i+1}*{metrics[i]}")
    
    # Add variances
    model_lines.append("\n# Variances")
    for metric in metrics:
        model_lines.append(f"{metric} ~~ {metric}")
    
    return "\n".join(model_lines)

def main():
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Data preparation options
            st.subheader("Data Preparation")
            
            # Select date column
            date_columns = [col for col in df.columns if df[col].dtype in ['datetime64[ns]'] or 
                          ('date' in col.lower() or 'time' in col.lower())]
            
            if not date_columns:
                st.error("No date columns detected in your data. Please ensure you have a column containing dates.")
                return
                
            date_column = st.selectbox("Select the date column:", date_columns)
            
            # Select metrics for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != date_column]
            
            st.subheader("Select Metrics for Analysis")
            st.markdown("Choose the metrics you want to include in your SEM analysis:")
            
            selected_metrics = st.multiselect(
                "Select metrics:", 
                numeric_cols,
                help="Select at least 2 metrics to perform SEM analysis"
            )
            
            if len(selected_metrics) < 2:
                st.warning("Please select at least 2 metrics to proceed with the analysis.")
                return
            
            frequency = st.selectbox("Select time frequency", ['W', 'M'], index=0)
            
            # Add options for time features and lags
            col1, col2 = st.columns(2)
            with col1:
                include_lags = st.checkbox("Include lag features", value=False,
                                         help="Generate lagged versions of selected metrics")
            with col2:
                include_time_features = st.checkbox("Include time features", value=False,
                                                  help="Add month, quarter, and trend features")
            
            # Prepare data
            prepared_df = prepare_time_series_data(
                df, 
                selected_metrics, 
                date_column, 
                frequency=frequency,
                include_lags=include_lags,
                include_time_features=include_time_features
            )
            
            if prepared_df is not None:
                st.success("Data prepared successfully!")
                
                # Show prepared data preview
                st.subheader("Prepared Data Preview")
                st.dataframe(prepared_df.head())
                
                # Dynamic model building
                st.subheader("SEM Model Specification")
                st.markdown("### Define Your SEM Model")
                st.markdown("Specify relationships between variables using the format: `dependent ~ coefficient*independent`")
                
                # Generate default model based on selected metrics
                default_model = generate_default_model(selected_metrics)
                
                model_spec = st.text_area(
                    "Enter your model specification:", 
                    value=default_model, 
                    height=300,
                    help="You can modify the relationships between variables here"
                )
                
                if st.button("Run SEM Analysis"):
                    with st.spinner("Running SEM analysis..."):
                        # Debug information
                        st.write("Available columns in prepared data:", list(prepared_df.columns))
                        st.write("Model specification:", model_spec)
                        
                        # Extract metrics from model spec for verification
                        model_metrics = set()
                        for line in model_spec.split('\n'):
                            if '~' in line and not line.strip().startswith('#'):
                                parts = line.split('~')
                                model_metrics.add(parts[0].strip())
                                if '*' in parts[1]:
                                    model_metrics.add(parts[1].split('*')[1].strip())
                        
                        st.write("Metrics needed by model:", list(model_metrics))
                        
                        # Run SEM analysis
                        sem_results = run_structural_equation_modeling(prepared_df, model_spec)
                        
                        if sem_results:
                            # Store results in session state
                            st.session_state.sem_results = sem_results
                            st.session_state.model_spec = model_spec
                            
                            st.subheader("SEM Analysis Results")
                            
                            # Parameter estimates table
                            st.markdown("#### Parameter Estimates")
                            results_df = format_sem_results(sem_results)
                            st.session_state.results_df = results_df
                            
                            # Display results directly without styling
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Model fit measures
                            st.markdown("#### Model Fit Measures")
                            fit_df = get_model_fit_measures(sem_results)
                            st.session_state.fit_df = fit_df
                            
                            if not fit_df.empty:
                                # Add descriptions for common fit measures
                                fit_descriptions = {
                                    'CFI': 'Comparative Fit Index (>0.95 indicates good fit)',
                                    'TLI': 'Tucker-Lewis Index (>0.95 indicates good fit)',
                                    'RMSEA': 'Root Mean Square Error of Approximation (<0.06 indicates good fit)',
                                    'SRMR': 'Standardized Root Mean Square Residual (<0.08 indicates good fit)',
                                    'AIC': 'Akaike Information Criterion (lower is better)',
                                    'BIC': 'Bayesian Information Criterion (lower is better)',
                                    'chi2': 'Chi-square test statistic',
                                    'df': 'Degrees of freedom',
                                    'p-value': 'P-value for chi-square test (>0.05 indicates good fit)',
                                    'Model Status': 'Current model status and identification'
                                }
                                
                                fit_df['Description'] = fit_df['Metric'].map(
                                    lambda x: fit_descriptions.get(x, '')
                                )
                                
                                # Display fit measures
                                st.dataframe(fit_df, use_container_width=True)
                                
                                # Add interpretation of fit measures if they exist
                                if any(metric in fit_df['Metric'].values for metric in ['CFI', 'RMSEA', 'SRMR']):
                                    st.markdown("#### Model Fit Interpretation")
                                    if 'CFI' in fit_df['Metric'].values:
                                        cfi = fit_df.loc[fit_df['Metric'] == 'CFI', 'Value'].iloc[0]
                                        st.write(f"CFI = {cfi:.3f} ({'Good' if cfi > 0.95 else 'Poor'} fit)")
                                    if 'RMSEA' in fit_df['Metric'].values:
                                        rmsea = fit_df.loc[fit_df['Metric'] == 'RMSEA', 'Value'].iloc[0]
                                        st.write(f"RMSEA = {rmsea:.3f} ({'Good' if rmsea < 0.06 else 'Poor'} fit)")
                                    if 'SRMR' in fit_df['Metric'].values:
                                        srmr = fit_df.loc[fit_df['Metric'] == 'SRMR', 'Value'].iloc[0]
                                        st.write(f"SRMR = {srmr:.3f} ({'Good' if srmr < 0.08 else 'Poor'} fit)")
                            else:
                                st.warning("No fit measures available for this model")
                            
                            # AI Interpretation
                            st.markdown("#### AI-Generated Results Interpretation")
                            gemini_initialized = initialize_gemini()
                            
                            # Display stored results if they exist
                            if st.session_state.results_df is not None:
                                st.dataframe(st.session_state.results_df, use_container_width=True)
                            
                            if st.session_state.fit_df is not None:
                                st.dataframe(st.session_state.fit_df, use_container_width=True)
                            
                            if gemini_initialized and st.session_state.interpretation_requested:
                                with st.spinner("Generating results interpretation..."):
                                    interpretation = generate_results_interpretation(
                                        st.session_state.results_df,
                                        st.session_state.fit_df,
                                        st.session_state.model_spec
                                    )
                                    st.markdown(interpretation)
                                    # Reset the interpretation request flag
                                    st.session_state.interpretation_requested = False
                            elif not gemini_initialized:
                                st.warning("Please provide a valid Gemini API key to get AI-generated interpretation of results.")
                            
                            # Add download buttons for results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "Download Parameter Estimates",
                                    data=results_df.to_csv(index=True),
                                    file_name="sem_parameter_estimates.csv",
                                    mime="text/csv"
                                )
                            with col2:
                                st.download_button(
                                    "Download Fit Measures",
                                    data=fit_df.to_csv(index=False),
                                    file_name="sem_fit_measures.csv",
                                    mime="text/csv"
                                )
                            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

if __name__ == "__main__":
    main()