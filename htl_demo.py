import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Configuration & Constants ---
TARGET_COLUMN = 'Diabetes_012' # Target variable in the dataset
# Features to be used for synthetic data generation and visualization (subset for simplicity)
# These are common features from the BRFSS2015 diabetes dataset
# Ensure these are present in your CSV.
# 0 = no diabetes or only during pregnancy, 1 = prediabetes, 2 = diabetes.
FEATURE_COLUMNS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]
# Make sure these feature columns are actually numeric in the dataset.
# Some might be categorical even if they look numeric.
# For the demo, we'll treat them as numeric and add noise, or flip binary.

# Mapping for the target variable for better readability
DIABETES_MAP = {
    0.0: 'No Diabetes',
    1.0: 'Prediabetes',
    2.0: 'Diabetes'
}
# Inverse map for creating synthetic data
DIABETES_MAP_INV = {v: k for k, v in DIABETES_MAP.items()}

N_SYNTHETIC_PER_REAL_MINORITY = 5 # Number of synthetic samples per real minority instance
NOISE_LEVEL_NUMERIC = 0.05 # Noise level for numeric features (as a fraction of std deviation)
FLIP_PROB_BINARY = 0.02 # Probability of flipping a binary feature

# --- Helper Functions ---

def load_data(uploaded_file):
    """Loads data from a CSV file."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Basic check for required columns
            if TARGET_COLUMN not in df.columns:
                st.error(f"Dataset must contain the target column '{TARGET_COLUMN}'.")
                return None
            
            # Convert target to float, as it might be read as int
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)

            missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
            if missing_features:
                st.warning(f"Expected feature columns not found: {', '.join(missing_features)}. Visualizations and generation might be affected or fail.")
                # Attempt to use only available features from the list
                # This is a simple fix; a more robust solution would adapt more dynamically
                # For now, we'll proceed, and errors might occur if essential features for plotting are missing.

            # Attempt to convert feature columns to numeric, coercing errors
            for col in FEATURE_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN in essential columns after coercion (simplistic handling)
            # df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN], inplace=True) # This might remove too much data
            # A gentler approach:
            df.dropna(subset=[TARGET_COLUMN], inplace=True) # Ensure target is not NaN
            # For features, synthetic generation will handle NaNs in original sample if any remain.


            if df.empty:
                st.error("Data is empty after attempting to load and clean. Please check the CSV format and content.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def generate_synthetic_sample_healthcare(original_sample, features_to_vary, numeric_cols, binary_cols, noise_level_numeric, flip_prob_binary):
    """
    Generates a single synthetic healthcare sample.
    - Adds Gaussian noise to numeric features.
    - Randomly flips binary features with a small probability.
    """
    synthetic_sample = original_sample.copy()

    for feature in features_to_vary:
        if pd.isna(synthetic_sample[feature]): # If original value is NaN, keep it NaN or fill with a mode/mean (not done here for simplicity)
            continue

        if feature in numeric_cols:
            mean_val = synthetic_sample[feature]
            # Use a fraction of the feature's value as std_dev for proportional noise,
            # or a small absolute value if feature value is 0.
            std_dev = abs(mean_val * noise_level_numeric) if mean_val != 0 else noise_level_numeric
            noise = np.random.normal(0, std_dev)
            synthetic_sample[feature] = synthetic_sample[feature] + noise
            # Basic constraints (e.g., Age, BMI shouldn't be negative)
            if feature in ['BMI', 'Age', 'MentHlth', 'PhysHlth'] and synthetic_sample[feature] < 0:
                synthetic_sample[feature] = 0
            if feature == 'Age': # Max age constraint
                 synthetic_sample[feature] = min(synthetic_sample[feature], 100) # Assuming age is in years and BRFSS has coded values
            if feature == 'BMI':
                 synthetic_sample[feature] = min(synthetic_sample[feature], 100) # Cap BMI


        elif feature in binary_cols: # Assuming binary are 0 or 1
            if np.random.rand() < flip_prob_binary:
                synthetic_sample[feature] = 1.0 - synthetic_sample[feature] # Flip 0 to 1 or 1 to 0

        # For other categorical features (not explicitly handled here for simplicity),
        # one might sample from the existing distribution or use more advanced techniques.
        # The BRFSS dataset often uses coded numeric values for categories (e.g., GenHlth 1-5).
        # For these, adding small noise and then rounding/clipping to valid range might be an approach.
        if feature in ['GenHlth', 'Education', 'Income']: # Ordinal-like features
            if feature in numeric_cols: # if treated as numeric
                 synthetic_sample[feature] = np.round(synthetic_sample[feature])
                 # Add clipping based on known min/max for these survey codes if available
                 # Example: GenHlth is 1-5
                 if feature == 'GenHlth':
                     synthetic_sample[feature] = np.clip(synthetic_sample[feature], 1, 5)
                 # Similar clipping for Education, Income based on their coding in BRFSS


    return synthetic_sample


def generate_synthetic_data_healthcare(df_original, class_to_augment_code, num_synthetic_per_instance):
    """
    Generates synthetic data for a specific class in the diabetes dataset.
    """
    class_name = DIABETES_MAP.get(class_to_augment_code, f"Class {class_to_augment_code}")
    st.write(f"Attempting to augment: '{class_name}' (Code: {class_to_augment_code})")
    
    real_minority_df = df_original[df_original[TARGET_COLUMN] == class_to_augment_code]
    
    synthetic_list = []
    if not real_minority_df.empty:
        st.write(f"Found {len(real_minority_df)} real instances of '{class_name}'. Generating {num_synthetic_per_instance} synthetic samples per real instance.")
        
        # Identify binary and numeric columns from FEATURE_COLUMNS that are present in the dataframe
        # Binary columns are assumed to have only 2 unique values (0 and 1 after potential coercion)
        # All others in FEATURE_COLUMNS are treated as numeric for this simplified demo
        available_features = [f for f in FEATURE_COLUMNS if f in df_original.columns]
        binary_cols = [col for col in available_features if df_original[col].nunique(dropna=True) <= 2 and df_original[col].min() == 0 and df_original[col].max() == 1]
        numeric_cols = [col for col in available_features if col not in binary_cols]

        st.caption(f"Identified binary features for flipping: {', '.join(binary_cols) if binary_cols else 'None'}")
        st.caption(f"Identified numeric/ordinal features for noise addition: {', '.join(numeric_cols) if numeric_cols else 'None'}")

        for _, real_sample_row in real_minority_df.iterrows():
             for _ in range(num_synthetic_per_instance):
                syn_sample = generate_synthetic_sample_healthcare(
                    real_sample_row, 
                    available_features, # Pass only available features
                    numeric_cols, 
                    binary_cols, 
                    NOISE_LEVEL_NUMERIC, 
                    FLIP_PROB_BINARY
                )
                # Ensure the target column is correctly set for the synthetic sample
                syn_sample[TARGET_COLUMN] = class_to_augment_code 
                synthetic_list.append(syn_sample)
        st.write(f"Generated a total of {len(synthetic_list)} synthetic '{class_name}' samples.")
    else:
        st.warning(f"No real instances of class '{class_name}' found in the provided data to base synthetic samples on.")

    df_synthetic = pd.DataFrame(synthetic_list)
    
    if not df_synthetic.empty:
        df_augmented = pd.concat([df_original, df_synthetic], ignore_index=True)
    else:
        df_augmented = df_original.copy() # No synthetic data generated
        
    return df_augmented, df_synthetic


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🩺 Healthcare Demo: Synthetic Data for Diabetes Risk Factors")
st.markdown("""
This demo uses the BRFSS2015 Diabetes Health Indicators dataset to show how synthetic data can help address class imbalance.
This is crucial for building fairer and more accurate health risk models, especially for underrepresented patient groups, while respecting privacy.
**Disclaimer:** This is a simplified demonstration for educational purposes. Real-world synthetic data generation for healthcare requires rigorous methods and validation.
""")

# --- Sidebar for Upload and Controls ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Diabetes Indicators CSV (BRFSS2015 format)", type="csv")
use_sample_data = st.sidebar.checkbox("Use Sample Data (Simulated)", value=True if uploaded_file is None else False)

# Initialize session state variables
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_augmented' not in st.session_state:
    st.session_state.df_augmented = None
if 'df_synthetic_only' not in st.session_state:
    st.session_state.df_synthetic_only = None
if 'selected_class_to_augment' not in st.session_state:
    st.session_state.selected_class_to_augment = DIABETES_MAP_INV['Diabetes'] # Default to augmenting 'Diabetes'


# --- Data Loading Logic ---
if use_sample_data and uploaded_file is None:
    if st.session_state.df_original is None: # Load sample data only once
        st.sidebar.info("Using a small, simulated sample dataset for speed. For full features, upload the BRFSS2015 CSV.")
        # Create a small, very simplified sample DataFrame
        n_rows = 1000
        data_s = {
            TARGET_COLUMN: np.random.choice([0.0, 1.0, 2.0], n_rows, p=[0.85, 0.10, 0.05]), # Imbalanced
            'HighBP': np.random.randint(0, 2, n_rows),
            'HighChol': np.random.randint(0, 2, n_rows),
            'CholCheck': np.random.randint(0, 2, n_rows),
            'BMI': np.random.normal(28, 5, n_rows).clip(15, 50),
            'Smoker': np.random.randint(0, 2, n_rows),
            'Stroke': np.random.randint(0, 2, n_rows),
            'HeartDiseaseorAttack': np.random.randint(0, 2, n_rows),
            'PhysActivity': np.random.randint(0, 2, n_rows),
            'Fruits': np.random.randint(0, 2, n_rows),
            'Veggies': np.random.randint(0, 2, n_rows),
            'HvyAlcoholConsump': np.random.randint(0, 2, n_rows),
            'AnyHealthcare': np.random.randint(0, 2, n_rows),
            'NoDocbcCost': np.random.randint(0, 2, n_rows),
            'GenHlth': np.random.randint(1, 6, n_rows).astype(float),
            'MentHlth': np.random.randint(0, 31, n_rows).astype(float),
            'PhysHlth': np.random.randint(0, 31, n_rows).astype(float),
            'DiffWalk': np.random.randint(0, 2, n_rows),
            'Sex': np.random.randint(0, 2, n_rows), # Assuming 0 for female, 1 for male
            'Age': np.random.randint(1, 14, n_rows).astype(float), # BRFSS Age is 13 categories
            'Education': np.random.randint(1, 7, n_rows).astype(float), # BRFSS Education is 6 categories
            'Income': np.random.randint(1, 9, n_rows).astype(float) # BRFSS Income is 8 categories
        }
        sample_df = pd.DataFrame(data_s)
        st.session_state.df_original = sample_df.copy()
        st.session_state.df_augmented = None # Reset augmented data
        st.session_state.df_synthetic_only = None

elif uploaded_file is not None:
    current_file_name = uploaded_file.name
    if st.session_state.df_original is None or getattr(st.session_state, 'last_uploaded_file_name', '') != current_file_name:
        st.sidebar.success(f"File '{current_file_name}' uploaded successfully!")
        st.session_state.df_original = load_data(uploaded_file)
        st.session_state.last_uploaded_file_name = current_file_name
        st.session_state.df_augmented = None # Reset augmented data
        st.session_state.df_synthetic_only = None
        if st.session_state.df_original is None:
            st.stop()


# --- Main App Body ---
if st.session_state.df_original is not None:
    df_display_original = st.session_state.df_original.copy()
    # Map target for display
    df_display_original[TARGET_COLUMN] = df_display_original[TARGET_COLUMN].map(DIABETES_MAP)


    st.header("1. Original Data Overview")
    st.write("Shape of the dataset:", df_display_original.shape)
    st.write("First 5 rows (Target mapped to names):")
    st.dataframe(df_display_original.head())

    st.header("2. The Challenge: Class Imbalance in Health Data")
    if TARGET_COLUMN in df_display_original.columns:
        class_counts = df_display_original[TARGET_COLUMN].value_counts()
        st.write("Distribution of Diabetes Status (Original Data):")
        st.dataframe(class_counts)
        fig_class_counts = px.bar(class_counts, 
                                   x=class_counts.index, y=class_counts.values,
                                   labels={'x': 'Diabetes Status', 'y': 'Count'},
                                   title="Original Distribution of Diabetes Status")
        fig_class_counts.update_xaxes(type='category')
        st.plotly_chart(fig_class_counts, use_container_width=True)
        
        # Allow user to select which class to augment
        # Make sure to use the original numeric codes for selection logic
        # but display human-readable names.
        # Get available classes from the original numeric data
        original_numeric_classes = sorted(st.session_state.df_original[TARGET_COLUMN].unique())
        
        # Filter out NaN if any, and ensure they are keys in DIABETES_MAP
        valid_classes_for_selection = {k: DIABETES_MAP[k] for k in original_numeric_classes if pd.notna(k) and k in DIABETES_MAP}

        if valid_classes_for_selection:
            # Create options for selectbox: list of tuples (label, code)
            options_for_selectbox = [(name, code) for code, name in valid_classes_for_selection.items() if name != 'No Diabetes'] # Don't augment 'No Diabetes'
            
            if options_for_selectbox:
                # Find the index of the default selection
                default_class_code = st.session_state.selected_class_to_augment
                default_index = 0
                for i, (name, code) in enumerate(options_for_selectbox):
                    if code == default_class_code:
                        default_index = i
                        break
                
                selected_option = st.sidebar.selectbox(
                    "Select Minority Class to Augment:",
                    options=options_for_selectbox,
                    format_func=lambda x: x[0], # Display name
                    index=default_index,
                    key="sb_class_augment"
                )
                st.session_state.selected_class_to_augment = selected_option[1] # Store the code

                num_synthetic_to_gen_per_instance = st.sidebar.slider(
                    f"Number of synthetic '{selected_option[0]}' samples to generate (per real instance):",
                    min_value=1, max_value=20, value=N_SYNTHETIC_PER_REAL_MINORITY, key="slider_synthetic_count"
                )
            else:
                st.sidebar.warning("No minority classes (Prediabetes/Diabetes) found or suitable for augmentation.")
                st.session_state.selected_class_to_augment = None
        else:
            st.sidebar.warning("Target classes could not be identified for augmentation.")
            st.session_state.selected_class_to_augment = None


    st.header("3. Generating Privacy-Preserving Synthetic Health Data")
    if st.session_state.selected_class_to_augment is not None:
        selected_class_name = DIABETES_MAP.get(st.session_state.selected_class_to_augment, "Selected Class")
        if st.button(f"Generate Synthetic Data for '{selected_class_name}'"):
            with st.spinner(f"Generating synthetic data for '{selected_class_name}'... This might take a moment for larger datasets or more synthetic samples."):
                df_aug, df_syn_only = generate_synthetic_data_healthcare(
                    st.session_state.df_original, # Pass original numeric data
                    st.session_state.selected_class_to_augment,
                    num_synthetic_to_gen_per_instance
                )
                st.session_state.df_augmented = df_aug
                st.session_state.df_synthetic_only = df_syn_only
                st.success(f"Synthetic data generated for '{selected_class_name}'!")
    elif TARGET_COLUMN in df_display_original.columns:
         st.info("Select a minority class (Prediabetes or Diabetes) from the sidebar to enable synthetic data generation.")


    if st.session_state.df_augmented is not None:
        st.header("4. Enhanced & More Balanced Dataset")
        df_display_augmented = st.session_state.df_augmented.copy()
        df_display_augmented[TARGET_COLUMN] = df_display_augmented[TARGET_COLUMN].map(DIABETES_MAP)

        st.write("Shape of the augmented dataset:", df_display_augmented.shape)
        
        augmented_class_counts = df_display_augmented[TARGET_COLUMN].value_counts()
        st.write("Distribution of Diabetes Status (Augmented Data):")
        st.dataframe(augmented_class_counts)
        fig_aug_class_counts = px.bar(augmented_class_counts,
                                        x=augmented_class_counts.index, y=augmented_class_counts.values,
                                        labels={'x': 'Diabetes Status', 'y': 'Count'},
                                        title="Augmented Distribution of Diabetes Status")
        fig_aug_class_counts.update_xaxes(type='category')
        st.plotly_chart(fig_aug_class_counts, use_container_width=True)


        st.subheader("Visualizing Original vs. Synthetic Data")
        current_augmented_class_name = DIABETES_MAP.get(st.session_state.selected_class_to_augment, "Augmented Class")
        st.markdown(f"""
        Compare original and synthetic data points for key health indicators.
        Synthetic data for **'{current_augmented_class_name}'** is shown in a distinct color.
        """)

        # Define features for plotting that are likely to be informative and numeric
        # These should be columns that were processed as numeric in generate_synthetic_sample_healthcare
        plot_features_options = [f for f in FEATURE_COLUMNS if f in st.session_state.df_original.columns and pd.api.types.is_numeric_dtype(st.session_state.df_original[f])]
        
        if len(plot_features_options) >=2:
            x_axis_feat = st.selectbox("Select X-axis feature for plot:", plot_features_options, index=plot_features_options.index('BMI') if 'BMI' in plot_features_options else 0, key="x_axis_select_hc")
            y_axis_feat = st.selectbox("Select Y-axis feature for plot:", plot_features_options, index=plot_features_options.index('Age') if 'Age' in plot_features_options else 1, key="y_axis_select_hc")

            # Prepare data for plotting
            df_plot_orig = st.session_state.df_original.copy()
            df_plot_orig['Source'] = 'Original'
            
            plot_data_frames = [df_plot_orig]
            
            if st.session_state.df_synthetic_only is not None and not st.session_state.df_synthetic_only.empty:
                df_plot_syn = st.session_state.df_synthetic_only.copy()
                df_plot_syn['Source'] = f'Synthetic {current_augmented_class_name}'
                plot_data_frames.append(df_plot_syn)
            
            df_final_plot = pd.concat(plot_data_frames, ignore_index=True)
            df_final_plot[TARGET_COLUMN] = df_final_plot[TARGET_COLUMN].map(DIABETES_MAP) # Map target for color
            
            # Cap number of points for scatter plot for performance if dataset is huge
            max_points_scatter = 5000
            if len(df_final_plot) > max_points_scatter:
                st.caption(f"Displaying a sample of {max_points_scatter} points in scatter plot for performance.")
                df_final_plot_scatter = df_final_plot.sample(max_points_scatter)
            else:
                df_final_plot_scatter = df_final_plot
            
            fig_scatter = px.scatter(df_final_plot_scatter, x=x_axis_feat, y=y_axis_feat,
                                     color=TARGET_COLUMN, symbol='Source',
                                     title=f'Original vs. Synthetic Data ({x_axis_feat} vs. {y_axis_feat})',
                                     hover_data=[col for col in ['BMI', 'Age', 'HighBP', 'HighChol'] if col in df_final_plot_scatter.columns],
                                     opacity=0.6,
                                     color_discrete_map={ # Ensure consistent coloring
                                         'No Diabetes': 'blue',
                                         'Prediabetes': 'orange',
                                         'Diabetes': 'red',
                                     })
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Histograms for a single feature
            hist_feat_default_index = plot_features_options.index(x_axis_feat) if x_axis_feat in plot_features_options else 0
            hist_feat = st.selectbox("Select feature for Histogram comparison:", plot_features_options, index=hist_feat_default_index, key="hist_feat_select_hc")
            
            # For histogram, compare original minority vs synthetic minority vs original majority
            hist_plot_frames = []
            # Original selected minority class
            orig_minority_hist = st.session_state.df_original[st.session_state.df_original[TARGET_COLUMN] == st.session_state.selected_class_to_augment].copy()
            if not orig_minority_hist.empty:
                orig_minority_hist['Plot Category'] = f'Original {current_augmented_class_name}'
                hist_plot_frames.append(orig_minority_hist)

            # Synthetic selected minority class
            if st.session_state.df_synthetic_only is not None and not st.session_state.df_synthetic_only.empty:
                syn_minority_hist = st.session_state.df_synthetic_only.copy()
                syn_minority_hist['Plot Category'] = f'Synthetic {current_augmented_class_name}'
                hist_plot_frames.append(syn_minority_hist)
            
            # Original majority class (No Diabetes) for reference
            orig_majority_hist = st.session_state.df_original[st.session_state.df_original[TARGET_COLUMN] == DIABETES_MAP_INV['No Diabetes']].copy()
            if not orig_majority_hist.empty:
                # Sample if too large
                orig_majority_hist = orig_majority_hist.sample(min(len(orig_majority_hist), len(orig_minority_hist) + len(st.session_state.df_synthetic_only if st.session_state.df_synthetic_only is not None else []) , 2000))
                orig_majority_hist['Plot Category'] = 'Original No Diabetes'
                hist_plot_frames.append(orig_majority_hist)

            if hist_plot_frames:
                df_final_hist_plot = pd.concat(hist_plot_frames, ignore_index=True)
                fig_hist = px.histogram(df_final_hist_plot, x=hist_feat, color='Plot Category',
                                        marginal="rug", barmode='overlay', opacity=0.7, histnorm='probability density',
                                        title=f"Distribution of {hist_feat} by Category")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("Not enough data categories to plot histogram comparison.")

        else:
            st.warning("Not enough suitable numeric features available in the dataset for plotting.")

        st.subheader("Key Benefits Demonstrated:")
        st.markdown(f"""
        -   **Addressing Data Imbalance:** We've increased the representation of the '{current_augmented_class_name}' group, crucial for training less biased and more sensitive predictive models.
        -   **Privacy Preservation (Concept):** While this demo uses a public dataset, the *technique* allows for generating data that mimics real patient characteristics *without using actual patient identities*. This is vital for sharing data or training models when real data is too sensitive.
        -   **Plausible Data for Model Training:** The visualizations show that synthetic individuals (for '{current_augmented_class_name}') exhibit characteristics that are generally consistent with the real individuals in that group.
        -   **Potential for Better Health Outcomes:** Models trained on more representative data can lead to earlier and more accurate identification of at-risk individuals, enabling timely interventions.
        """)
    elif st.session_state.df_augmented is None and st.session_state.selected_class_to_augment is not None:
         selected_class_name_msg = DIABETES_MAP.get(st.session_state.selected_class_to_augment, "Selected Class")
         st.info(f"Click the 'Generate Synthetic Data for {selected_class_name_msg}' button to see augmented results and visualizations.")

else:
    st.info("Please upload a CSV file or select 'Use Sample Data' to begin.")

st.sidebar.markdown("---")
st.sidebar.markdown("Healthcare Synthetic Data Demo.")
