# -*- coding: utf-8 -*-
"""Predictive Maintenance Synthetic Data: GenAI Demo

Original file is located at
    https://colab.research.google.com/drive/10qqjCZ5PEwifDMS2UJko4j6VqxDBL8nv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Configuration & Constants ---
TARGET_COLUMN = 'Target' # 0 or 1 for failure
FAILURE_TYPE_COLUMN = 'Failure Type' # Specific failure modes
# Common features in the AI4I dataset
# Product ID, Type, UDI are not typically used as direct predictors for the failure event itself
# but are useful for identification.
IDENTIFIER_COLUMNS = ['UDI', 'Product ID', 'Type']
SENSOR_FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]
ALL_FEATURE_COLUMNS = SENSOR_FEATURE_COLUMNS # For this demo, we focus on sensor features for generation

# Define some known failure types from AI4I 2020 dataset
# Actual values in the 'Failure Type' column are:
# 'No Failure', 'Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Random Failures', 'Tool Wear Failure'
FAILURE_MODES = {
    "OSF": "Overstrain Failure",
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "RNF": "Random Failures" # Less predictable, harder to simulate specific conditions for
}
# For the demo, let's focus on a couple that have clear sensor relationships
DEMO_FAILURE_MODES = {
    "OSF": "Overstrain Failure", # Often related to high Torque and Tool Wear
    "TWF": "Tool Wear Failure",  # Directly related to Tool Wear
    "HDF": "Heat Dissipation Failure" # Related to Air/Process Temp difference and Speed
}

# --- Helper Functions ---

def load_data(uploaded_file):
    """Loads data from a CSV file."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = [TARGET_COLUMN, FAILURE_TYPE_COLUMN] + SENSOR_FEATURE_COLUMNS
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Dataset must contain the following columns: {', '.join(missing_cols)}.")
                return None

            # Ensure numeric types for sensor features
            for col in SENSOR_FEATURE_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=SENSOR_FEATURE_COLUMNS, inplace=True) # Drop rows where sensor data is not numeric

            # Ensure Target is 0 or 1
            if not df[TARGET_COLUMN].isin([0, 1]).all():
                st.warning(f"'{TARGET_COLUMN}' column contains values other than 0 or 1. Attempting to coerce...")
                df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x != 0 else 0)


            if df.empty:
                st.error("Data is empty after attempting to load and clean. Please check CSV.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def simulate_genai_sample(base_sample, failure_type_full_name, conditions, all_features):
    """
    Generates a single synthetic sample based on conditions,
    mimicking a GenAI that understands feature relationships for that failure type.
    """
    synthetic_sample = base_sample.copy()
    synthetic_sample[TARGET_COLUMN] = 1 # It's a failure
    synthetic_sample[FAILURE_TYPE_COLUMN] = failure_type_full_name

    # Apply user-defined conditions and  correlated changes
    # This is where the " GenAI" logic comes in.
    # It's rule-based for the demo, but a real GenAI would learn these.

    if failure_type_full_name == FAILURE_MODES["OSF"]: # Overstrain Failure
        # If Torque is a condition, set it and potentially adjust speed slightly inversely
        if 'Torque [Nm]' in conditions:
            synthetic_sample['Torque [Nm]'] = np.random.uniform(conditions['Torque [Nm]'][0], conditions['Torque [Nm]'][1])
            # Simulate: higher torque might slightly reduce speed if not compensated
            if 'Rotational speed [rpm]' not in conditions: # if speed isn't also a condition
                 synthetic_sample['Rotational speed [rpm]'] *= np.random.uniform(0.95, 0.99)
        if 'Tool wear [min]' in conditions:
            synthetic_sample['Tool wear [min]'] = np.random.uniform(conditions['Tool wear [min]'][0], conditions['Tool wear [min]'][1])
        # Other features get small random noise based on original sample
        for feat in all_features:
            if feat not in conditions and feat in synthetic_sample and pd.api.types.is_numeric_dtype(synthetic_sample[feat]):
                noise = np.random.normal(0, abs(synthetic_sample[feat] * 0.02) + 0.1) # Small noise
                synthetic_sample[feat] += noise

    elif failure_type_full_name == FAILURE_MODES["TWF"]: # Tool Wear Failure
        if 'Tool wear [min]' in conditions:
            synthetic_sample['Tool wear [min]'] = np.random.uniform(conditions['Tool wear [min]'][0], conditions['Tool wear [min]'][1])
        # For TWF, torque might also increase as tool wears, speed might be maintained or slightly drop
        if 'Torque [Nm]' not in conditions and synthetic_sample['Tool wear [min]'] > 180 : # If high wear
            synthetic_sample['Torque [Nm]'] *= np.random.uniform(1.01, 1.05)
        for feat in all_features:
            if feat not in conditions and feat in synthetic_sample and pd.api.types.is_numeric_dtype(synthetic_sample[feat]):
                noise = np.random.normal(0, abs(synthetic_sample[feat] * 0.02) + 0.1)
                synthetic_sample[feat] += noise

    elif failure_type_full_name == FAILURE_MODES["HDF"]: # Heat Dissipation Failure
        # HDF: if (AirTemp - ProcessTemp) is small AND speed is low
        if 'Air temperature [K]' in conditions:
            synthetic_sample['Air temperature [K]'] = np.random.uniform(conditions['Air temperature [K]'][0], conditions['Air temperature [K]'][1])
        if 'Process temperature [K]' in conditions:
            synthetic_sample['Process temperature [K]'] = np.random.uniform(conditions['Process temperature [K]'][0], conditions['Process temperature [K]'][1])

        # Ensure Process Temp is higher than Air Temp for HDF
        if synthetic_sample['Process temperature [K]'] <= synthetic_sample['Air temperature [K]']:
            synthetic_sample['Process temperature [K]'] = synthetic_sample['Air temperature [K]'] + np.random.uniform(1, 5) # Make it slightly higher

        if 'Rotational speed [rpm]' in conditions:
            synthetic_sample['Rotational speed [rpm]'] = np.random.uniform(conditions['Rotational speed [rpm]'][0], conditions['Rotational speed [rpm]'][1])
        else: # If not specified, HDF often occurs at lower speeds
            synthetic_sample['Rotational speed [rpm]'] = np.random.uniform(base_sample['Rotational speed [rpm]']*0.8, base_sample['Rotational speed [rpm]']*0.95)
            synthetic_sample['Rotational speed [rpm]'] = max(1000, synthetic_sample['Rotational speed [rpm]']) # Ensure it's not too low

        for feat in all_features:
            if feat not in conditions and feat in synthetic_sample and pd.api.types.is_numeric_dtype(synthetic_sample[feat]):
                noise = np.random.normal(0, abs(synthetic_sample[feat] * 0.02) + 0.1)
                synthetic_sample[feat] += noise
    else: # Default for other failure types or if no specific logic
        for feat in all_features:
            if feat in synthetic_sample and pd.api.types.is_numeric_dtype(synthetic_sample[feat]):
                noise_val = abs(synthetic_sample[feat] * 0.05) + 0.1 # Slightly larger noise if no specific rules
                synthetic_sample[feat] += np.random.normal(0, noise_val)


    # Ensure non-negative values for certain features
    for feat in ['Tool wear [min]', 'Torque [Nm]', 'Rotational speed [rpm]', 'Air temperature [K]', 'Process temperature [K]']:
        if feat in synthetic_sample and synthetic_sample[feat] < 0:
            synthetic_sample[feat] = base_sample[feat] * np.random.uniform(0.1, 0.5) # or a small positive value
            if synthetic_sample[feat] <0: synthetic_sample[feat] = 0


    # Copy identifiers if they exist in base_sample
    for id_col in IDENTIFIER_COLUMNS:
        if id_col in base_sample:
            synthetic_sample[id_col] = base_sample[id_col] # Keep original UDI/Product ID for reference to base
    if 'UDI' in synthetic_sample: # Modify UDI to indicate synthetic
        synthetic_sample['UDI'] = f"SYNTH_{base_sample.get('UDI', np.random.randint(10000,20000))}"


    return synthetic_sample


def generate_synthetic_data_pdm(df_original, failure_type_code, num_samples_to_generate, conditions):
    """
    Generates synthetic data by simulating GenAI output.
    """
    failure_type_full_name = FAILURE_MODES.get(failure_type_code, "Unknown Failure")
    st.write(f"Generating data for: '{failure_type_full_name}' with conditions: {conditions}")

    # Find real instances of this failure type to use as diverse starting points
    # Or, if none, use other failure instances or even normal instances near failure boundaries (more complex)
    base_samples_df = df_original[df_original[FAILURE_TYPE_COLUMN] == failure_type_full_name]
    if base_samples_df.empty:
        # Fallback: use any failure if specific type not found, or even normal data if no failures at all
        st.warning(f"No real instances of '{failure_type_full_name}' found. Using other failures as base for demo (less ideal).")
        base_samples_df = df_original[df_original[TARGET_COLUMN] == 1]
        if base_samples_df.empty:
            st.error("No failure data in the original set to base synthetic samples on. Cannot proceed with this demo.")
            return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames

    synthetic_list = []
    for i in range(num_samples_to_generate):
        # Cycle through base samples or pick randomly to get variety
        base_sample_row = base_samples_df.sample(1).iloc[0]
        syn_sample = simulate_genai_sample(base_sample_row, failure_type_full_name, conditions, SENSOR_FEATURE_COLUMNS)
        synthetic_list.append(syn_sample)

    df_synthetic = pd.DataFrame(synthetic_list)

    if not df_synthetic.empty:
        df_augmented = pd.concat([df_original, df_synthetic], ignore_index=True)
    else:
        df_augmented = df_original.copy()

    return df_augmented, df_synthetic

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Kearney Digital Brown Bag Lunch - Synthetic Data for Predictive Maintenance Demo")
st.markdown("""
This demo demonstrates how Generative AI can create well-labeled, trustable synthetic data for predictive maintenance.
You "instruct" the GenAI by specifying failure types and sensor conditions.
""")

# --- Sidebar for Upload and Controls ---
st.sidebar.header("1. Load Data")
uploaded_file = st.sidebar.file_uploader("Upload PdM CSV", type="csv")
use_sample_data = st.sidebar.checkbox("Use Minimal Sample Data", value=True if uploaded_file is None else False)

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_augmented' not in st.session_state:
    st.session_state.df_augmented = None
if 'df_synthetic_only' not in st.session_state:
    st.session_state.df_synthetic_only = None
if 'selected_failure_code' not in st.session_state:
    st.session_state.selected_failure_code = list(DEMO_FAILURE_MODES.keys())[0]
if 'conditions' not in st.session_state:
    st.session_state.conditions = {}


# --- Data Loading ---
if use_sample_data and uploaded_file is None:
    if st.session_state.df_original is None: # Load sample only once
        st.sidebar.info("Using a small,  AI4I dataset sample.")
        n_normal = 200
        n_fail_osf = 5
        n_fail_twf = 5
        n_fail_hdf = 5
        n_total = n_normal + n_fail_osf + n_fail_twf + n_fail_hdf

        data_s = {
            'UDI': range(1, n_total + 1),
            'Product ID': [f'L{i}' for i in range(1,151)] + [f'M{i}' for i in range(151, n_total + 1)],
            'Type': ['L'] * 150 + ['M'] * (n_total - 150),
            'Air temperature [K]': np.random.uniform(298, 303, n_total),
            'Process temperature [K]': np.random.uniform(308, 313, n_total),
            'Rotational speed [rpm]': np.random.uniform(1300, 1800, n_total),
            'Torque [Nm]': np.random.uniform(30, 50, n_total),
            'Tool wear [min]': np.random.uniform(0, 150, n_total),
            TARGET_COLUMN: [0]*n_normal + [1]*(n_fail_osf + n_fail_twf + n_fail_hdf),
            FAILURE_TYPE_COLUMN: ['No Failure']*n_normal + \
                                 [DEMO_FAILURE_MODES["OSF"]]*n_fail_osf + \
                                 [DEMO_FAILURE_MODES["TWF"]]*n_fail_twf + \
                                 [DEMO_FAILURE_MODES["HDF"]]*n_fail_hdf
        }
        # Adjust some values to be more characteristic for failures for the sample
        # OSF: Higher Torque, Higher Wear
        for i in range(n_normal, n_normal + n_fail_osf):
            data_s['Torque [Nm]'][i] = np.random.uniform(55, 75)
            data_s['Tool wear [min]'][i] = np.random.uniform(180, 230)
        # TWF: Higher Wear
        for i in range(n_normal + n_fail_osf, n_normal + n_fail_osf + n_fail_twf):
            data_s['Tool wear [min]'][i] = np.random.uniform(200, 250) # Max wear for failure
            data_s['Torque [Nm]'][i] = np.random.uniform(40, 60)
        # HDF: Temp diff small, low speed
        for i in range(n_normal + n_fail_osf + n_fail_twf, n_total):
            data_s['Air temperature [K]'][i] = np.random.uniform(300, 302)
            data_s['Process temperature [K]'][i] = data_s['Air temperature [K]'][i] + np.random.uniform(2, 5) # Small diff
            data_s['Rotational speed [rpm]'][i] = np.random.uniform(1200, 1400)


        sample_df = pd.DataFrame(data_s)
        sample_df = sample_df.sample(frac=1).reset_index(drop=True) # Shuffle
        st.session_state.df_original = sample_df.copy()
        st.session_state.df_augmented = None
        st.session_state.df_synthetic_only = None

elif uploaded_file is not None:
    current_file_name = uploaded_file.name
    if st.session_state.df_original is None or getattr(st.session_state, 'last_uploaded_file_name', '') != current_file_name:
        st.sidebar.success(f"File '{current_file_name}' uploaded!")
        st.session_state.df_original = load_data(uploaded_file)
        st.session_state.last_uploaded_file_name = current_file_name
        st.session_state.df_augmented = None
        st.session_state.df_synthetic_only = None
        if st.session_state.df_original is None: st.stop()

# --- Main App ---
if st.session_state.df_original is not None:
    df_orig_display = st.session_state.df_original

#    st.sidebar.header("2. Explore Real Data")
#    st.sidebar.dataframe(df_orig_display[[FAILURE_TYPE_COLUMN, TARGET_COLUMN]].value_counts().reset_index(name='count'), height=200)

    st.sidebar.header("2. Instruct  GenAI")
    st.session_state.selected_failure_code = st.sidebar.selectbox(
        "Select Failure Type to Synthesize:",
        options=list(DEMO_FAILURE_MODES.keys()),
        format_func=lambda x: f"{x} - {DEMO_FAILURE_MODES[x]}",
        key="sb_failure_select_pdm"
    )
    selected_failure_full_name = DEMO_FAILURE_MODES[st.session_state.selected_failure_code]

    st.sidebar.subheader("Optional: Specify Sensor Conditions")
    st.session_state.conditions = {} # Reset conditions for each selection

    # Conditional inputs based on failure type
    if st.session_state.selected_failure_code == "OSF": # Overstrain
        cond_torque_osf = st.sidebar.slider("Target Torque [Nm] for OSF:", float(df_orig_display['Torque [Nm]'].min()), float(df_orig_display['Torque [Nm]'].max()), (60.0, 80.0))
        st.session_state.conditions['Torque [Nm]'] = cond_torque_osf
        cond_wear_osf = st.sidebar.slider("Target Tool Wear [min] for OSF:", float(df_orig_display['Tool wear [min]'].min()), float(df_orig_display['Tool wear [min]'].max()), (180.0, 240.0))
        st.session_state.conditions['Tool wear [min]'] = cond_wear_osf

    elif st.session_state.selected_failure_code == "TWF": # Tool Wear
        cond_wear_twf = st.sidebar.slider("Target Tool Wear [min] for TWF (near max):", float(df_orig_display['Tool wear [min]'].min()), float(df_orig_display['Tool wear [min]'].max()), (200.0, 250.0))
        st.session_state.conditions['Tool wear [min]'] = cond_wear_twf

    elif st.session_state.selected_failure_code == "HDF": # Heat Dissipation
        # For HDF, it's about the *difference* and low speed.
        # We'll let the demo logic handle the temp diff, but user can set speed.
        cond_speed_hdf = st.sidebar.slider("Target Rotational Speed [rpm] for HDF (low):", float(df_orig_display['Rotational speed [rpm]'].min()), float(df_orig_display['Rotational speed [rpm]'].max()), (1000.0, 1400.0))
        st.session_state.conditions['Rotational speed [rpm]'] = cond_speed_hdf
        st.sidebar.caption(" HDF will also try to make Process Temp > Air Temp by a small margin.")


    num_synthetic_to_generate = st.sidebar.slider(
        "Number of Synthetic Samples to Generate:",
        min_value=10, max_value=500, value=50, step=10, key="slider_num_synth_pdm"
    )

    if st.sidebar.button(f"Generate Synthetic '{selected_failure_full_name}' Data"):
        with st.spinner(f"Simulating GenAI for '{selected_failure_full_name}'..."):
            df_aug, df_syn = generate_synthetic_data_pdm(
                st.session_state.df_original,
                st.session_state.selected_failure_code,
                num_synthetic_to_generate,
                st.session_state.conditions
            )
            st.session_state.df_augmented = df_aug
            st.session_state.df_synthetic_only = df_syn
            st.sidebar.success("Synthetic data generated!")

    # --- Display Area ---
    tab1, tab2, tab3 = st.tabs(["Original Data Insights", "Synthetic Data Details", "Comparison View"])

    with tab1:
        st.subheader("Original Data Overview")
        st.write("Shape:", df_orig_display.shape)
        st.write("Failure Type Distribution:")
        original_failure_dist = df_orig_display[FAILURE_TYPE_COLUMN].value_counts().reset_index(name='count')
        fig_orig_dist = px.bar(original_failure_dist, x=FAILURE_TYPE_COLUMN, y='count', title="Original Failure Type Distribution")
        st.plotly_chart(fig_orig_dist, use_container_width=True)
        st.dataframe(df_orig_display.head())

    with tab2:
        st.subheader(f"Generated Synthetic Data for '{selected_failure_full_name}'")
        if st.session_state.df_synthetic_only is not None and not st.session_state.df_synthetic_only.empty:
            st.write("Shape of Synthetic Data:", st.session_state.df_synthetic_only.shape)
            st.write("First 5 Synthetic Samples:")
            st.dataframe(st.session_state.df_synthetic_only.head())
            st.write("Summary Statistics of Synthetic Sensor Data:")
            st.dataframe(st.session_state.df_synthetic_only[SENSOR_FEATURE_COLUMNS].describe())
        else:
            st.info("No synthetic data generated yet. Use the sidebar controls.")

    with tab3:
        st.subheader("Original vs. Synthetic Data Comparison")
        if st.session_state.df_augmented is not None and st.session_state.df_synthetic_only is not None and not st.session_state.df_synthetic_only.empty:
            st.write("Distribution in Augmented Dataset:")
            augmented_failure_dist = st.session_state.df_augmented[FAILURE_TYPE_COLUMN].value_counts().reset_index(name='count')
            fig_aug_dist = px.bar(augmented_failure_dist, x=FAILURE_TYPE_COLUMN, y='count', title="Augmented Failure Type Distribution")
            st.plotly_chart(fig_aug_dist, use_container_width=True)

            # Prepare data for combined plotting
            df_plot_orig_failures = df_orig_display[df_orig_display[FAILURE_TYPE_COLUMN] == selected_failure_full_name].copy()
            if not df_plot_orig_failures.empty:
                 df_plot_orig_failures['Source'] = f'Original {selected_failure_full_name}'

            df_plot_synth = st.session_state.df_synthetic_only.copy()
            df_plot_synth['Source'] = f'Synthetic {selected_failure_full_name}'

            # Include some 'No Failure' data for context in scatter plots
            df_plot_orig_normal = df_orig_display[df_orig_display[FAILURE_TYPE_COLUMN] == 'No Failure'].sample(min(len(df_plot_synth), len(df_orig_display[df_orig_display[FAILURE_TYPE_COLUMN] == 'No Failure']), 200)).copy()
            if not df_plot_orig_normal.empty:
                df_plot_orig_normal['Source'] = 'Original No Failure'

            plot_df_list = []
            if not df_plot_orig_failures.empty: plot_df_list.append(df_plot_orig_failures)
            if not df_plot_synth.empty: plot_df_list.append(df_plot_synth)
            if not df_plot_orig_normal.empty: plot_df_list.append(df_plot_orig_normal)


            if plot_df_list:
                df_comparison_plot = pd.concat(plot_df_list, ignore_index=True)

                st.markdown(f"**Visualizing '{selected_failure_full_name}' (Original vs. Synthetic) against 'No Failure'**")

                # Feature selection for plotting
                available_plot_features = SENSOR_FEATURE_COLUMNS
                col1_plot, col2_plot = st.columns(2)
                with col1_plot:
                    x_axis_pdm = st.selectbox("X-axis:", available_plot_features, index=available_plot_features.index('Torque [Nm]') if 'Torque [Nm]' in available_plot_features else 0, key="x_pdm")
                with col2_plot:
                    y_axis_pdm = st.selectbox("Y-axis:", available_plot_features, index=available_plot_features.index('Tool wear [min]') if 'Tool wear [min]' in available_plot_features else 1, key="y_pdm")

                if x_axis_pdm and y_axis_pdm:
                    fig_scatter_pdm = px.scatter(
                        df_comparison_plot,
                        x=x_axis_pdm,
                        y=y_axis_pdm,
                        color='Source',
                        title=f"Comparison: {x_axis_pdm} vs {y_axis_pdm}",
                        opacity=0.7,
                        hover_data=IDENTIFIER_COLUMNS + [FAILURE_TYPE_COLUMN]
                    )
                    st.plotly_chart(fig_scatter_pdm, use_container_width=True)

                    # Histograms for the selected features
                    fig_hist_x = px.histogram(df_comparison_plot, x=x_axis_pdm, color='Source', marginal="rug", barmode='overlay', opacity=0.7, histnorm='probability density', title=f"Distribution of {x_axis_pdm}")
                    st.plotly_chart(fig_hist_x, use_container_width=True)

                    fig_hist_y = px.histogram(df_comparison_plot, x=y_axis_pdm, color='Source', marginal="rug", barmode='overlay', opacity=0.7, histnorm='probability density', title=f"Distribution of {y_axis_pdm}")
                    st.plotly_chart(fig_hist_y, use_container_width=True)

                st.subheader("Key Takeaways from demo:")
                st.markdown(f"""
                - **Convenience & Control:** You specified conditions for '{selected_failure_full_name}', and the system generated data accordingly.
                - **Well-Labeled Data:** All synthetic samples are explicitly and correctly labeled as '{selected_failure_full_name}'.
                - **Trustable & Dependable (Plausible):** - The synthetic data points (see plots) generally align with the characteristics of the original failures and the conditions you set.
                    - A *true* GenAI model would learn deeper statistical patterns and correlations to ensure even higher fidelity and novelty.
                - **Addressing Scarcity:** We now have more examples of this specific failure mode, which is crucial for training robust predictive models.
                """)
            else:
                st.info("Generate synthetic data to see comparisons.")
        else:
            st.info("Generate synthetic data to view this tab.")
else:
    st.info("Load data to begin.")

st.sidebar.markdown("---")
st.sidebar.markdown("Synthetic Data for Predictive Maintenance - Demo")