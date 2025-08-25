import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os


def create_separate_plotly_violin_plots_final(file_path, sheet_name=None, save_directory=None):
    """
    Reads an Excel file and creates three SEPARATE interactive violin plots
    for Food, Energy, and Water, each with populations 5M, 7.5M, and 10M.
    Enhanced with Times New Roman fonts, rectangular layout, and better axis limits.
    """

    print(f"--- Starting Enhanced Plotly Plot Generation Script ---")
    print(f"Loading Excel file from: {file_path}")
    print(f"Target sheet name: {sheet_name}")

    try:
        if sheet_name:
            df_full = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        else:
            df_full = pd.read_excel(file_path, header=None)

        print("Full Excel sheet loaded successfully!")
        print(f"Full data shape: {df_full.shape}")
        print("\nFirst 10 rows of full data (for inspection):")
        print(df_full.head(10).to_string())

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'. Please check the file path.")
        return
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print("Ensure required packages (pandas, openpyxl) are installed.")
        return

    # --- PRECISE Fixed Column Mappings ---
    data_start_row_idx = 8  # Data starts from Excel row 9

    data_columns_map = {
        'Water': {
            '5M': 1,  # Excel Column B
            '7.5M': 2,  # Excel Column C
            '10M': 3  # Excel Column D
        },
        'Energy': {
            '5M': 6,  # Excel Column G
            '7.5M': 7,  # Excel Column H
            '10M': 8  # Excel Column I
        },
        'Food': {
            '5M': 11,  # Excel Column L
            '7.5M': 12,  # Excel Column M
            '10M': 13  # Excel Column N
        }
    }

    all_data_for_df = []

    print("\n--- Extracting data for all populations ---")
    for resource, pop_cols in data_columns_map.items():
        for pop_label, pd_col_idx in pop_cols.items():
            excel_col_letter = chr(65 + pd_col_idx)
            print(f"  Processing {resource} ({pop_label}) from column {excel_col_letter}...")

            if pd_col_idx < df_full.shape[1]:
                raw_series = df_full.iloc[data_start_row_idx:1009, pd_col_idx]
                col_values = pd.to_numeric(raw_series.astype(str).str.strip(), errors='coerce').dropna()

                if not col_values.empty:
                    for val in col_values:
                        all_data_for_df.append({
                            'Resource': resource,
                            'Population': pop_label,
                            'Value': val
                        })
                    print(f"    Successfully extracted {len(col_values)} numeric values.")
                else:
                    print(f"    WARNING: No valid numeric data found.")
            else:
                print(f"    ERROR: Column {excel_col_letter} does not exist.")

    if not all_data_for_df:
        print("\nCRITICAL ERROR: No data extracted for plotting.")
        return

    df_plot_data = pd.DataFrame(all_data_for_df)
    print(f"\n--- DataFrame Created ---")
    print(f"Total rows: {df_plot_data.shape[0]}")

    population_order = ['5M', '7.5M', '10M']
    df_plot_data['Population'] = pd.Categorical(df_plot_data['Population'], categories=population_order, ordered=True)

    # --- Print detailed data analysis for troubleshooting ---
    print("\n" + "=" * 60)
    print("DETAILED DATA ANALYSIS FOR VIOLIN PLOTS")
    print("=" * 60)
    if not df_plot_data.empty:
        for resource in ['Water', 'Energy', 'Food']:
            resource_data = df_plot_data[df_plot_data['Resource'] == resource]
            if not resource_data.empty:
                print(f"\n{resource.upper()} DATA:")
                for pop in population_order:
                    pop_data = resource_data[resource_data['Population'] == pop]['Value']
                    if len(pop_data) > 0:
                        print(f"  {pop}: {len(pop_data)} points, Range: {pop_data.min():.2f} - {pop_data.max():.2f}")
                        print(f"       Mean: {pop_data.mean():.2f}, Std: {pop_data.std():.2f}")
                    else:
                        print(f"  {pop}: NO DATA POINTS")
            else:
                print(f"\n{resource.upper()}: NO DATA AVAILABLE")
    print("=" * 60)

    # --- Create Enhanced Violin Plots ---
    if save_directory:
        os.makedirs(save_directory, exist_ok=True)

    resource_details = {
        'Food': {
            'ylabel': 'Food (Ton×10⁶)',
            'color': '#2ca02c',
            'title': 'Food Distribution by Population'
        },
        'Energy': {
            'ylabel': 'Energy (BOE)',
            'color': '#d62728',
            'title': 'Energy Distribution by Population'
        },
        'Water': {
            'ylabel': 'Water (MCM)',
            'color': '#1f77b4',
            'title': 'Water Distribution by Population'
        }
    }

    print("\n--- Generating Enhanced Plots ---")
    for resource_name, details in resource_details.items():
        current_resource_df = df_plot_data[df_plot_data['Resource'] == resource_name].copy()

        if not current_resource_df.empty:
            # Calculate appropriate axis limits to crop long lines
            min_val = current_resource_df['Value'].min()
            max_val = current_resource_df['Value'].max()
            range_val = max_val - min_val

            # Add minimal padding (5%) to just show violin shape
            y_min = min_val - 0.05 * range_val
            y_max = max_val + 0.05 * range_val

            # Create violin plot using graph_objects for better control
            fig = go.Figure()

            for pop in population_order:
                pop_data = current_resource_df[current_resource_df['Population'] == pop]['Value']

                if len(pop_data) > 0:
                    fig.add_trace(go.Violin(
                        x=[pop] * len(pop_data),
                        y=pop_data,
                        name=pop,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=details['color'],
                        opacity=0.7,
                        line_color='black',
                        points='outliers',
                        pointpos=0,
                        jitter=0.3,
                        scalemode='width',
                        width=0.8,
                        side='both',
                        spanmode='hard',  # محدود کردن به محدوده داده
                        bandwidth=0.5,  # کنترل نرمی violin
                        hoveron='points+violins',
                        hovertemplate='<b>Population:</b> %{x}<br>' +
                                      '<b>' + details['ylabel'] + ':</b> %{y}<br>' +
                                      '<extra></extra>',
                        # حذف کامل خط‌های extension
                        line=dict(width=0)  # خط‌های کناری رو حذف می‌کنه
                    ))

            fig.update_layout(
                title=details['title'],
                showlegend=False  # Hide legend since x-axis already shows populations
            )

            # Enhanced layout for Times New Roman and rectangular shape
            fig.update_layout(
                # Font settings - Times New Roman
                font=dict(
                    family="Times New Roman, serif",
                    size=14,
                    color="black"
                ),

                # Title formatting
                title=dict(
                    text=details['title'],
                    x=0.5,
                    font=dict(
                        family="Times New Roman, serif",
                        size=18,
                        color="black"
                    )
                ),

                # Rectangular dimensions (width > height for side-by-side arrangement)
                width=500,
                height=400,

                # Margins
                margin=dict(l=80, r=30, t=80, b=60),

                # Background
                plot_bgcolor='white',
                paper_bgcolor='white',

                # Grid
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    title=dict(
                        text="Population (Million)",
                        font=dict(
                            family="Times New Roman, serif",
                            size=16,
                            color="black"
                        )
                    ),
                    tickfont=dict(
                        family="Times New Roman, serif",
                        size=14,
                        color="black"
                    )
                ),

                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    range=[y_min, y_max],  # محدود کردن محدوده Y
                    fixedrange=False,  # اجازه zoom
                    title=dict(
                        text=details['ylabel'],
                        font=dict(
                            family="Times New Roman, serif",
                            size=16,
                            color="black"
                        )
                    ),
                    tickfont=dict(
                        family="Times New Roman, serif",
                        size=14,
                        color="black"
                    )
                )
            )

            # Additional formatting for violin plots
            fig.update_traces(
                box_fillcolor='white',
                box_line_color='black',
                box_line_width=1.5,
                meanline_color='red',
                meanline_width=2
            )

            # Save plot
            output_html_path = os.path.join(save_directory, f'{resource_name}_Violin_Plot_Enhanced.html')
            fig.write_html(output_html_path)

            # Also save as PNG for Word documents
            output_png_path = os.path.join(save_directory, f'{resource_name}_Violin_Plot_Enhanced.png')
            try:
                fig.write_image(output_png_path, width=500, height=400, scale=2)  # High resolution
                print(f"Plot for {resource_name} saved as HTML and PNG")
            except Exception as e:
                print(f"Plot for {resource_name} saved as HTML (PNG save failed: {e})")

            print(f"  HTML: {output_html_path}")
            fig.show()
        else:
            print(f"WARNING: No data available for {resource_name}.")

    print("\nScript finished successfully!")
    print("\nNote: PNG files are optimized for A4 Word documents.")
    print("Arrange the three plots side by side for best visual impact.")


# --- Usage Example ---
if __name__ == "__main__":
    file_path = r"C:\Users\LOQ\OneDrive\Desktop\For plots.xlsx"
    sheet_name = 'Fig.6'
    save_output_to_directory = r"C:\Users\LOQ\OneDrive\Desktop\Plots_Output"

    create_separate_plotly_violin_plots_final(
        file_path=file_path,
        sheet_name=sheet_name,
        save_directory=save_output_to_directory
    )