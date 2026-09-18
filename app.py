import os
import json
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

# --- 1. Define Paths (Adjusted for Hugging Face Space structure) ---
# Identificador exacto de tu Dataset en Hugging Face
DATASET_REPO = "LichiRMN/BananaSpectral-DB"
logo_image_path = 'banana spectral.jpeg'

# Lista de archivos a descargar
ARCHIVOS_JSON = [
    "Complete_splib07a.json",
    "Complete_splib07b.json",
    "Complete_splib07b_cvASD.json",
    "database_200to1000_pigmentcheker.json",
    "database_NIR_pigmentcheker1.json",
    "database_pigmentcheker1.json"
]

# --- 2. Data Loading and Processing Functions ---
def load_json_databases(base_path):
    """
    Loads all JSON files from a specified Hugging Face Dataset repo into a dictionary.
    Args:
        base_path (str): HF Dataset repo ID (e.g., 'LANCIC-IF/bd-laboratorio').
    Returns:
        dict: A dictionary where keys are filenames and values are the loaded JSON data.
    """
    loaded_databases = {}
    token = os.getenv("HF_TOKEN")

    try:
        api = HfApi()
        all_files = api.list_repo_files(repo_id=base_path, repo_type="dataset", token=token)
        json_files = [f for f in all_files if f.endswith('.json')]
    except Exception as e:
        print(f"HF Dataset not found '{base_path}': {e}")
        return loaded_databases

    for filename in json_files:
        try:
            filepath = hf_hub_download(
                repo_id=base_path,
                filename=filename,
                repo_type="dataset",
                token=token
            )
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                loaded_databases[filename] = data
            print(f"  - Loaded '{filename}'")
        except Exception as e:
            print(f"  - Error loading '{filename}': {e}")

    return loaded_databases

def detect_hierarchy_type(db):
    """
    Detects whether a loaded JSON database follows 'Tipo 1' or 'Tipo 2' hierarchy.
    Args:
        db (dict): A loaded JSON database.
    Returns:
        str: 'Tipo 2', 'Tipo 1', or 'Unknown'.
    """
    if not isinstance(db, dict) or not db:
        return 'Unknown'

    # Check for Hierarchy Type 2
    first_level_value = next(iter(db.values()), None)
    if isinstance(first_level_value, dict) and first_level_value:
        second_level_value = next(iter(first_level_value.values()), None)
        if isinstance(second_level_value, dict):
            if all(k in second_level_value for k in ['minor_wavelength', 'major_wavelength', 'reflectance']):
                return 'Tipo 2'

    # Check for Hierarchy Type 1
    if isinstance(first_level_value, dict):
        if all(k in first_level_value for k in ['minor_wavelength', 'major_wavelength', 'reflectance']):
            return 'Tipo 1'

    return 'Unknown'

def extract_data_type1(db):
    """
    Extracts spectral data and other relevant information from a 'Tipo 1' database.
    Returns:
        list: A list of dictionaries, where each dictionary represents a material
              with its extracted properties.
    """
    extracted_data = []
    for material_name, properties in db.items():
        material_info = {'name': material_name}
        for key, value in properties.items():
            material_info[key] = value
        extracted_data.append(material_info)
    return extracted_data

def extract_data_type2(db):
    """
    Extracts spectral data and other relevant information from a 'Tipo 2' database.
    Returns:
        list: A list of dictionaries, where each dictionary represents a material
              with its extracted properties.
    """
    extracted_data = []
    for category, category_data in db.items():
        if isinstance(category_data, dict):
            for material_name, properties in category_data.items():
                material_info = {'name': material_name, 'category': category}
                for key, value in properties.items():
                    material_info[key] = value
                extracted_data.append(material_info)
    return extracted_data

# --- 3. Initial Data Loading and Processing for the App ---
print("Loading databases...")
loaded_databases = load_json_databases(DATASET_REPO)

all_extracted_spectral_data = {}
print("\nExtracting data from loaded databases:")
for db_name, db_content in loaded_databases.items():
    hierarchy_type = detect_hierarchy_type(db_content)
    extracted_list = []
    if hierarchy_type == 'Tipo 1':
        extracted_list = extract_data_type1(db_content)
    elif hierarchy_type == 'Tipo 2':
        extracted_list = extract_data_type2(db_content)
    else:
        print(f"Skipped '{db_name}': Unknown hierarchy type during data extraction.")

    if extracted_list:
        all_extracted_spectral_data[db_name] = extracted_list

combined_spectral_data = []
for db_name, data_list in all_extracted_spectral_data.items():
    for material in data_list:
        material_with_source = material.copy()
        material_with_source['source_database'] = db_name
        combined_spectral_data.append(material_with_source)
print(f"Combined data from all databases into a single list. Total materials: {len(combined_spectral_data)}")

# Load logo image
try:
    logo_image = Image.open(logo_image_path)
    print(f"Logo image loaded successfully from {logo_image_path}.")
except FileNotFoundError:
    print(f"Error: The file was not found at {logo_image_path}. Displaying placeholder.")
    logo_image = None # Set to None or a placeholder if loading fails
except Exception as e:
    print(f"An error occurred while loading the logo image: {e}. Displaying placeholder.")
    logo_image = None

# --- 4. Core Application Functions ---
def search_database(combined_data, query):
    """
    Searches for materials in the combined spectral data based on a query string.
    """
    results = []
    lower_query = query.lower()
    for material in combined_data:
        match = False
        if 'name' in material and lower_query in str(material['name']).lower():
            match = True
        if not match and 'category' in material and lower_query in str(material['category']).lower():
            match = True
        for key, value in material.items():
            if key not in ['name', 'category', 'minor_wavelength', 'major_wavelength', 'reflectance', 'source_database'] and isinstance(value, str):
                if lower_query in value.lower():
                    match = True
                    break
        if match:
            results.append(material)
    return results

def plot_spectral_data_from_query_results(materials_to_plot):
    """
    Generates a list of Plotly figures, where each figure displays the spectral data
    (reflectance vs. wavelength) for a single material.
    """
    figures = []
    for material in materials_to_plot:
        material_name = material.get('name', 'Unknown Material')
        minor_wavelength_um = material.get('minor_wavelength')
        major_wavelength_um = material.get('major_wavelength')
        reflectance = material.get('reflectance')
        source_database = material.get('source_database', 'N/A')

        if minor_wavelength_um is None or major_wavelength_um is None or reflectance is None:
            print(f"Skipping '{material_name}' from '{source_database}' due to missing wavelength or reflectance data.")
            continue
        try:
            no_conversion_dbs = [
                'database_pigmentcheker1.json',
                'database_NIR_pigmentcheker1.json',
                'database_200to1000_pigmentcheker.json'
            ]
            if source_database in no_conversion_dbs:
                minor_wavelength_nm = float(minor_wavelength_um)
                major_wavelength_nm = float(major_wavelength_um)
            else:
                minor_wavelength_nm = float(minor_wavelength_um) * 1000
                major_wavelength_nm = float(major_wavelength_um) * 1000
        except ValueError:
            print(f"Skipping '{material_name}' from '{source_database}' due to invalid wavelength format.")
            continue
        reflectance_array = np.array(reflectance, dtype=float)
        if len(reflectance_array) > 1:
            wavelengths_nm = np.linspace(minor_wavelength_nm, major_wavelength_nm, len(reflectance_array))
        elif len(reflectance_array) == 1:
            wavelengths_nm = np.array([minor_wavelength_nm])
        else:
            print(f"Skipping '{material_name}' from '{source_database}' due to empty reflectance data.")
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wavelengths_nm,
            y=reflectance_array,
            mode='lines',
            name=f"{material_name}",
            hoverinfo='x+y'
        ))
        fig.update_layout(
            title=f'Espectro de Reflectancia: {material_name} (Origen: {source_database})',
            xaxis_title='Longitud de Onda (nm)',
            yaxis_title='Reflectancia',
            hovermode='x unified'
        )
        figures.append(fig)
    return figures

def plot_single_material_spectrum(material_to_plot):
    """
    Generates a single Plotly figure displaying the spectral data
    (reflectance vs. wavelength) for a *single* material.
    """
    fig = go.Figure()
    if not material_to_plot:
        fig.update_layout(title='No se encontraron datos para graficar', xaxis_title='Longitud de Onda (nm)', yaxis_title='Reflectancia')
        return fig
    material_name = material_to_plot.get('name', 'Unknown Material')
    minor_wavelength_um = material_to_plot.get('minor_wavelength')
    major_wavelength_um = material_to_plot.get('major_wavelength')
    reflectance = material_to_plot.get('reflectance')
    source_database = material_to_plot.get('source_database', 'N/A')

    if minor_wavelength_um is None or major_wavelength_um is None or reflectance is None:
        fig.update_layout(title=f'Error al graficar {material_name} (Faltan datos)', xaxis_title='Longitud de Onda (nm)', yaxis_yaxis_title='Reflectancia')
        return fig
    try:
        no_conversion_dbs = [
            'database_pigmentcheker1.json',
            'database_NIR_pigmentcheker1.json',
            'database_200to1000_pigmentcheker.json'
        ]
        if source_database in no_conversion_dbs:
            minor_wavelength_nm = float(minor_wavelength_um)
            major_wavelength_nm = float(major_wavelength_um)
        else:
            minor_wavelength_nm = float(minor_wavelength_um) * 1000
            major_wavelength_nm = float(major_wavelength_um) * 1000
    except ValueError:
        fig.update_layout(title=f'Error al graficar {material_name} (Formato de onda inválido)', xaxis_title='Longitud de Onda (nm)', yaxis_title='Reflectancia')
        return fig

    reflectance_array = np.array(reflectance, dtype=float)
    if len(reflectance_array) > 1:
        wavelengths_nm = np.linspace(minor_wavelength_nm, major_wavelength_nm, len(reflectance_array))
    elif len(reflectance_array) == 1:
        wavelengths_nm = np.array([minor_wavelength_nm])
    else:
        fig.update_layout(title=f'Error al graficar {material_name} (Datos de reflectancia vacíos)', xaxis_title='Longitud de Onda (nm)', yaxis_title='Reflectancia')
        return fig

    fig.add_trace(go.Scatter(
        x=wavelengths_nm,
        y=reflectance_array,
        mode='lines',
        name=f"{material_name} (Origen: {source_database})",
        hoverinfo='x+y+name'
    ))
    fig.update_layout(
        title=f'Espectro de Reflectancia: {material_name} (Origen: {source_database})',
        xaxis_title='Longitud de Onda (nm)',
        yaxis_title='Reflectancia',
        hovermode='x unified',
        xaxis_tickformat=' ',
        yaxis_tickformat=''
    )
    return fig

def combine_plotly_figures_traces(list_of_figures, title="Resultados de la Búsqueda por Consulta"):
    """
    Combines traces from a list of Plotly figures into a single Plotly figure with subplots,
    where each subplot displays a single material's spectrum.
    """
    if not list_of_figures:
        fig = go.Figure()
        fig.update_layout(
            title="No se encontraron resultados para graficar",
            xaxis_title='Longitud de Onda (nm)',
            yaxis_title='Reflectancia'
        )
        return fig
    num_figures = len(list_of_figures)
    subplot_titles = [
        fig.layout.title.text if fig.layout.title and fig.layout.title.text else f"Material {i+1}"
        for i, fig in enumerate(list_of_figures)
    ]
    fig = make_subplots(rows=num_figures, cols=1,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.05)
    for i, individual_fig in enumerate(list_of_figures):
        for trace in individual_fig.data:
            fig.add_trace(trace, row=i+1, col=1)
        fig.update_yaxes(title_text='Reflectancia', row=i+1, col=1, autorange=True, showgrid=True)
        if i == num_figures - 1:
            fig.update_xaxes(title_text='Longitud de Onda (nm)', row=i+1, col=1, showticklabels=True, showgrid=True)
        else:
            fig.update_xaxes(showticklabels=True, row=i+1, col=1, showgrid=True)
    fig.update_layout(
        title_text=title,
        height=300 * num_figures,
        hovermode='x unified'
    )
    return fig

def export_spectral_data_to_csv(material):
    """
    Converts a single material's spectral data (wavelength and reflectance) into a CSV string.
    """
    material_name = material.get('name', 'unknown_material')
    minor_wavelength_um = material.get('minor_wavelength')
    major_wavelength_um = material.get('major_wavelength')
    reflectance = material.get('reflectance')
    source_database = material.get('source_database', '')

    if minor_wavelength_um is None or major_wavelength_um is None or reflectance is None:
        return ""
    try:
        no_conversion_dbs = [
            'database_pigmentcheker1.json',
            'database_NIR_pigmentcheker1.json',
            'database_200to1000_pigmentcheker.json'
        ]
        if source_database in no_conversion_dbs:
            minor_wavelength_nm = float(minor_wavelength_um)
            major_wavelength_nm = float(major_wavelength_um)
        else:
            minor_wavelength_nm = float(minor_wavelength_um) * 1000
            major_wavelength_nm = float(major_wavelength_um) * 1000
    except ValueError:
        return ""
    reflectance_array = np.array(reflectance, dtype=float)
    if len(reflectance_array) > 1:
        wavelengths_nm = np.linspace(minor_wavelength_nm, major_wavelength_nm, len(reflectance_array))
    elif len(reflectance_array) == 1:
        wavelengths_nm = np.array([minor_wavelength_nm])
    else:
        return ""
    wavelengths_nm = np.round(wavelengths_nm, 1)
    df = pd.DataFrame({
        'Wavelength_nm': wavelengths_nm,
        'Reflectance': reflectance_array
    })
    csv_string = df.to_csv(index=False)
    return csv_string

# --- 5. Gradio Interface Functions ---
def query_search_interface(query):
    """
    Searches the combined spectral database for materials matching the query
    and generates a single Plotly figure for the found materials, combining their traces.
    """
    found_materials = search_database(combined_spectral_data, query)
    if found_materials:
        materials_to_plot = found_materials[:10]
        individual_figures = plot_spectral_data_from_query_results(materials_to_plot)
        combined_figure = combine_plotly_figures_traces(individual_figures)
        return combined_figure
    else:
        fig = go.Figure()
        fig.update_layout(title=f"No se encontraron materiales para la consulta: {query}",
                          xaxis_title="Longitud de Onda (nm)",
                          yaxis_title="Reflectancia")
        return fig

def get_database_names():
    """
    Returns a list of database names available in `loaded_databases`.
    """
    return list(loaded_databases.keys())

def get_materials_from_db(db_name):
    """
    Retrieves a list of material names for a given database.
    """
    materials = []
    if db_name in all_extracted_spectral_data:
        for material_data in all_extracted_spectral_data[db_name]:
            materials.append(material_data.get('name', 'Unknown Material'))
    return gr.update(choices=materials, value=None)

def browse_and_plot_material_interface(db_name, material_name):
    """
    Finds a specific material within a database from `combined_spectral_data`
    and plots its spectral data, also extracting and returning additional metadata.
    """
    found_material = None
    for material in combined_spectral_data:
        if material.get('source_database') == db_name and material.get('name') == material_name:
            found_material = material
            break
    if found_material:
        figure = plot_single_material_spectrum(found_material)
        metadata_parts = []
        excluded_keys = ['name', 'minor_wavelength', 'major_wavelength', 'reflectance', 'source_database', 'category']
        for key, value in found_material.items():
            if key not in excluded_keys and value is not None:
                display_key = key.replace('_', ' ').title()
                metadata_parts.append(f"**{display_key}**: {value}")
        metadata_string = "\n".join(metadata_parts) if metadata_parts else "No hay metadatos adicionales disponibles para este material."
        return figure, metadata_string
    else:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No se encontró el material seleccionado.")
        return empty_fig, "No se encontró el material seleccionado en la base de datos."

def export_selected_material_csv(db_name, material_name):
    """
    Generates a temporary CSV file for a selected material from a specific database
    and returns its path for Gradio's DownloadButton.
    """
    found_material = None
    for material in combined_spectral_data:
        if material.get('source_database') == db_name and material.get('name') == material_name:
            found_material = material
            break
    if found_material:
        csv_string = export_spectral_data_to_csv(found_material)
        if csv_string:
            filename_base = f"{material_name.replace(' ', '_').replace('/', '-')}_spectral_data"
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as tmp_file:
                tmp_file.write(csv_string)
            return tmp_file.name, "¡Exportado correctamente! Haga clic en el botón 'Descargar CSV' para bajar el archivo."
        else:
            return None, "Error al generar CSV: Los datos del material están incompletos o son inválidos."
    else:
        return None, "Error: No se encontró el material seleccionado en la base de datos."

# --- 6. Gradio App Layout ---
with gr.Blocks(title="Spectral Data Explorer") as demo:
    # Load logo image for display
    if os.path.exists(logo_image_path):
        try:
            logo_image = Image.open(logo_image_path)
        except Exception as e:
            print(f"An error occurred while loading the logo image: {e}. Displaying placeholder.")
            logo_image = None
    else:
        print(f"Logo image not found at {logo_image_path}. Displaying placeholder.")
        logo_image = None

    if logo_image:
        gr.Image(logo_image, label="Banana Spectral Database Explorer", show_label=True, height=100)
    else:
        gr.Markdown("# Banana Spectral Database Explorer (Logo Missing or Error)")

    gr.Markdown("# Banana Spectral Database Explorer")

    with gr.Tab("Búsqueda por Consulta"):
        with gr.Column():
            gr.Markdown("## Búsqueda Directa por Consulta")
            query_input = gr.Textbox(label="Introduzca su consulta (ej. 'azurite', 'goethite')", placeholder="Escriba aquí...")
            search_button = gr.Button("Buscar y Graficar")
            query_output_plot = gr.Plot(label="Resultados de la Búsqueda")

        search_button.click(
            fn=query_search_interface,
            inputs=query_input,
            outputs=query_output_plot
        )

    with gr.Tab("Explorar por Base de Datos"):
        with gr.Column():
            gr.Markdown("## Explorar por Base de Datos y Material")
            db_dropdown = gr.Dropdown(
                choices=get_database_names(),
                label="Seleccione Base de Datos",
                interactive=True
            )
            material_dropdown = gr.Dropdown(
                choices=[],
                label="Seleccione Material",
                interactive=True
            )
            plot_button = gr.Button("Graficar Material Seleccionado")
            browse_output_plot = gr.Plot(label="Espectro del Material")
            metadata_display = gr.Markdown(label="Metadatos Adicionales")

            gr.Markdown("### Exportar Datos Espectrales a CSV")
            export_csv_button = gr.Button("Exportar a CSV")
            csv_file_output = gr.DownloadButton(label="Descargar CSV", interactive=True)
            export_status_message = gr.Textbox(label="Estado de Exportación", interactive=False, value="")

        db_dropdown.change(
            fn=get_materials_from_db,
            inputs=db_dropdown,
            outputs=material_dropdown
        )

        plot_button.click(
            fn=browse_and_plot_material_interface,
            inputs=[db_dropdown, material_dropdown],
            outputs=[browse_output_plot, metadata_display]
        )

        export_csv_button.click(
            fn=export_selected_material_csv,
            inputs=[db_dropdown, material_dropdown],
            outputs=[csv_file_output, export_status_message]
        )

demo.launch()


if __name__ == "__main__":
    demo.launch()