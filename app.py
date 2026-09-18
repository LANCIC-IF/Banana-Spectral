import os
import json
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
import gc

# --- 1. Define Local Paths ---
BASE_DIR = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
DB_FOLDER = os.path.join(BASE_DIR, "Databases")
logo_image_path = os.path.join(BASE_DIR, "banana spectral.jpeg")

# --- 2. Data Loading and Processing Functions ---
def load_json_databases(folder_path):
    """
    Loads all JSON files directly from the local Databases directory into a dictionary.
    """
    loaded_databases = {}
    
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' not found.")
        return loaded_databases

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for filename in json_files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                loaded_databases[filename] = data
            print(f"  - Loaded '{filename}' locally")
        except Exception as e:
            print(f"  - Error loading '{filename}': {e}")

    return loaded_databases

def detect_hierarchy_type(db):
    if not isinstance(db, dict) or not db:
        return 'Unknown'

    first_level_value = next(iter(db.values()), None)
    if isinstance(first_level_value, dict) and first_level_value:
        second_level_value = next(iter(first_level_value.values()), None)
        if isinstance(second_level_value, dict):
            if all(k in second_level_value for k in ['minor_wavelength', 'major_wavelength', 'reflectance']):
                return 'Tipo 2'

    if isinstance(first_level_value, dict):
        if all(k in first_level_value for k in ['minor_wavelength', 'major_wavelength', 'reflectance']):
            return 'Tipo 1'

    return 'Unknown'

def extract_data_type1(db):
    extracted_data = []
    for material_name, properties in db.items():
        material_info = {'name': material_name}
        for key, value in properties.items():
            material_info[key] = value
        extracted_data.append(material_info)
    return extracted_data

def extract_data_type2(db):
    extracted_data = []
    for category, category_data in db.items():
        if isinstance(category_data, dict):
            for material_name, properties in category_data.items():
                material_info = {'name': material_name, 'category': category}
                for key, value in properties.items():
                    material_info[key] = value
                extracted_data.append(material_info)
    return extracted_data

# --- 3. Initial Data Loading and Processing ---
print("Loading databases locally...")
all_extracted_spectral_data = {}
combined_spectral_data = []

if os.path.exists(DB_FOLDER):
    json_files = [f for f in os.listdir(DB_FOLDER) if f.endswith('.json')]
    for filename in json_files:
        filepath = os.path.join(DB_FOLDER, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                db_content = json.load(f)
            
            hierarchy_type = detect_hierarchy_type(db_content)
            extracted_list = []
            if hierarchy_type == 'Tipo 1':
                extracted_list = extract_data_type1(db_content)
            elif hierarchy_type == 'Tipo 2':
                extracted_list = extract_data_type2(db_content)
            
            if extracted_list:
                all_extracted_spectral_data[filename] = extracted_list
                for material in extracted_list:
                    mat_copy = material.copy()
                    mat_copy['source_database'] = filename
                    combined_spectral_data.append(mat_copy)
            
            # Liberar la memoria RAM del archivo pesado inmediatamente
            del db_content
            gc.collect()
            print(f"  - Procesado y liberado de RAM: '{filename}'")
        except Exception as e:
            print(f"  - Error al cargar '{filename}': {e}")

print(f"Procesamiento completo. Total de materiales: {len(combined_spectral_data)}")

# Load logo image
try:
    logo_image = Image.open(logo_image_path)
    print(f"Logo image loaded successfully from {logo_image_path}.")
except Exception as e:
    print(f"Notice: Logo image not loaded: {e}. Using placeholder.")
    logo_image = None

# --- 4. Core Application Functions ---
def search_database(combined_data, query):
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
    figures = []
    for material in materials_to_plot:
        material_name = material.get('name', 'Unknown Material')
        minor_wavelength_um = material.get('minor_wavelength')
        major_wavelength_um = material.get('major_wavelength')
        reflectance = material.get('reflectance')
        source_database = material.get('source_database', 'N/A')

        if minor_wavelength_um is None or major_wavelength_um is None or reflectance is None:
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
            continue
        reflectance_array = np.array(reflectance, dtype=float)
        if len(reflectance_array) > 1:
            wavelengths_nm = np.linspace(minor_wavelength_nm, major_wavelength_nm, len(reflectance_array))
        elif len(reflectance_array) == 1:
            wavelengths_nm = np.array([minor_wavelength_nm])
        else:
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
        fig.update_layout(title=f'Error al graficar {material_name} (Faltan datos)', xaxis_title='Longitud de Onda (nm)', yaxis_title='Reflectancia')
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
        hovermode='x unified'
    )
    return fig

def combine_plotly_figures_traces(list_of_figures, title="Resultados de la Búsqueda por Consulta"):
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
    return df.to_csv(index=False)

# --- 5. Gradio Interface Functions ---
def query_search_interface(query):
    found_materials = search_database(combined_spectral_data, query)
    if found_materials:
        materials_to_plot = found_materials[:10]
        individual_figures = plot_spectral_data_from_query_results(materials_to_plot)
        return combine_plotly_figures_traces(individual_figures)
    else:
        fig = go.Figure()
        fig.update_layout(title=f"No se encontraron materiales para la consulta: {query}",
                          xaxis_title="Longitud de Onda (nm)",
                          yaxis_title="Reflectancia")
        return fig

def get_database_names():
    return list(all_extracted_spectral_data.keys())

def get_materials_from_db(db_name):
    materials = []
    if db_name in all_extracted_spectral_data:
        for material_data in all_extracted_spectral_data[db_name]:
            materials.append(material_data.get('name', 'Unknown Material'))
    return gr.update(choices=materials, value=None)

def browse_and_plot_material_interface(db_name, material_name):
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
    found_material = None
    for material in combined_spectral_data:
        if material.get('source_database') == db_name and material.get('name') == material_name:
            found_material = material
            break
    if found_material:
        csv_string = export_spectral_data_to_csv(found_material)
        if csv_string:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as tmp_file:
                tmp_file.write(csv_string)
            return tmp_file.name, "¡Exportado correctamente! Haga clic en el botón 'Descargar CSV' para bajar el archivo."
        else:
            return None, "Error al generar CSV: Los datos del material están incompletos o son inválidos."
    else:
        return None, "Error: No se encontró el material seleccionado en la base de datos."

# --- 6. Gradio App Layout ---
with gr.Blocks(title="Spectral Data Explorer") as demo:
    if logo_image:
        gr.Image(logo_image, label="Banana Spectral Database Explorer", show_label=True, height=100)
    else:
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

# --- 7. Execution Block for Cloud Deployment (Koyeb Dynamic Port) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)