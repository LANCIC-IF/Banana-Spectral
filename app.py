import gradio as gr

def saludar(nombre):
    return f"¡Hola {nombre}! Esta es la app oficial del laboratorio LANCIC-IF en Hugging Face."

# Crear la interfaz
demo = gr.Interface(
    fn=saludar,
    inputs=gr.Textbox(label="Tu nombre", placeholder="Escribe tu nombre aquí..."),
    outputs=gr.Textbox(label="Mensaje"),
    title="LANCIC-IF - Laboratorio App",
    description="Aplicación de prueba inicial."
)

if __name__ == "__main__":
    demo.launch()