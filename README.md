# IAAgentsFinalGerardoRodITAM
Este es el repositorio de mi proyecto final de IA Agents


# AI-Powered Text Analysis & Content Generation Platform

¡Bienvenido a la solución integral para análisis, transformación y generación de contenidos académicos y técnicos usando inteligencia artificial y AWS!

## Descripción

Este proyecto proporciona una interfaz interactiva en Streamlit que permite a los usuarios realizar tareas avanzadas de análisis textual, transformación de estilos, evaluación de calidad, generación de referencias, creación de documentación en LaTeX, y exploración de hipótesis, todo integrado con los servicios en la nube de AWS (Bedrock y S3). La plataforma está diseñada para facilitar el trabajo profesional, académico y de investigación, democratizando el acceso a modelos de IA potentes y configurables.

## Funcionalidades principales

- **Análisis profundo de textos:** descomposición en componentes clave, correcciones y propuestas mejoradas.
- **Cambio de tono y estilo:** adapta textos a diferentes audiencias y propósitos.
- **Explorador de hipótesis:** genera ideas, estadísticas, referencias y esquemas de investigación.
- **Evaluación de calidad:** califica y corrige textos en aspectos como ortografía, gramática, coherencia y estilo.
- **Generación de código LaTeX:** crea informes estructurados en RMarkdown listos para exportar a PDF.
- **Formateo de referencias:** cita en estilos APA, MLA y Chicago.
- **Gestión en la nube:** configuración y acceso a modelos de AWS Bedrock y almacenamiento en S3.
- **Interfaz amigable y modular:** navegación sencilla con pestañas para cada funcionalidad.
- **Soporte técnico y ayuda:** guía para configuración y resolución de problemas en AWS.

## Tecnologías utilizadas

- Python 3.x
- Streamlit
- boto3 (AWS SDK para Python)
- PyPDF2
- JSON y Standard Python Libraries
- Servicios AWS: Bedrock y S3

## Requisitos

- Cuenta en AWS con acceso a Bedrock y permisos adecuados.
- AWS CLI configurado con perfiles y credenciales.
- Python 3.x instalado.
- Librerías requeridas (instalables vía `pip install -r requirements.txt`).

## Instalación y Ejecución

### Pasos para correr el proyecto:

1. **Clona este repositorio:**

```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
```

2. **Accede a la carpeta del proyecto:**
```bash
Copy
cd tu-repositorio
```
3. **Instala las dependencias necesarias:**
```bash
Copy
pip install -r requirements.txt
```
4. **Configura tus credenciales AWS:**

Asegúrate de tener AWS CLI instalado y configurado con un perfil que tenga permisos en Bedrock y S3:
bash
Copy
aws configure --profile recruitment-assistant
5. **Verifica que puedes listar modelos Bedrock y buckets S3:**
```bash
Copy
aws bedrock list-foundation-models --region us-east-1 --recruitment-assistant
aws s3 ls --profile recruitment-assistant
```
6. **Ejecuta la aplicación con Streamlit:**
bash
Copy
streamlit run app.py

