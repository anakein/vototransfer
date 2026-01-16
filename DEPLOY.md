# Guía de Despliegue a Producción

Esta aplicación está construida con **Streamlit**. Aquí tienes las mejores opciones para ponerla online.

## Opción 1: Streamlit Community Cloud (Gratis y Fácil)
Ideal si el proyecto es público o para demos rápidas.

1. Sube tu código a un repositorio de **GitHub**.
2. Regístrate en [share.streamlit.io](https://share.streamlit.io/).
3. Conecta tu cuenta de GitHub y selecciona el repositorio.
4. Streamlit detectará automáticamente `requirements.txt` e instalará todo.

## Opción 2: Docker (Recomendado para Empresas/VPS)
Si tienes un servidor propio o usas servicios como AWS, Azure, o Google Cloud.

1. Asegúrate de tener el archivo `Dockerfile` en la raíz (ya creado).
2. Construye la imagen:
   ```bash
   docker build -t elecciones-andalucia .
   ```
3. Ejecuta el contenedor:
   ```bash
   docker run -p 8501:8501 elecciones-andalucia
   ```
4. Accede en `http://tu-servidor:8501`.

## Opción 3: Servidor Linux (VPS tradicional)
Si prefieres instalarlo manualmente en un Ubuntu/Debian.

1. Instala Python y entorno virtual:
   ```bash
   sudo apt update && sudo apt install python3-venv
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la app en segundo plano (usando `nohup` o `systemd`):
   ```bash
   nohup streamlit run src/app.py --server.port 80 &
   ```

## Estructura de Archivos Necesaria
Para que funcione, el servidor debe tener esta estructura mínima:

```text
/app
  ├── requirements.txt
  ├── Dockerfile (opcional)
  ├── src/
  │    ├── app.py
  │    ├── data_processing.py
  │    ├── clustering.py
  │    ├── inference_model.py
  │    └── visualization.py
  └── datos/
       └── normalizado.csv
```
**Nota Importante**: Asegúrate de subir el archivo `datos/normalizado.csv` al servidor, ya que la app lo necesita para funcionar.
