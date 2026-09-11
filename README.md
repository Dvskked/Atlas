# Atlas

Sistema de gestión de reciclaje inteligente con inteligencia artificial. Detecta botellas PET, tapas y etiquetas mediante YOLOv8 y asigna puntos de reciclaje (AtlasPuntos) a los usuarios, quienes pueden canjearlos por productos ecológicos del catálogo.

El proyecto se compone de dos aplicaciones:

- **Sistema principal (hardware)**: aplicación de escritorio (Electron + Flask) que controla la máquina de reciclaje, realiza el escaneo inteligente con YOLOv8 y expone el panel administrativo.
- **App de usuario (`app-usuario/`)**: aplicación web independiente a la que el usuario accede escaneando el **código QR** de la máquina. Permite registrarse, iniciar sesión y consultar su información, puntos y clasificación en tiempo real, sin necesidad de la máquina.

Desarrollado por **Andres Forero**, desarrollador semi junior de software enfocado en análisis de datos y desarrollo de escritorio.

## Funcionalidades

### Usuario
- **Acceso por QR**: el usuario escanea el código QR de la máquina de reciclaje y accede a la app web de usuario (`app-usuario/`).
- **Registro e inicio de sesión** con número de identificación y roles (USUARIO / ADMINISTRADOR)
- **Detección IA**: escanea botellas PET mediante la cámara usando un modelo YOLOv8 entrenado con dataset propio; identifica botella, tapa y etiqueta (solo en el sistema de hardware)
- **Sistema de AtlasPuntos**: asigna puntos automáticamente según lo detectado (50 base + 10 tapa + 5 etiqueta)
- **Comprobantes de reciclaje**: genera automáticamente un comprobante numerado (ATLA-XXXXXX) por cada reciclaje registrado
- **Catálogo / Tienda**: canjea AtlasPuntos por productos ecológicos
- **Dashboard** (app de usuario): saldo de puntos, botellas recicladas, puntos obtenidos, último reciclaje, movimientos recientes y posición en el ranking
- **Clasificación** (app de usuario): tablas de mayores AtlasPuntos, más reciclajes y más canjes, con posición del usuario
- **Términos y condiciones**: consentimiento de tratamiento de datos personales (Ley 1581 de 2012) al registrarse, registrado en la base de datos
- **Correos automáticos**: bienvenida, recuperación de usuario y restablecimiento de contraseña
- **Comprobantes de canje**: numeración CANJ-XXXXXX

### Administrador
- **Panel administrativo** con estadísticas globales (usuarios, escaneos IA, puntos entregados, productos)
- **Gestión de usuarios**: crear, editar y eliminar usuarios; asignar rol y tipo de cuenta
- **Gestión de puntos**: agregar (bonificación) o quitar (penalización) AtlasPuntos manualmente
- **Gestión de productos / catálogo**: administrar catálogo de canje
- **Registro de reciclajes**: historial completo de análisis IA y comprobantes
- **Movimientos y canjes**: gestionar el intercambio de puntos por productos desde el panel
- **Auditoría**: registro de todas las acciones administrativas (quién, qué, cuándo e IP), usuarios nuevos, canjes recientes, análisis IA recientes, movimientos de puntos y **ranking** de recicladores
- **Ranking**: top de usuarios con más AtlasPuntos

## Tecnologías

### Backend
- Python 3.11 / Flask
- MySQL (gestionado en Clever Cloud)
- gunicorn (servidor de producción)

### IA / Visión
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch (CPU)

### Frontend
- HTML / CSS / JavaScript (DOM)
- Jinja2 (motor de plantillas)

### Desktop / Distribución
- Electron (aplicación de escritorio)
- PyInstaller (empaquetado del backend Python)
- electron-builder (instalador NSIS para Windows)

### Despliegue
- Render (Web Service, configurado con `render.yaml` y `Procfile`)
- Clever Cloud (base de datos MySQL)

## Estructura del proyecto

```
Atlas/
├── app.py              # Aplicación principal Flask (rutas, API, lógica, IA)
├── app-usuario/        # App web independiente del usuario (QR -> ver su README)
├── conexion.py         # Conexión a MySQL (Clever Cloud)
├── train.py            # Entrenamiento del modelo YOLO
├── camara.py           # Prueba de cámara en tiempo real
├── main.js             # Aplicación de escritorio Electron
├── app.spec            # Configuración de empaquetado PyInstaller
├── requirements.txt    # Dependencias Python
├── runtime.txt         # Versión de Python para despliegue
├── Procfile            # Comando de inicio en Render
├── render.yaml         # Configuración del despliegue en Render
├── migracion_acepta_terminos.sql  # Migración de consentimiento de datos
├── package.json        # Configuración de Electron / electron-builder
├── yolov8n.pt          # Pesos base de YOLOv8n
├── runs/detect/        # Pesos del modelo entrenado (train-5/best.pt)
├── static/             # Archivos estáticos (CSS, JS, imágenes)
├── templates/          # Plantillas Jinja2
│   ├── auth/           # Login y registro
│   ├── usuario/        # Dashboard, escaneo, catálogo, comprobantes
│   └── admin/          # Panel administrativo
├── build/              # Artefactos de electron-builder
└── dist/               # Aplicación empaquetada (Electron + PyInstaller)
```

## Requisitos

- Python 3.9+
- MySQL 8.0+
- pip
- Node.js 18+ y npm (solo para construir la app de escritorio)

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd Atlas

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración de la base de datos

1. Crear la base de datos en MySQL (local):

```sql
CREATE DATABASE atlas;
```

2. Importar el esquema SQL (`atlas (3).sql`) si se dispone de él.

3. Actualizar la conexión en `conexion.py` (por defecto apunta a una instancia gestionada en Clever Cloud):

```python
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="atlas",
    port=3306
)
```

4. Aplicar la migración del consentimiento de datos (`migracion_acepta_terminos.sql`) si la tabla `usuarios` aún no tiene la columna `acepto_terminos`:

```bash
mysql -h <host> -u <usuario> -p <basededatos> < migracion_acepta_terminos.sql
```

## Ejecución en modo web

```bash
python app.py
```

El servidor arranca en `http://localhost:5000`. El modelo de IA se carga desde `runs/detect/train-5/weights/best.pt`; si no está disponible, las funciones de escaneo quedan inhabilitadas con un mensaje de error.

## Ejecución como aplicación de escritorio (Electron)

La app combina un backend Flask (Python) con una ventana Electron (frontend).

**Modo desarrollo:**

```bash
npm install
npm start   # inicia Python (app.py) y abre la ventana de Electron
```

**Generar instalador de Windows (NSIS):**

```bash
# 1. Empaquetar el backend Python con PyInstaller
pyinstaller app.spec

# 2. Crear el instalador de Electron
npm run dist
```

El instalador se genera en la carpeta `dist/`.

## Entrenamiento del modelo

```bash
python train.py
```

Entrena YOLOv8n con el dataset propio. Los pesos se guardan en `runs/detect/`. El modelo usado por la aplicación está en `runs/detect/train-5/weights/best.pt`.

## Despliegue en Render

Las dos aplicaciones se despliegan como servicios independientes definidos en sus respectivos `render.yaml` y `Procfile`. Render redespliega automáticamente con cada push a la rama principal.

### Sistema principal (atlas)
- Definido en `render.yaml` y `Procfile` del directorio raíz.
- **URL de producción:** `https://atlas.onrender.com`
- **CORS:** configura los orígenes permitidos con la variable de entorno `CORS_ORIGINS` (por defecto incluye el dominio de Render).

### App de usuario (atlas-usuario)
- Definido en `app-usuario/render.yaml` y `app-usuario/Procfile` (usando `rootDir: app-usuario`).
- **URL de producción:** `https://atlas-usuario.onrender.com`
- El **código QR** de la máquina de reciclaje debe apuntar a esta URL para que los usuarios accedan a su cuenta.
- Las credenciales SMTP se configuran como variables de entorno (`ATLAS_SMTP_USER`, `ATLAS_SMTP_PASS`, etc.).

## Roles de usuario

| Rol | Sistema principal | App de usuario |
|-----|-------------------|----------------|
| USUARIO | Escaneo, catálogo, canje, dashboard | Dashboard, clasificación, información, gestión de cuenta |
| ADMINISTRADOR | Gestión completa (usuarios, puntos, catálogo, canjes, auditoría) | Bloqueado: usa el panel del sistema principal |
