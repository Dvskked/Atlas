import cv2
import numpy as np
import base64
import os
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from flask import (
    Flask,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    render_template,
    flash,
    send_file
)

from io import BytesIO
from datetime import datetime
import re
import smtplib
import secrets
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from werkzeug.security import generate_password_hash, check_password_hash

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)

from ultralytics import YOLO

from conexion import obtener_conexion


def validar_contrasena(contrasena):
    """Valida una contraseña con una política mínima de seguridad.

    Requisitos:
      - Mínimo 8 caracteres
      - Al menos una mayúscula
      - Al menos una minúscula
      - Al menos un número
    """
    if not contrasena or len(contrasena) < 8 or len(contrasena) > 255:
        return False

    if not re.search(r"[A-Z]", contrasena):
        return False

    if not re.search(r"[a-z]", contrasena):
        return False

    if not re.search(r"[0-9]", contrasena):
        return False

    return True


def generar_contrasena_temporal():
    """Genera una contraseña aleatoria que cumple la política mínima."""
    minusculas = string.ascii_lowercase
    mayusculas = string.ascii_uppercase
    numeros = string.digits
    simbolos = "!@#$%&*"

    # Al menos una de cada grupo requerido.
    contrasena = (
        secrets.choice(mayusculas)
        + secrets.choice(minusculas)
        + secrets.choice(numeros)
    )

    # Completar hasta 12 caracteres con una mezcla segura de todos los grupos.
    todos = minusculas + mayusculas + numeros + simbolos
    contrasena += "".join(
        secrets.choice(todos) for _ in range(9)
    )

    # Mezclamos para que los caracteres requeridos no queden fijos al inicio.
    lista = list(contrasena)
    secrets.SystemRandom().shuffle(lista)
    contrasena = "".join(lista)

    return contrasena


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
model_error = None

try:
    model_path = os.path.join(
        BASE_DIR,
        'runs', 'detect',
        'train-5', 'weights',
        'best.pt'
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Modelo no encontrado en: " + model_path
        )

    model = YOLO(model_path)

    print("======================================")
    print("MODELO Atlas CARGADO")
    print("RUTA:", model.ckpt_path)
    print("CLASES:", model.names)
    print("======================================")

except Exception as e:
    model_error = str(e)
    print("======================================")
    print("ERROR CARGANDO MODELO:", e)
    print("======================================")

app = Flask(__name__)
app.secret_key = os.getenv(
    "ATLAS_SECRET_KEY",
    "Atlas_CAMBIAR_ESTA_CLAVE"
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "https://atlas.onrender.com"
    ).split(",")
    if origin.strip()
]

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")

    if origin in cors_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Vary"] = "Origin"

    return response

# ==========================================
# ENVÍO DE CORREOS (SMTP)
# ==========================================
def enviar_correo(destinatario, asunto, texto_plano):
    """Envía un correo de texto plano usando SMTP.

    La configuración se toma de variables de entorno:

      ATLAS_SMTP_HOST      (ej. smtp.gmail.com)
      ATLAS_SMTP_PORT      (ej. 587)
      ATLAS_SMTP_USER      (correo que envía)
      ATLAS_SMTP_PASS      (contraseña / app password)
      ATLAS_SMTP_FROM      (correo remitente, por defecto ATLAS_SMTP_USER)
      ATLAS_SMTP_TLS       ("1" para STARTTLS, por defecto activo)

    Si no se definen las variables, se usan las credenciales de la cuenta
    configurada por defecto (Gmail).

    Devuelve True si el correo se envió correctamente, False en caso contrario.
    """
    host = os.getenv("ATLAS_SMTP_HOST", "").strip() or "smtp.gmail.com"
    port = int(os.getenv("ATLAS_SMTP_PORT", "").strip() or "587")
    usuario = os.getenv("ATLAS_SMTP_USER", "").strip() or "siriusplanet76@gmail.com"
    contrasena = os.getenv("ATLAS_SMTP_PASS", "").strip() or "bsbk gjwa gthf pean"
    remitente = os.getenv("ATLAS_SMTP_FROM", "").strip() or usuario
    usar_tls = os.getenv("ATLAS_SMTP_TLS", "1").strip() == "1"

    if not host or not usuario or not contrasena:
        print("ERROR CORREO: Faltan variables ATLAS_SMTP_* de configuración.")
        return False

    if not destinatario or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", destinatario):
        print("ERROR CORREO: Dirección de destino no válida:", destinatario)
        return False

    mensaje = MIMEMultipart()
    mensaje["From"] = remitente
    mensaje["To"] = destinatario
    mensaje["Subject"] = asunto
    mensaje.attach(MIMEText(texto_plano, "plain", "utf-8"))

    try:
        servidor = smtplib.SMTP(host, port, timeout=30)
        servidor.ehlo()

        if usar_tls:
            servidor.starttls()
            servidor.ehlo()

        servidor.login(usuario, contrasena)
        servidor.sendmail(remitente, [destinatario], mensaje.as_string())
        servidor.quit()

        print("CORREO ENVIADO A:", destinatario)
        return True

    except Exception as e:
        print("ERROR ENVIANDO CORREO:", e)
        return False


def enviar_correo_html(destinatario, asunto, html_contenido, texto_plano=""):
    """Envía un correo HTML usando SMTP.

    Misma configuración de variables de entorno que enviar_correo().
    Si texto_plano se omite, se genera una versión básica del HTML.
    """
    host = os.getenv("ATLAS_SMTP_HOST", "").strip() or "smtp.gmail.com"
    port = int(os.getenv("ATLAS_SMTP_PORT", "").strip() or "587")
    usuario = os.getenv("ATLAS_SMTP_USER", "").strip() or "siriusplanet76@gmail.com"
    contrasena = os.getenv("ATLAS_SMTP_PASS", "").strip() or "bsbk gjwa gthf pean"
    remitente = os.getenv("ATLAS_SMTP_FROM", "").strip() or usuario
    usar_tls = os.getenv("ATLAS_SMTP_TLS", "1").strip() == "1"

    if not host or not usuario or not contrasena:
        print("ERROR CORREO HTML: Faltan variables ATLAS_SMTP_* de configuración.")
        return False

    if not destinatario or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", destinatario):
        print("ERROR CORREO HTML: Dirección de destino no válida:", destinatario)
        return False

    mensaje = MIMEMultipart("alternative")
    mensaje["From"] = remitente
    mensaje["To"] = destinatario
    mensaje["Subject"] = asunto

    if texto_plano:
        mensaje.attach(MIMEText(texto_plano, "plain", "utf-8"))

    mensaje.attach(MIMEText(html_contenido, "html", "utf-8"))

    try:
        servidor = smtplib.SMTP(host, port, timeout=30)
        servidor.ehlo()

        if usar_tls:
            servidor.starttls()
            servidor.ehlo()

        servidor.login(usuario, contrasena)
        servidor.sendmail(remitente, [destinatario], mensaje.as_string())
        servidor.quit()

        print("CORREO HTML ENVIADO A:", destinatario)
        return True

    except Exception as e:
        print("ERROR ENVIANDO CORREO HTML:", e)
        return False


def _plantilla_bienvenida(nombre, usuario, correo, fecha_registro, telefono=""):
    """Genera el contenido HTML del correo de bienvenida de Atlas."""

    telefono_bloque = ""
    if telefono:
        telefono_bloque = f"""
                        <tr>
                            <td style="padding:8px 0;color:#555;font-size:14px;">Telefono:</td>
                            <td style="padding:8px 0;color:#1a1a2e;font-weight:600;font-size:14px;">{telefono}</td>
                        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background-color:#f4f6f8;font-family:'Segoe UI',Arial,Helvetica,sans-serif;">

    <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f6f8;padding:40px 20px;">
        <tr>
            <td align="center">

                <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08);">

                    <!-- HEADER -->
                    <tr>
                        <td style="background:linear-gradient(135deg,#0f3460,#16213e);padding:40px 30px;text-align:center;">
                            <h1 style="margin:0;color:#ffffff;font-size:28px;font-weight:700;letter-spacing:1px;">
                                &#127758; ATLAS
                            </h1>
                            <p style="margin:8px 0 0;color:#a8d8ea;font-size:13px;letter-spacing:0.5px;">
                                SISTEMA DE GESTION DE RECICLAJE INTELIGENTE
                            </p>
                        </td>
                    </tr>

                    <!-- SALUDO -->
                    <tr>
                        <td style="padding:35px 35px 10px;text-align:center;">
                            <h2 style="margin:0;color:#1a1a2e;font-size:22px;font-weight:700;">
                                !Bienvenido a Atlas, {nombre}!
                            </h2>
                            <p style="margin:12px 0 0;color:#555;font-size:14px;line-height:1.6;">
                                Tu cuenta ha sido creada exitosamente. Ahora formas parte de una comunidad
                                comprometida con el reciclaje inteligente y la automatizacion para la UATF.
                            </p>
                        </td>
                    </tr>

                    <!-- SEPARADOR -->
                    <tr>
                        <td style="padding:0 35px;">
                            <hr style="border:none;border-top:1px solid #e8ecf1;margin:10px 0;">
                        </td>
                    </tr>

                    <!-- QUE ES ATLAS -->
                    <tr>
                        <td style="padding:15px 35px;">
                            <h3 style="margin:0 0 8px;color:#0f3460;font-size:16px;">
                                &#9881; Que es Atlas?
                            </h3>
                            <p style="margin:0;color:#555;font-size:13px;line-height:1.7;">
                                Atlas es un <strong style="color:#1a1a2e;">sistema de gestion inteligente</strong>
                                disenado para la automatizacion y trazabilidad del reciclaje en la
                                Universidad Autonoma de Tomina. Utiliza inteligencia artificial (YOLOv8)
                                para detectar botellas PET, tapas y etiquetas en tiempo real, recompensandote
                                con <strong style="color:#0f3460;">AtlasPuntos</strong> por cada reciclaje.
                            </p>
                        </td>
                    </tr>

                    <!-- TUS DATOS -->
                    <tr>
                        <td style="padding:15px 35px;">
                            <h3 style="margin:0 0 12px;color:#0f3460;font-size:16px;">
                                &#128196; Tus datos de registro
                            </h3>
                            <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f8f9fb;border-radius:8px;border:1px solid #e8ecf1;">
                                <tr>
                                    <td style="padding:8px 15px;color:#555;font-size:14px;">Nombre:</td>
                                    <td style="padding:8px 15px;color:#1a1a2e;font-weight:600;font-size:14px;">{nombre}</td>
                                </tr>
                                <tr style="background-color:#f0f2f5;">
                                    <td style="padding:8px 15px;color:#555;font-size:14px;">Usuario:</td>
                                    <td style="padding:8px 15px;color:#1a1a2e;font-weight:600;font-size:14px;">{usuario}</td>
                                </tr>
                                <tr>
                                    <td style="padding:8px 15px;color:#555;font-size:14px;">Correo:</td>
                                    <td style="padding:8px 15px;color:#1a1a2e;font-weight:600;font-size:14px;">{correo}</td>
                                </tr>
                                <tr style="background-color:#f0f2f5;">
                                    <td style="padding:8px 15px;color:#555;font-size:14px;">Fecha de registro:</td>
                                    <td style="padding:8px 15px;color:#1a1a2e;font-weight:600;font-size:14px;">{fecha_registro}</td>
                                </tr>
                                {telefono_bloque}
                            </table>
                        </td>
                    </tr>

                    <!-- PASOS SIGUIENTES -->
                    <tr>
                        <td style="padding:15px 35px;">
                            <h3 style="margin:0 0 10px;color:#0f3460;font-size:16px;">
                                &#128640; Siguientes pasos
                            </h3>
                            <table width="100%" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td style="padding:5px 0;color:#555;font-size:13px;line-height:1.6;">
                                        <strong style="color:#0f3460;">1.</strong> Inicia sesion con tu usuario y contrasena.<br>
                                        <strong style="color:#0f3460;">2.</strong> Acumula AtlasPuntos escaneando botellas PET con la camara.<br>
                                        <strong style="color:#0f3460;">3.</strong> Canjea tus puntos por productos ecologicos en el catalogo.<br>
                                        <strong style="color:#0f3460;">4.</strong> Contribuye al medio ambiente y a la automatizacion inteligente.
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- CONTACTO -->
                    <tr>
                        <td style="padding:15px 35px;">
                            <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0f7ff;border-radius:8px;border:1px solid #d0e3f7;">
                                <tr>
                                    <td style="padding:15px 20px;">
                                        <p style="margin:0 0 6px;color:#0f3460;font-size:13px;font-weight:700;">
                                            &#128231; ¿Necesitas ayuda?
                                        </p>
                                        <p style="margin:0;color:#555;font-size:13px;line-height:1.6;">
                                            Si tienes dudas o necesitas asistencia, contacta al desarrollador del proyecto:<br>
                                            <strong style="color:#1a1a2e;">Correo:</strong>
                                            <a href="mailto:siriusplanet76@gmail.com" style="color:#0f3460;text-decoration:none;">siriusplanet76@gmail.com</a><br>
                                            <strong style="color:#1a1a2e;">Telefono:</strong>
                                            <span style="color:#0f3460;">+57 3153806797</span>
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- FOOTER -->
                    <tr>
                        <td style="background-color:#f4f6f8;padding:25px 35px;text-align:center;border-top:1px solid #e8ecf1;">
                            <p style="margin:0 0 5px;color:#888;font-size:11px;">
                                Atlas - Sistema de Gestion de Reciclaje Inteligente
                            </p>
                            <p style="margin:0;color:#aaa;font-size:10px;">
                                Universidad Autonoma de Tomina &copy; 2025. Todos los derechos reservados.
                            </p>
                        </td>
                    </tr>

                </table>

            </td>
        </tr>
    </table>

</body>
</html>"""

    texto_plano = (
        f"!Bienvenido a Atlas, {nombre}!\n\n"
        f"Tu cuenta ha sido creada exitosamente.\n\n"
        f"Tus datos:\n"
        f"  Nombre: {nombre}\n"
        f"  Usuario: {usuario}\n"
        f"  Correo: {correo}\n"
        f"  Fecha de registro: {fecha_registro}\n"
        + (f"  Telefono: {telefono}\n" if telefono else "")
        + "\nSi necesitas ayuda, contacta al desarrollador:\n"
        "  Correo: siriusplanet76@gmail.com\n"
        "  Telefono: +57 3153806797\n\n"
        "Atlas - Sistema de Gestion de Reciclaje Inteligente\n"
        "Universidad Autonoma de Tomina"
    )

    return html, texto_plano


# ==========================================
# INICIO
# ==========================================
@app.route("/")
def index():
    return redirect(url_for("login"))

# ==========================================
# LOGIN
# ==========================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        usuario = request.form.get("usuario", "").strip()
        contrasena = request.form.get("contrasena", "")

        if not usuario or not contrasena:
            flash("Debes ingresar tu usuario y contraseña.", "danger")
            return redirect(url_for("login"))

        # Validaciones básicas del nombre de usuario.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,50}", usuario):
            flash(
                "El usuario solo puede contener letras, números, "
                "puntos, guiones y debe tener entre 3 y 50 caracteres.",
                "danger"
            )
            return redirect(url_for("login"))

        # Límite de longitud de la contraseña para evitar abuso.
        if len(contrasena) > 255:
            flash("La contraseña no puede superar los 255 caracteres.", "danger")
            return redirect(url_for("login"))

        conexion = obtener_conexion()

        if conexion is None:
            flash("No fue posible conectar con la base de datos.", "danger")
            return redirect(url_for("login"))

        cursor = conexion.cursor(dictionary=True)

        try:
            # Consulta parametrizada: previene inyección SQL.
            # La contraseña nunca se compara en la base de datos:
            # se recupera su hash y se verifica con check_password_hash.
            consulta = """
                SELECT
                    id_usuario,
                    usuario,
                    contrasena,
                    numero_identificacion,
                    nombre_completo,
                    correo,
                    telefono,
                    tipo_usuario
                FROM usuarios
                WHERE usuario = %s
                LIMIT 1
            """

            cursor.execute(
                consulta,
                (usuario,)
            )

            registro = cursor.fetchone()

            if registro is None or not check_password_hash(
                registro["contrasena"],
                contrasena
            ):
                flash(
                    "Usuario o contraseña incorrectos.",
                    "danger"
                )
                return redirect(url_for("login"))

            # No guardamos el hash en la sesión.
            session["id_usuario"] = registro["id_usuario"]
            session["usuario"] = registro["usuario"]
            session["nombre_completo"] = registro["nombre_completo"]
            session["numero_identificacion"] = registro["numero_identificacion"]
            session["tipo_usuario"] = registro["tipo_usuario"]

            if registro["tipo_usuario"] == "ADMINISTRADOR":
                return redirect(url_for("admin_dashboard"))

            return redirect(url_for("dashboard"))

        except Exception as e:
            print("ERROR LOGIN:", e)
            flash("Ocurrió un error al iniciar sesión.", "danger")
            return redirect(url_for("login"))

        finally:
            cursor.close()
            conexion.close()

    return render_template("auth/login.html")

# ==========================================
# REGISTER
# ==========================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        nombre_completo = request.form.get("nombre_completo", "").strip()
        numero_identificacion = request.form.get("numero_identificacion", "").strip()
        correo = request.form.get("correo", "").strip()
        telefono = request.form.get("telefono", "").strip()
        usuario = request.form.get("usuario", "").strip()
        contrasena = request.form.get("contrasena", "")
        confirmar_contrasena = request.form.get("confirmar_contrasena", "")

        # Conservamos los datos ingresados para no borrar el formulario
        # cuando ocurre un error de validación.
        datos = {
            "nombre_completo": nombre_completo,
            "numero_identificacion": numero_identificacion,
            "correo": correo,
            "telefono": telefono,
            "usuario": usuario,
            "contrasena": contrasena,
            "confirmar_contrasena": confirmar_contrasena
        }

        def volver_al_formulario():
            return render_template(
                "auth/register.html",
                datos=datos
            )

        if not all([
            nombre_completo,
            numero_identificacion,
            correo,
            usuario,
            contrasena,
            confirmar_contrasena
        ]):
            flash("Completa todos los campos obligatorios.", "danger")
            return volver_al_formulario()

        if not numero_identificacion.isdigit():
            flash("La identificación debe contener solo números.", "danger")
            return volver_al_formulario()

        # Nombre de usuario seguro: solo letras, números, puntos, guiones.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,50}", usuario):
            flash(
                "El usuario solo puede contener letras, números, puntos, "
                "guiones y debe tener entre 3 y 50 caracteres.",
                "danger"
            )
            return volver_al_formulario()

        # Política mínima de contraseña segura.
        if not validar_contrasena(contrasena):
            flash(
                "La contraseña debe tener al menos 8 caracteres, una "
                "mayúscula, una minúscula y un número.",
                "danger"
            )
            return volver_al_formulario()

        if contrasena != confirmar_contrasena:
            flash("Las contraseñas no coinciden.", "danger")
            return volver_al_formulario()

        conexion = obtener_conexion()

        if conexion is None:
            flash("No fue posible conectar con la base de datos.", "danger")
            return volver_al_formulario()

        cursor = conexion.cursor()

        try:
            # Generar el hash de la contraseña: nunca se guarda en texto plano.
            hash_contrasena = generate_password_hash(contrasena)

            consulta = """
                INSERT INTO usuarios (
                    usuario,
                    contrasena,
                    numero_identificacion,
                    nombre_completo,
                    correo,
                    telefono,
                    tipo_usuario
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            valores = (
                usuario,
                hash_contrasena,
                numero_identificacion,
                nombre_completo,
                correo,
                telefono or None,
                "USUARIO"
            )

            cursor.execute(
                consulta,
                valores
            )

            conexion.commit()

            # --- Correo de bienvenida (HTML) ---
            try:
                fecha_registro = datetime.now().strftime("%d/%m/%Y %H:%M")
                html_bienvenida, texto_bienvenida = _plantilla_bienvenida(
                    nombre=nombre_completo,
                    usuario=usuario,
                    correo=correo,
                    fecha_registro=fecha_registro,
                    telefono=telefono
                )
                enviar_correo_html(
                    destinatario=correo,
                    asunto="!Bienvenido a Atlas! - Tu cuenta ha sido creada",
                    html_contenido=html_bienvenida,
                    texto_plano=texto_bienvenida
                )
            except Exception as e_email:
                print("ERROR ENVIANDO CORREO DE BIENVENIDA:", e_email)

            flash(
                "Cuenta creada correctamente. Ya puedes iniciar sesión.",
                "success"
            )

            return redirect(url_for("login"))

        except Exception as e:
            conexion.rollback()

            print("ERROR REGISTRO:", e)

            if "Duplicate entry" in str(e):
                flash(
                    "El usuario, la identificación o el correo "
                    "ya están registrados.",
                    "danger"
                )
            else:
                flash(
                    "No fue posible crear la cuenta.",
                    "danger"
                )

            return volver_al_formulario()

        finally:
            cursor.close()
            conexion.close()

    return render_template("auth/register.html")

# ==========================================
# VERIFICAR DATOS EXISTENTES (REGISTER)
# ==========================================
@app.route("/api/verificar-datos", methods=["POST"])
def api_verificar_datos():

    datos = request.get_json() or {}

    valor = datos.get("valor", "").strip().lower()
    campo = datos.get("campo", "").strip().lower()

    campos_validos = ["usuario", "correo", "numero_identificacion"]

    if campo not in campos_validos:
        return jsonify({
            "error": "Campo no válido."
        }), 400

    if not valor:
        return jsonify({
            "existe": False
        })

    conexion = obtener_conexion()

    if conexion is None:
        return jsonify({
            "error": "Error de conexión."
        }), 500

    cursor = conexion.cursor()

    try:
        columna = {
            "usuario": "usuario",
            "correo": "correo",
            "numero_identificacion": "numero_identificacion"
        }[campo]

        cursor.execute(
            f"SELECT id_usuario FROM usuarios WHERE LOWER({columna}) = %s LIMIT 1",
            (valor,)
        )

        existe = cursor.fetchone() is not None

        return jsonify({
            "existe": existe
        })

    except Exception as e:
        print("ERROR VERIFICAR DATOS:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conexion.close()

# ==========================================
# RECUPERAR NOMBRE DE USUARIO
# ==========================================
@app.route("/recuperar-usuario", methods=["GET", "POST"])
def recuperar_usuario():

    if request.method == "POST":
        correo = request.form.get("correo", "").strip().lower()

        if not correo or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", correo):
            flash(
                "Ingresa un correo electrónico válido.",
                "danger"
            )
            return redirect(url_for("recuperar_usuario"))

        conexion = obtener_conexion()

        if conexion is None:
            flash(
                "No fue posible conectar con la base de datos.",
                "danger"
            )
            return redirect(url_for("recuperar_usuario"))

        cursor = conexion.cursor(dictionary=True)

        try:
            cursor.execute(
                """
                SELECT usuario, nombre_completo
                FROM usuarios
                WHERE LOWER(correo) = %s
                LIMIT 1
                """,
                (correo,)
            )

            registro = cursor.fetchone()

            if registro is None:
                flash(
                    "No hay ningún usuario registrado con ese correo.",
                    "danger"
                )
                return redirect(url_for("recuperar_usuario"))

            nombre_usuario = registro["usuario"]
            nombre_completo = registro["nombre_completo"]

            asunto = "Tu nombre de usuario - Atlas"
            texto = (
                "Hola " + nombre_completo + ",\n\n"
                "Somos Atlas. Aquí tienes tu nombre de usuario:\n\n"
                "   Usuario: " + nombre_usuario + "\n\n"
                "Puedes usarlo para iniciar sesión en el sistema.\n\n"
                "Si tienes alguna duda, contáctanos.\n"
                "Atlas - Recicla. Suma. Transforma."
            )

            if not enviar_correo(correo, asunto, texto):
                flash(
                    "Ocurrió un error al enviar el correo. "
                    "Inténtalo de nuevo más tarde.",
                    "danger"
                )
                return redirect(url_for("recuperar_usuario"))

            flash(
                "Te hemos enviado tu nombre de usuario al correo "
                + correo + ".",
                "success"
            )
            return redirect(url_for("login"))

        except Exception as e:
            print("ERROR RECUPERAR USUARIO:", e)
            flash(
                "Ocurrió un error al procesar la solicitud.",
                "danger"
            )
            return redirect(url_for("recuperar_usuario"))

        finally:
            cursor.close()
            conexion.close()

    return render_template("auth/recuperar_usuario.html")


# ==========================================
# RESTABLECER CONTRASEÑA
# ==========================================
@app.route("/restablecer-contrasena", methods=["GET", "POST"])
def restablecer_contrasena():

    if request.method == "POST":
        correo = request.form.get("correo", "").strip().lower()

        if not correo or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", correo):
            flash(
                "Ingresa un correo electrónico válido.",
                "danger"
            )
            return redirect(url_for("restablecer_contrasena"))

        conexion = obtener_conexion()

        if conexion is None:
            flash(
                "No fue posible conectar con la base de datos.",
                "danger"
            )
            return redirect(url_for("restablecer_contrasena"))

        cursor = conexion.cursor(dictionary=True)

        try:
            cursor.execute(
                """
                SELECT id_usuario, nombre_completo
                FROM usuarios
                WHERE LOWER(correo) = %s
                LIMIT 1
                """,
                (correo,)
            )

            registro = cursor.fetchone()

            if registro is None:
                flash(
                    "No hay ningún usuario registrado con ese correo.",
                    "danger"
                )
                return redirect(url_for("restablecer_contrasena"))

            nombre_completo = registro["nombre_completo"]
            id_usuario = registro["id_usuario"]

            # No se puede recuperar la contraseña original porque solo se
            # guarda su hash. Se genera una contraseña temporal nueva.
            nueva_contrasena = generar_contrasena_temporal()
            nuevo_hash = generate_password_hash(nueva_contrasena)

            cursor.execute(
                """
                UPDATE usuarios
                SET contrasena = %s
                WHERE id_usuario = %s
                """,
                (nuevo_hash, id_usuario)
            )

            conexion.commit()

            asunto = "Tu nueva contraseña de Atlas"
            texto = (
                "Hola " + nombre_completo + ",\n\n"
                "Somos Atlas. Has solicitado restablecer tu contraseña.\n"
                "Debido a que tus datos están protegidos, tu contraseña "
                "anterior no puede recuperarse, por eso generamos una nueva:\n\n"
                "   Contraseña: " + nueva_contrasena + "\n\n"
                "Te recomendamos iniciar sesión y cambiarla por una "
                "contraseña personal desde tu panel de usuario.\n\n"
                "Atlas - Recicla. Suma. Transforma."
            )

            if not enviar_correo(correo, asunto, texto):
                flash(
                    "Ocurrió un error al enviar el correo. "
                    "Inténtalo de nuevo más tarde.",
                    "danger"
                )
                return redirect(url_for("restablecer_contrasena"))

            flash(
                "Te hemos enviado tu contraseña al correo "
                + correo + ".",
                "success"
            )
            return redirect(url_for("login"))

        except Exception as e:
            print("ERROR RESTABLECER CONTRASEÑA:", e)
            flash(
                "Ocurrió un error al procesar la solicitud.",
                "danger"
            )
            return redirect(url_for("restablecer_contrasena"))

        finally:
            cursor.close()
            conexion.close()

    return render_template("auth/restablecer_contrasena.html")


# ==========================================
# CAMBIAR CONTRASEÑA (DESDE EL PANEL)
# ==========================================
@app.route("/cambiar-contrasena", methods=["POST"])
def cambiar_contrasena():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    contrasena_actual = request.form.get("contrasena_actual", "")
    nueva_contrasena = request.form.get("nueva_contrasena", "")
    confirmar_contrasena = request.form.get("confirmar_nueva_contrasena", "")

    if not contrasena_actual or not nueva_contrasena or not confirmar_contrasena:
        flash(
            "Completa todos los campos para cambiar tu contraseña.",
            "danger"
        )
        return redirect(url_for("dashboard"))

    if not validar_contrasena(nueva_contrasena):
        flash(
            "La nueva contraseña debe tener al menos 8 caracteres, una "
            "mayúscula, una minúscula y un número.",
            "danger"
        )
        return redirect(url_for("dashboard"))

    if nueva_contrasena != confirmar_contrasena:
        flash(
            "Las contraseñas nuevas no coinciden.",
            "danger"
        )
        return redirect(url_for("dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(url_for("dashboard"))

    cursor = conexion.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT contrasena
            FROM usuarios
            WHERE id_usuario = %s
            LIMIT 1
            """,
            (session["id_usuario"],)
        )

        registro = cursor.fetchone()

        if registro is None:
            flash(
                "El usuario no existe.",
                "danger"
            )
            return redirect(url_for("logout"))

        if not check_password_hash(
            registro["contrasena"],
            contrasena_actual
        ):
            flash(
                "La contraseña actual es incorrecta.",
                "danger"
            )
            return redirect(url_for("dashboard"))

        nuevo_hash = generate_password_hash(nueva_contrasena)

        cursor.execute(
            """
            UPDATE usuarios
            SET contrasena = %s
            WHERE id_usuario = %s
            """,
            (nuevo_hash, session["id_usuario"])
        )

        conexion.commit()

        flash(
            "Contraseña actualizada correctamente.",
            "success"
        )
        return redirect(url_for("dashboard"))

    except Exception as e:
        conexion.rollback()
        print("ERROR CAMBIAR CONTRASEÑA:", e)
        flash(
            "No fue posible actualizar la contraseña.",
            "danger"
        )
        return redirect(url_for("dashboard"))

    finally:
        cursor.close()
        conexion.close()


# ==========================================
# DASHBOARD USUARIO
# ==========================================
@app.route("/dashboard")
def dashboard():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") == "ADMINISTRADOR":
        return redirect(url_for("admin_dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(url_for("login"))

    cursor = conexion.cursor(dictionary=True)

    try:
        consulta = """
            SELECT
                u.id_usuario,
                u.nombre_completo,
                u.numero_identificacion,

                COALESCE(
                    (
                        SELECT SUM(mp1.puntos)
                        FROM movimientos_puntos mp1
                        WHERE mp1.id_usuario = u.id_usuario
                    ),
                    0
                ) AS puntos_totales,

                COALESCE(
                    (
                        SELECT COUNT(*)
                        FROM movimientos_puntos mp2
                        WHERE mp2.id_usuario = u.id_usuario
                        AND mp2.tipo_movimiento = 'RECICLAJE'
                    ),
                    0
                ) AS botellas_recicladas,

                COALESCE(
                    (
                        SELECT SUM(mp3.puntos)
                        FROM movimientos_puntos mp3
                        WHERE mp3.id_usuario = u.id_usuario
                        AND mp3.tipo_movimiento = 'RECICLAJE'
                    ),
                    0
                ) AS puntos_obtenidos,

                (
                    SELECT MAX(mp4.fecha_movimiento)
                    FROM movimientos_puntos mp4
                    WHERE mp4.id_usuario = u.id_usuario
                    AND mp4.tipo_movimiento = 'RECICLAJE'
                ) AS ultimo_reciclaje

            FROM usuarios u
            WHERE u.id_usuario = %s
        """

        cursor.execute(
            consulta,
            (session["id_usuario"],)
        )

        usuario = cursor.fetchone()

        if usuario is None:
            session.clear()

            flash(
                "El usuario no existe.",
                "danger"
            )

            return redirect(url_for("login"))

        return render_template(
            "usuario/dashboard.html",
            usuario=usuario
        )

    except Exception as e:
        print("ERROR DASHBOARD:", e)

        flash(
            f"Error al cargar el dashboard: {e}",
            "danger"
        )

        return redirect(url_for("login"))

    finally:
        cursor.close()
        conexion.close()

# ==========================================
# INFORMACIÓN Atlas
# ==========================================
@app.route("/informacion")
def informacion():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") == "ADMINISTRADOR":
        return redirect(url_for("admin_dashboard"))

    return render_template(
        "usuario/informacion.html"
    )


@app.route("/escanear")
def escanear():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") == "ADMINISTRADOR":
        return redirect(url_for("admin_dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )

        return redirect(url_for("dashboard"))

    cursor = conexion.cursor(dictionary=True)

    try:

        cursor.execute(
            """
            SELECT *
            FROM usuarios
            WHERE id_usuario = %s
            """,
            (session["id_usuario"],)
        )

        usuario = cursor.fetchone()

        if usuario is None:
            return redirect(url_for("logout"))

        return render_template(
            "usuario/escanear.html",
            usuario=usuario
        )

    finally:

        cursor.close()
        conexion.close()



# API DE ESCANEO Y REGISTRO DEL RECICLAJE

@app.route("/api/escanear", methods=["POST"])
def api_escanear():

    if "id_usuario" not in session:
        return jsonify({
            "error": "Sesión no válida."
        }), 401

    if "imagen" not in request.files:
        return jsonify({
            "error": "No se recibió ninguna imagen."
        }), 400

    if model is None:
        return jsonify({
            "error":
                "El modelo de IA no está disponible. "
                "Error al cargar: " +
                (model_error or "desconocido")
        }), 503

    archivo = request.files["imagen"]

    if archivo.filename == "":
        return jsonify({
            "error": "La imagen está vacía."
        }), 400

    try:

        # =====================================================
        # LEER IMAGEN
        # =====================================================

        datos = archivo.read()

        print("TAMAÑO IMAGEN RECIBIDA:", len(datos), "bytes")

        imagen_array = np.frombuffer(
            datos,
            np.uint8
        )

        frame = cv2.imdecode(
            imagen_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            print("ERROR: cv2.imdecode devolvió None")
            return jsonify({
                "error": "No fue posible procesar la imagen."
            }), 400

        print(
            "IMAGEN DECODED:",
            frame.shape[1], "x", frame.shape[0]
        )


        # =====================================================
        # REDIMENSIONAR PARA ACELERAR YOLO
        # =====================================================

        h_img, w_img = frame.shape[:2]
        max_dim = 640

        if max(h_img, w_img) > max_dim:
            scale = max_dim / max(h_img, w_img)
            frame = cv2.resize(
                frame,
                (int(w_img * scale), int(h_img * scale)),
                interpolation=cv2.INTER_AREA
            )


        # =====================================================
        # ANALIZAR CON YOLO
        # =====================================================

        try:
            resultados = model(
                frame,
                conf=0.15,
                imgsz=320,
                device="cpu",
                max_det=10,
                verbose=False
            )
        except Exception as yolo_err:
            print("ERROR YOLO INFERENCE:", yolo_err)
            return jsonify({
                "error":
                    "Error del modelo de IA: " +
                    str(yolo_err)
            }), 500

        resultado = resultados[0]

        clases_detectadas = []

        confianza_maxima = 0.0

        for box in resultado.boxes:

            clase_id = int(box.cls[0])

            confianza = float(box.conf[0])

            nombre_clase = (
                resultado.names[clase_id]
                .lower()
            )

            clases_detectadas.append({
                "clase": nombre_clase,
                "confianza": round(
                    confianza,
                    4
                )
            })

            if confianza > confianza_maxima:
                confianza_maxima = confianza


        print("OBJETOS DETECTADOS:")
        print(clases_detectadas)


        # =====================================================
        # DETERMINAR OBJETOS
        # =====================================================

        clases = [
            item["clase"]
            for item in clases_detectadas
        ]

        botella_detectada = (
            "botella" in clases
        )

        tapa_detectada = (
            "tapa" in clases
        )

        etiqueta_detectada = (
            "etiqueta" in clases
        )


        # =====================================================
        # CALCULAR PUNTOS
        # =====================================================

        puntos_base = 0
        puntos_tapa = 0
        puntos_etiqueta = 0
        puntos_totales = 0

        if botella_detectada:

            puntos_base = 50

            if tapa_detectada:
                puntos_tapa = 10

            if etiqueta_detectada:
                puntos_etiqueta = 5

            puntos_totales = (
                puntos_base
                + puntos_tapa
                + puntos_etiqueta
            )


        # =====================================================
        # DIBUJAR RESULTADO YOLO
        # =====================================================

        annotated_frame = resultado.plot()


        # =====================================================
        # CONVERTIR IMAGEN A BASE64
        # =====================================================

        _, buffer = cv2.imencode(
            ".jpg",
            annotated_frame
        )

        imagen_base64 = base64.b64encode(
            buffer
        ).decode("utf-8")


        # =====================================================
        # SI NO HAY BOTELLA
        # =====================================================

        if not botella_detectada:

            return jsonify({

                "success": False,

                "botella_detectada": False,

                "titulo":
                    "Botella no reconocida",

                "mensaje":
                    "No se detectó una botella válida.",

                "puntos": 0,

                "detecciones":
                    clases_detectadas,

                "imagen":
                    imagen_base64

            })


        # =====================================================
        # BOTELLA DETECTADA
        #
        # IMPORTANTE:
        # AQUÍ TODAVÍA NO SE GUARDA NADA EN MYSQL.
        # =====================================================

        return jsonify({

            "success": True,

            "botella_detectada": True,

            "titulo":
                "Botella reconocida",

            "mensaje":
                "La botella fue reconocida correctamente.",

            "puntos":
                puntos_totales,

            "puntos_base":
                puntos_base,

            "puntos_tapa":
                puntos_tapa,

            "puntos_etiqueta":
                puntos_etiqueta,

            "confianza":
                confianza_maxima * 100,

            "tapa_detectada":
                tapa_detectada,

            "etiqueta_detectada":
                etiqueta_detectada,

            "detecciones":
                clases_detectadas,

            "imagen":
                imagen_base64

        })


    except Exception as e:

        print(
            "ERROR ANALIZANDO:",
            e
        )
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500



@app.route("/api/registrar-reciclaje", methods=["POST"])
def registrar_reciclaje():

    if "id_usuario" not in session:

        return jsonify({
            "error": "Sesión no válida."
        }), 401


    datos = request.get_json()

    if not datos:

        return jsonify({
            "error": "No se recibieron datos."
        }), 400


    # =====================================================
    # RECIBIR DATOS DEL ANÁLISIS
    # =====================================================

    botella_detectada = datos.get(
        "botella_detectada",
        False
    )

    tapa_detectada = datos.get(
        "tapa_detectada",
        False
    )

    etiqueta_detectada = datos.get(
        "etiqueta_detectada",
        False
    )

    confianza = datos.get(
        "confianza",
        0
    )

    puntos_base = datos.get(
        "puntos_base",
        0
    )

    puntos_tapa = datos.get(
        "puntos_tapa",
        0
    )

    puntos_etiqueta = datos.get(
        "puntos_etiqueta",
        0
    )

    puntos_totales = datos.get(
        "puntos",
        0
    )


    # =====================================================
    # SEGURIDAD
    # =====================================================

    if not botella_detectada:

        return jsonify({
            "error":
                "No se puede registrar un reciclaje "
                "sin una botella detectada."
        }), 400


    conexion = obtener_conexion()

    if conexion is None:

        return jsonify({
            "error":
                "No fue posible conectar con "
                "la base de datos."
        }), 500


    cursor = conexion.cursor()


    try:

        # =================================================
        # BUSCAR BOTELLA PET
        # =================================================

        cursor.execute(
            """
            SELECT id_botella
            FROM botellas
            WHERE nombre = %s
            LIMIT 1
            """,
            ("Botella PET",)
        )

        botella = cursor.fetchone()

        id_botella = None

        if botella:
            id_botella = botella[0]


        # =================================================
        # INSERTAR ANÁLISIS IA
        # =================================================

        consulta_analisis = """
            INSERT INTO analisis_ia (
                id_usuario,
                id_botella,
                imagen,
                botella_detectada,
                tapa_detectada,
                etiqueta_detectada,
                confianza,
                puntos_base,
                puntos_tapa,
                puntos_etiqueta,
                puntos_totales,
                estado_analisis,
                modelo_ia,
                version_modelo
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
        """


        valores_analisis = (

            session["id_usuario"],

            id_botella,

            None,

            1 if botella_detectada else 0,

            1 if tapa_detectada else 0,

            1 if etiqueta_detectada else 0,

            confianza,

            puntos_base,

            puntos_tapa,

            puntos_etiqueta,

            puntos_totales,

            "IDENTIFICADO",

            "YOLO",

            "train-5"

        )


        cursor.execute(
            consulta_analisis,
            valores_analisis
        )


        id_analisis = cursor.lastrowid


        # =================================================
        # OBTENER SALDO ACTUAL
        # =================================================

        cursor.execute(
            """
            SELECT COALESCE(
                SUM(puntos),
                0
            )
            FROM movimientos_puntos
            WHERE id_usuario = %s
            """,
            (
                session["id_usuario"],
            )
        )


        resultado_saldo = (
            cursor.fetchone()
        )


        saldo_anterior = int(
            resultado_saldo[0] or 0
        )


        saldo_nuevo = (
            saldo_anterior
            + puntos_totales
        )


        # =================================================
        # MOVIMIENTO DE PUNTOS
        # =================================================

        motivo = (
            "Reciclaje detectado por IA: "
            "botella"
        )


        if tapa_detectada:

            motivo += " + tapa"


        if etiqueta_detectada:

            motivo += " + etiqueta"


        consulta_movimiento = """
            INSERT INTO movimientos_puntos (
                id_usuario,
                id_analisis,
                tipo_movimiento,
                puntos,
                motivo
            )
            VALUES (
                %s,
                %s,
                'RECICLAJE',
                %s,
                %s
            )
        """


        cursor.execute(
            consulta_movimiento,
            (
                session["id_usuario"],
                id_analisis,
                puntos_totales,
                motivo
            )
        )


        # =================================================
        # GENERAR COMPROBANTE
        # =================================================

        numero_comprobante = (
            "ATLA-"
            + str(id_analisis).zfill(6)
        )


        # =================================================
        # INSERTAR COMPROBANTE
        # =================================================

        consulta_comprobante = """
            INSERT INTO comprobantes_reciclaje (
                id_analisis,
                id_usuario,
                numero_comprobante,
                saldo_anterior,
                puntos_ganados,
                saldo_nuevo
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
        """


        cursor.execute(
            consulta_comprobante,
            (
                id_analisis,
                session["id_usuario"],
                numero_comprobante,
                saldo_anterior,
                puntos_totales,
                saldo_nuevo
            )
        )


        # =================================================
        # CONFIRMAR
        # =================================================

        conexion.commit()


        # =================================================
        # RESPUESTA
        # =================================================

        return jsonify({

            "success": True,

            "mensaje":
                "Botella registrada correctamente "
                "y puntos asignados.",

            "puntos":
                puntos_totales,

            "saldo_anterior":
                saldo_anterior,

            "saldo_nuevo":
                saldo_nuevo,

            "numero_comprobante":
                numero_comprobante

        })


    except Exception as e:

        conexion.rollback()

        print(
            "ERROR REGISTRANDO RECICLAJE:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


    finally:

        cursor.close()

        conexion.close()



@app.route("/catalogo")
def catalogo():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") == "ADMINISTRADOR":
        return redirect(url_for("admin_dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(url_for("dashboard"))

    cursor = conexion.cursor(dictionary=True)

    try:

        cursor.execute(
            """
            SELECT *
            FROM usuarios
            WHERE id_usuario = %s
            """,
            (session["id_usuario"],)
        )

        usuario = cursor.fetchone()

        if usuario is None:
            flash(
                "No se encontró la información del usuario.",
                "danger"
            )
            return redirect(url_for("dashboard"))

        return render_template(
            "usuario/catalogo.html",
            usuario=usuario
        )

    except Exception as e:

        print("ERROR CATÁLOGO:", e)

        flash(
            f"No fue posible cargar el catálogo: {e}",
            "danger"
        )

        return redirect(url_for("dashboard"))

    finally:

        cursor.close()
        conexion.close()


# ==========================================
# DASHBOARD ADMINISTRADOR
# ==========================================
@app.route("/admin")
def admin_dashboard():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(url_for("login"))

    cursor = conexion.cursor(dictionary=True)

    try:
        consulta = """
            SELECT
                (SELECT COUNT(*) FROM usuarios) AS usuarios_total,

                (SELECT COUNT(*) FROM analisis_ia) AS escaneos_ia,

                (
                    SELECT COALESCE(SUM(puntos), 0)
                    FROM movimientos_puntos
                    WHERE tipo_movimiento = 'RECICLAJE'
                ) AS puntos_entregados,

                (SELECT COUNT(*) FROM productos) AS productos_total
        """

        cursor.execute(consulta)

        estadisticas = cursor.fetchone()

        return render_template(
            "admin/dashboard.html",
            estadisticas=estadisticas
        )

    except Exception as e:
        print(
            "ERROR DASHBOARD ADMIN:",
            e
        )

        flash(
            f"Error al cargar el dashboard: {e}",
            "danger"
        )

        return redirect(
            url_for("dashboard")
        )

    finally:
        cursor.close()
        conexion.close()

# ==========================================
# GESTIÓN DE USUARIOS
# ==========================================
@app.route("/admin/usuarios")
def admin_usuarios():
    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash("No fue posible conectar con la base de datos.", "danger")
        return redirect(url_for("admin_dashboard"))

    cursor = conexion.cursor(dictionary=True)

    try:
        filtro_usuario = request.args.get("usuario", "").strip()
        filtro_identificacion = request.args.get("identificacion", "").strip()

        condiciones = []
        parametros = []

        if filtro_usuario:
            condiciones.append("u.nombre_completo LIKE %s")
            parametros.append(f"%{filtro_usuario}%")

        if filtro_identificacion:
            condiciones.append("u.numero_identificacion = %s")
            parametros.append(filtro_identificacion)

        consulta = """
            SELECT
                u.id_usuario,
                u.usuario,
                u.nombre_completo,
                u.numero_identificacion,
                u.correo,
                u.telefono,
                u.tipo_usuario,

                COALESCE(
                    (
                        SELECT SUM(mp.puntos)
                        FROM movimientos_puntos mp
                        WHERE mp.id_usuario = u.id_usuario
                    ),
                    0
                ) AS puntos_totales,

                (
                    SELECT COUNT(*)
                    FROM analisis_ia ai
                    WHERE ai.id_usuario = u.id_usuario
                ) AS escaneos_ia

            FROM usuarios u
        """

        if condiciones:
            consulta += " WHERE " + " AND ".join(condiciones)

        consulta += " ORDER BY u.id_usuario DESC"

        cursor.execute(consulta, parametros)
        usuarios = cursor.fetchall()

        return render_template(
            "admin/usuarios.html",
            usuarios=usuarios,
            filtro_usuario=filtro_usuario,
            filtro_identificacion=filtro_identificacion
        )

    except Exception as e:
        print("ERROR GESTION USUARIOS:", e)
        flash(f"Error al cargar los usuarios: {e}", "danger")
        return redirect(url_for("admin_dashboard"))

    finally:
        cursor.close()
        conexion.close()

# ==========================================
# EDITAR USUARIO
# ==========================================
@app.route(
    "/admin/usuarios/editar/<int:id_usuario>",
    methods=["GET", "POST"]
)
def admin_editar_usuario(id_usuario):

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(
            url_for("admin_usuarios")
        )

    cursor = conexion.cursor(dictionary=True)

    try:

        if request.method == "GET":

            consulta = """
                SELECT
                    id_usuario,
                    usuario,
                    numero_identificacion,
                    nombre_completo,
                    correo,
                    telefono,
                    tipo_usuario
                FROM usuarios
                WHERE id_usuario = %s
            """

            cursor.execute(
                consulta,
                (id_usuario,)
            )

            usuario = cursor.fetchone()

            if usuario is None:

                flash(
                    "El usuario no existe.",
                    "danger"
                )

                return redirect(
                    url_for("admin_usuarios")
                )

            return render_template(
                "admin/usuarios.html",
                usuarios=[usuario],
                editar=True,
                usuario_editar=usuario,
                busqueda=""
            )

        # ==================================
        # DATOS DEL FORMULARIO
        # ==================================

        nombre_completo = request.form.get(
            "nombre_completo",
            ""
        ).strip()

        numero_identificacion = request.form.get(
            "numero_identificacion",
            ""
        ).strip()

        correo = request.form.get(
            "correo",
            ""
        ).strip()

        telefono = request.form.get(
            "telefono",
            ""
        ).strip()

        usuario = request.form.get(
            "usuario",
            ""
        ).strip()

        contrasena = request.form.get(
            "contrasena",
            ""
        )

        tipo_usuario = request.form.get(
            "tipo_usuario",
            "USUARIO"
        ).strip()

        # ==================================
        # VALIDACIONES
        # ==================================

        if not all([
            nombre_completo,
            numero_identificacion,
            correo,
            usuario
        ]):

            flash(
                "Completa todos los campos obligatorios.",
                "danger"
            )

            return redirect(
                url_for(
                    "admin_editar_usuario",
                    id_usuario=id_usuario
                )
            )

        if not numero_identificacion.isdigit():
            flash(
                "La identificación debe contener solo números.",
                "danger"
            )
            return redirect(url_for("admin_usuarios"))

        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,50}", usuario):

            flash(
                "El usuario solo puede contener letras, números, puntos, "
                "guiones y debe tener entre 3 y 50 caracteres.",
                "danger"
            )

            return redirect(
                url_for(
                    "admin_editar_usuario",
                    id_usuario=id_usuario
                )
            )

        # Si se proporciona una nueva contraseña, debe cumplir la política.
        if contrasena and not validar_contrasena(contrasena):

            flash(
                "La contraseña debe tener al menos 8 caracteres, una "
                "mayúscula, una minúscula y un número.",
                "danger"
            )

            return redirect(
                url_for(
                    "admin_editar_usuario",
                    id_usuario=id_usuario
                )
            )

        tipos_validos = [
            "USUARIO",
            "ADMINISTRADOR"
        ]

        if tipo_usuario not in tipos_validos:

            flash(
                "El tipo de usuario seleccionado no es válido.",
                "danger"
            )

            return redirect(
                url_for(
                    "admin_editar_usuario",
                    id_usuario=id_usuario
                )
            )

        # ==================================
        # EVITAR QUE EL ADMIN SE ELIMINE
        # SU PROPIO ACCESO
        # ==================================

        if (
            id_usuario == session["id_usuario"]
            and tipo_usuario != "ADMINISTRADOR"
        ):

            flash(
                "No puedes quitarte a ti mismo el acceso de administrador.",
                "danger"
            )

            return redirect(
                url_for(
                    "admin_editar_usuario",
                    id_usuario=id_usuario
                )
            )

        # ==================================
        # ACTUALIZAR
        # ==================================

        # Si se indica una nueva contraseña, se actualiza su hash.
        if contrasena:
            hash_contrasena = generate_password_hash(contrasena)

            consulta = """
                UPDATE usuarios
                SET
                    usuario = %s,
                    contrasena = %s,
                    numero_identificacion = %s,
                    nombre_completo = %s,
                    correo = %s,
                    telefono = %s,
                    tipo_usuario = %s
                WHERE id_usuario = %s
            """

            valores = (
                usuario,
                hash_contrasena,
                numero_identificacion,
                nombre_completo,
                correo,
                telefono,
                tipo_usuario,
                id_usuario
            )

        else:

            consulta = """
                UPDATE usuarios
                SET
                    usuario = %s,
                    numero_identificacion = %s,
                    nombre_completo = %s,
                    correo = %s,
                    telefono = %s,
                    tipo_usuario = %s
                WHERE id_usuario = %s
            """

            valores = (
                usuario,
                numero_identificacion,
                nombre_completo,
                correo,
                telefono,
                tipo_usuario,
                id_usuario
            )

        cursor.execute(
            consulta,
            valores
        )

        # ==================================
        # REGISTRAR AUDITORÍA
        # ==================================

        descripcion_auditoria = (
            f"Se modificaron los datos del usuario "
            f"{nombre_completo} "
            f"(ID: {id_usuario})."
        )

        cursor.execute(
            """
            INSERT INTO auditoria
            (
                id_usuario,
                accion,
                tabla_afectada,
                id_registro,
                descripcion,
                ip
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                session.get("id_usuario"),
                "ACTUALIZAR",
                "usuarios",
                id_usuario,
                descripcion_auditoria,
                request.remote_addr
            )
        )

        conexion.commit()

        # Si el administrador editó su propia información,
        # actualizamos también la sesión.
        if id_usuario == session["id_usuario"]:

            session["usuario"] = usuario
            session["nombre_completo"] = nombre_completo
            session["numero_identificacion"] = numero_identificacion
            session["tipo_usuario"] = tipo_usuario

        flash(
            "Usuario actualizado correctamente.",
            "success"
        )

        return redirect(
            url_for("admin_usuarios")
        )

    except Exception as e:

        conexion.rollback()

        print(
            "ERROR EDITANDO USUARIO:",
            e
        )

        if "Duplicate entry" in str(e):

            flash(
                "El usuario, la identificación o el correo "
                "ya pertenecen a otro usuario.",
                "danger"
            )

        else:

            flash(
                f"No fue posible actualizar el usuario: {e}",
                "danger"
            )

        return redirect(
            url_for("admin_usuarios")
        )

    finally:
        cursor.close()
        conexion.close()

# ==========================================
# ELIMINAR USUARIO
# ==========================================
@app.route(
    "/admin/usuarios/eliminar/<int:id_usuario>",
    methods=["POST"]
)
def admin_eliminar_usuario(id_usuario):

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    # ==================================
    # NO PERMITIR ELIMINARSE A SÍ MISMO
    # ==================================

    if id_usuario == session["id_usuario"]:

        flash(
            "No puedes eliminar tu propia cuenta de administrador.",
            "danger"
        )

        return redirect(
            url_for("admin_usuarios")
        )

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(
            url_for("admin_usuarios")
        )

    cursor = conexion.cursor()

    try:

        consulta = """
            DELETE FROM usuarios
            WHERE id_usuario = %s
        """

        cursor.execute(
            consulta,
            (id_usuario,)
        )

        if cursor.rowcount == 0:

            flash(
                "El usuario no existe.",
                "danger"
            )

            return redirect(
                url_for("admin_usuarios")
            )

        conexion.commit()

        flash(
            "Usuario eliminado correctamente.",
            "success"
        )

        return redirect(
            url_for("admin_usuarios")
        )

    except Exception as e:

        conexion.rollback()

        print(
            "ERROR ELIMINANDO USUARIO:",
            e
        )

        flash(
            "No se pudo eliminar el usuario. "
            "Es posible que tenga registros relacionados "
            "en otras tablas.",
            "danger"
        )

        return redirect(
            url_for("admin_usuarios")
        )

    finally:
        cursor.close()
        conexion.close()


# ==========================================
# AGREGAR PUNTOS
# ==========================================
@app.route("/admin/usuarios/puntos/agregar/<int:id_usuario>", methods=["POST"])
def agregar_puntos(id_usuario):
    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    try:
        puntos = int(request.form.get("puntos", 0))
    except (ValueError, TypeError):
        puntos = 0

    motivo = request.form.get("motivo", "").strip()

    if puntos <= 0:
        flash("La cantidad de puntos debe ser mayor que 0.", "danger")
        return redirect(url_for("admin_usuarios"))

    conexion = obtener_conexion()

    if conexion is None:
        flash("No fue posible conectar con la base de datos.", "danger")
        return redirect(url_for("admin_usuarios"))

    cursor = conexion.cursor()

    try:
        consulta = """
            INSERT INTO movimientos_puntos (
                id_usuario,
                tipo_movimiento,
                puntos,
                motivo,
                id_usuario_admin
            )
            VALUES (%s, %s, %s, %s, %s)
        """

        valores = (
            id_usuario,
            "BONIFICACION",
            puntos,
            motivo if motivo else "Bonificación manual realizada por administrador",
            session["id_usuario"]
        )

        cursor.execute(consulta, valores)
        conexion.commit()

        flash(
            f"Se agregaron {puntos} AtlasPuntos correctamente.",
            "success"
        )

    except Exception as e:
        conexion.rollback()
        print("ERROR AGREGANDO PUNTOS:", e)
        flash(f"No fue posible agregar los puntos: {e}", "danger")

    finally:
        cursor.close()
        conexion.close()

    return redirect(url_for("admin_usuarios"))


# ==========================================
# QUITAR PUNTOS
# ==========================================
@app.route("/admin/usuarios/puntos/quitar/<int:id_usuario>", methods=["POST"])
def quitar_puntos(id_usuario):
    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    try:
        puntos = int(request.form.get("puntos", 0))
    except (ValueError, TypeError):
        puntos = 0

    motivo = request.form.get("motivo", "").strip()

    if puntos <= 0:
        flash("La cantidad de puntos debe ser mayor que 0.", "danger")
        return redirect(url_for("admin_usuarios"))

    conexion = obtener_conexion()

    if conexion is None:
        flash("No fue posible conectar con la base de datos.", "danger")
        return redirect(url_for("admin_usuarios"))

    cursor = conexion.cursor(dictionary=True)

    try:
        consulta_saldo = """
            SELECT COALESCE(SUM(puntos), 0) AS saldo
            FROM movimientos_puntos
            WHERE id_usuario = %s
        """

        cursor.execute(consulta_saldo, (id_usuario,))
        resultado = cursor.fetchone()

        saldo_actual = resultado["saldo"]

        if puntos > saldo_actual:
            flash(
                f"El usuario solamente tiene {saldo_actual} AtlasPuntos.",
                "danger"
            )
            return redirect(url_for("admin_usuarios"))

        consulta = """
            INSERT INTO movimientos_puntos (
                id_usuario,
                tipo_movimiento,
                puntos,
                motivo,
                id_usuario_admin
            )
            VALUES (%s, %s, %s, %s, %s)
        """

        valores = (
            id_usuario,
            "PENALIZACION",
            -puntos,
            motivo if motivo else "Descuento manual realizado por administrador",
            session["id_usuario"]
        )

        cursor.execute(consulta, valores)
        conexion.commit()

        flash(
            f"Se quitaron {puntos} AtlasPuntos correctamente.",
            "success"
        )

    except Exception as e:
        conexion.rollback()
        print("ERROR QUITANDO PUNTOS:", e)
        flash(f"No fue posible quitar los puntos: {e}", "danger")

    finally:
        cursor.close()
        conexion.close()

    return redirect(url_for("admin_usuarios"))


# ==========================================
# LOGOUT
# ==========================================
@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )



@app.route("/admin/reciclajes")
def admin_reciclajes():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(url_for("admin_dashboard"))

    cursor = conexion.cursor(dictionary=True)

    try:

        # ==========================================
        # PARÁMETROS (LÍMITE Y RANGO DE FECHAS)
        # ==========================================

        limite_max = 10
        limite_param = request.args.get("limite", "").strip().lower()
        ver_todos = limite_param in ("all", "todos", "0", "-1", "true")

        fecha_desde = request.args.get("desde", "").strip()
        fecha_hasta = request.args.get("hasta", "").strip()

        condiciones = []
        parametros = []

        if fecha_desde:
            condiciones.append("DATE(a.fecha_analisis) >= %s")
            parametros.append(fecha_desde)

        if fecha_hasta:
            condiciones.append("DATE(a.fecha_analisis) <= %s")
            parametros.append(fecha_hasta)

        where_sql = (" WHERE " + " AND ".join(condiciones)) if condiciones else ""

        # ==========================================
        # ANÁLISIS IA REGISTRADOS
        # ==========================================

        cursor.execute(
            """
            SELECT
                a.id_analisis,
                a.id_usuario,
                a.botella_detectada,
                a.tapa_detectada,
                a.etiqueta_detectada,
                a.confianza,
                a.puntos_base,
                a.puntos_tapa,
                a.puntos_etiqueta,
                a.puntos_totales,
                a.estado_analisis,
                a.fecha_analisis,

                u.nombre_completo,
                u.numero_identificacion,

                cr.numero_comprobante,
                cr.saldo_anterior,
                cr.saldo_nuevo

            FROM analisis_ia a

            INNER JOIN usuarios u
                ON a.id_usuario = u.id_usuario

            LEFT JOIN comprobantes_reciclaje cr
                ON cr.id_analisis = a.id_analisis

            """ + where_sql + """
            ORDER BY a.fecha_analisis DESC
        """,
            parametros
        )

        reciclajes = cursor.fetchall()

        total_general = 0
        cursor.execute(
            "SELECT COUNT(*) AS total FROM analisis_ia a" + where_sql,
            parametros
        )
        fila_total = cursor.fetchone()
        total_general = fila_total["total"] if fila_total else 0

        if not ver_todos and len(reciclajes) > limite_max:
            reciclajes = reciclajes[:limite_max]


        # ==========================================
        # MOSTRAR VISTA
        # ==========================================

        return render_template(
            "admin/admin_reciclajes.html",
            reciclajes=reciclajes,
            ver_todos=ver_todos,
            limite_max=limite_max,
            total_general=total_general,
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta
        )


    except Exception as e:

        print("ERROR RECICLAJES ADMIN:", e)

        return render_template(
            "admin/admin_reciclajes.html",
            reciclajes=[],
            ver_todos=False,
            limite_max=10,
            total_general=0,
            fecha_desde="",
            fecha_hasta=""
        )


    finally:

        cursor.close()
        conexion.close()




# ==========================================
# EXPORTAR RECICLAJES PDF
# ==========================================
@app.route("/admin/reciclajes/exportar")
def admin_exportar_reciclajes_pdf():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash("No fue posible conectar con la base de datos.", "danger")
        return redirect(url_for("admin_reciclajes"))

    cursor = conexion.cursor(dictionary=True)

    try:

        fecha_desde = request.args.get("desde", "").strip()
        fecha_hasta = request.args.get("hasta", "").strip()

        condiciones = []
        parametros = []

        if fecha_desde:
            condiciones.append("DATE(a.fecha_analisis) >= %s")
            parametros.append(fecha_desde)

        if fecha_hasta:
            condiciones.append("DATE(a.fecha_analisis) <= %s")
            parametros.append(fecha_hasta)

        where_sql = (" WHERE " + " AND ".join(condiciones)) if condiciones else ""

        cursor.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(a.puntos_totales), 0) AS suma "
            "FROM analisis_ia a" + where_sql,
            parametros
        )
        resumen = cursor.fetchone() or {"total": 0, "suma": 0}

        cursor.execute(
            "SELECT a.id_analisis, a.botella_detectada, a.tapa_detectada, "
            "a.etiqueta_detectada, a.puntos_totales, u.nombre_completo, "
            "u.numero_identificacion, cr.numero_comprobante, a.fecha_analisis "
            "FROM analisis_ia a "
            "INNER JOIN usuarios u ON a.id_usuario = u.id_usuario "
            "LEFT JOIN comprobantes_reciclaje cr ON cr.id_analisis = a.id_analisis"
            + where_sql + " "
            "ORDER BY a.fecha_analisis DESC",
            parametros
        )
        reciclajes = cursor.fetchall()

        nombre_archivo = "reciclajes_atlas.pdf"
        if fecha_desde and fecha_hasta:
            nombre_archivo = f"reciclajes_atlas_{fecha_desde}_a_{fecha_hasta}.pdf"

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(letter),
            leftMargin=14 * mm,
            rightMargin=14 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
            title="Reciclajes - Atlas",
            author="Atlas"
        )

        estilos = getSampleStyleSheet()

        estilo_titulo = ParagraphStyle(
            "TituloAtlas",
            parent=estilos["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#2f6bff"),
            spaceAfter=2
        )

        estilo_subtitulo = ParagraphStyle(
            "SubtituloAtlas",
            parent=estilos["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=8
        )

        estilo_encabezado = ParagraphStyle(
            "EncabezadoAtlas",
            parent=estilos["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=colors.white
        )

        estilo_celda = ParagraphStyle(
            "CeldaAtlas",
            parent=estilos["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10
        )

        elementos = []

        elementos.append(Paragraph("Reporte de Reciclajes", estilo_titulo))
        elementos.append(
            Paragraph("Atlas — Sistema de gestión de reciclaje inteligente", estilo_subtitulo)
        )

        rango_texto = "Todo el historial"
        if fecha_desde or fecha_hasta:
            rango_texto = f"{fecha_desde or 'inicio'} → {fecha_hasta or 'hoy'}"

        elementos.append(
            Paragraph(
                f"Rango analizado: <b>{rango_texto}</b> &nbsp;|&nbsp; "
                f"Total de reciclajes: <b>{resumen['total']}</b> &nbsp;|&nbsp; "
                f"AtlasPuntos otorgados: <b>{resumen['suma']}</b>",
                estilo_subtitulo
            )
        )

        elementos.append(
            HRFlowable(
                width="100%",
                thickness=1.4,
                color=colors.HexColor("#00e5ff"),
                spaceBefore=2,
                spaceAfter=12
            )
        )

        encabezados = [
            "ID",
            "Usuario",
            "Identificación",
            "Comprobante",
            "Botella",
            "Tapa",
            "Etiqueta",
            "Confianza",
            "Puntos",
            "Fecha"
        ]

        filas = [encabezados]

        if reciclajes:
            for r in reciclajes:
                filas.append([
                    str(r["id_analisis"]),
                    r["nombre_completo"],
                    r["numero_identificacion"],
                    r["numero_comprobante"] or "—",
                    "Sí" if r["botella_detectada"] else "No",
                    "Sí" if r["tapa_detectada"] else "No",
                    "Sí" if r["etiqueta_detectada"] else "No",
                    f"{float(r['confianza'] or 0):.1f}%",
                    str(r["puntos_totales"]),
                    str(r["fecha_analisis"])
                ])
        else:
            filas.append(["", "Sin registros en el rango seleccionado", "", "", "", "", "", "", "", ""])

        tabla = Table(
            filas,
            colWidths=[
                20 * mm,
                55 * mm,
                35 * mm,
                40 * mm,
                22 * mm,
                22 * mm,
                22 * mm,
                28 * mm,
                20 * mm,
                55 * mm
            ],
            repeatRows=1
        )

        tabla.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f6bff")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4ff")]),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3d3ef")),
                    ("ALIGN", (4, 1), (8, -1), "CENTER"),
                    ("ALIGN", (0, 1), (0, -1), "CENTER"),
                ]
            )
        )

        elementos.append(tabla)

        elementos.append(Spacer(1, 10))
        elementos.append(
            Paragraph(
                f"Documento generado por Atlas el {datetime.now().strftime('%d/%m/%Y %H:%M')}.",
                estilo_subtitulo
            )
        )

        doc.build(elementos)

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=nombre_archivo,
            mimetype="application/pdf"
        )

    except Exception as e:

        print("ERROR EXPORTAR RECICLAJES PDF:", e)
        flash(f"Error al generar el PDF: {e}", "danger")
        return redirect(url_for("admin_reciclajes"))

    finally:

        cursor.close()
        conexion.close()




@app.route("/admin/catalogo")
def admin_catalogo():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(url_for("login"))

    cursor = conexion.cursor(dictionary=True)

    try:

        cursor.execute("""
            SELECT
                id_producto,
                nombre,
                descripcion,
                imagen,
                costo_puntos,
                stock,
                fecha_registro
            FROM productos
            ORDER BY fecha_registro DESC
        """)

        productos = cursor.fetchall()

        print("PRODUCTOS CATALOGO:", productos)

        return render_template(
            "admin/admin_catalogo.html",
            productos=productos
        )

    except Exception as e:

        print("ERROR CATALOGO ADMIN:", e)

        flash(
            f"Error al cargar el catálogo: {e}",
            "danger"
        )

        return render_template(
            "admin/admin_catalogo.html",
            productos=[]
        )

    finally:

        cursor.close()
        conexion.close()





# ==========================================
# AUDITORÍAS
# ==========================================
@app.route("/admin/auditorias")
def admin_auditorias():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    conexion = None
    cursor = None

    try:

        conexion = obtener_conexion()

        if conexion is None:
            flash(
                "No fue posible conectar con la base de datos.",
                "danger"
            )
            return redirect(url_for("admin_dashboard"))

        cursor = conexion.cursor(dictionary=True)

        # ==========================================
        # AUDITORÍAS (REGISTRO DEL SISTEMA)
        # ==========================================

        cursor.execute("""
            SELECT
                a.id_auditoria,
                a.id_usuario,
                a.accion,
                a.tabla_afectada,
                a.id_registro,
                a.descripcion,
                a.ip,
                a.fecha,

                u.nombre_completo,
                u.numero_identificacion

            FROM auditoria a

            LEFT JOIN usuarios u
                ON a.id_usuario = u.id_usuario

            ORDER BY a.fecha DESC

            LIMIT 100
        """)

        auditorias = cursor.fetchall()

        # ==========================================
        # USUARIOS RECIENTES
        # ==========================================

        cursor.execute("""
            SELECT
                u.id_usuario,
                u.usuario,
                u.numero_identificacion,
                u.nombre_completo,
                u.correo,

                COALESCE(
                    (
                        SELECT SUM(mp.puntos)
                        FROM movimientos_puntos mp
                        WHERE mp.id_usuario = u.id_usuario
                    ),
                    0
                ) AS puntos_totales

            FROM usuarios u

            WHERE u.tipo_usuario = 'USUARIO'

            ORDER BY u.id_usuario DESC

            LIMIT 10
        """)

        usuarios_nuevos = cursor.fetchall()

        # ==========================================
        # CANJES RECIENTES
        # ==========================================

        cursor.execute("""
            SELECT
                c.id_canje,
                c.total_puntos,
                c.fecha_canje,

                u.nombre_completo,
                u.numero_identificacion,

                (
                    SELECT mp.motivo
                    FROM movimientos_puntos mp
                    WHERE mp.id_canje = c.id_canje
                    LIMIT 1
                ) AS descripcion_canje

            FROM canjes c

            LEFT JOIN usuarios u
                ON c.id_usuario = u.id_usuario

            ORDER BY c.fecha_canje DESC

            LIMIT 15
        """)

        canjes_recientes = cursor.fetchall()

        # ==========================================
        # ANÁLISIS IA RECIENTES
        # ==========================================

        cursor.execute("""
            SELECT
                ai.id_analisis,
                ai.fecha_analisis,
                ai.botella_detectada,
                ai.tapa_detectada,
                ai.etiqueta_detectada,
                ai.puntos_totales,
                ai.confianza,
                ai.estado_analisis,

                u.nombre_completo,
                u.numero_identificacion

            FROM analisis_ia ai

            LEFT JOIN usuarios u
                ON ai.id_usuario = u.id_usuario

            ORDER BY ai.fecha_analisis DESC

            LIMIT 15
        """)

        analisis_recientes = cursor.fetchall()

        # ==========================================
        # MOVIMIENTOS DE PUNTOS RECIENTES
        # ==========================================

        cursor.execute("""
            SELECT
                mp.id_movimiento,
                mp.tipo_movimiento,
                mp.puntos,
                mp.motivo,
                mp.fecha_movimiento,

                u.nombre_completo,
                u.numero_identificacion

            FROM movimientos_puntos mp

            LEFT JOIN usuarios u
                ON mp.id_usuario = u.id_usuario

            ORDER BY mp.fecha_movimiento DESC

            LIMIT 20
        """)

        movimientos_recientes = cursor.fetchall()

        # ==========================================
        # RANKING - TOP USUARIOS CON MÁS PUNTOS
        # ==========================================

        cursor.execute("""
            SELECT
                u.id_usuario,
                u.numero_identificacion,
                u.nombre_completo,

                COALESCE(SUM(mp.puntos), 0) AS puntos_totales,

                (
                    SELECT COUNT(*)
                    FROM analisis_ia ai
                    WHERE ai.id_usuario = u.id_usuario
                ) AS total_analisis,

                (
                    SELECT COUNT(*)
                    FROM canjes c
                    WHERE c.id_usuario = u.id_usuario
                ) AS total_canjes

            FROM usuarios u

            LEFT JOIN movimientos_puntos mp
                ON mp.id_usuario = u.id_usuario

            WHERE u.tipo_usuario = 'USUARIO'

            GROUP BY
                u.id_usuario,
                u.numero_identificacion,
                u.nombre_completo

            HAVING puntos_totales > 0

            ORDER BY puntos_totales DESC

            LIMIT 20
        """)

        ranking = cursor.fetchall()

        # ==========================================
        # ESTADÍSTICAS
        # ==========================================

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM auditoria
        """)

        total_acciones = cursor.fetchone()["total"]

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM usuarios
            WHERE tipo_usuario = 'USUARIO'
        """)

        total_usuarios = cursor.fetchone()["total"]

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM canjes
        """)

        total_canjes = cursor.fetchone()["total"]

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM analisis_ia
        """)

        total_analisis_ia = cursor.fetchone()["total"]

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM movimientos_puntos
        """)

        total_movimientos = cursor.fetchone()["total"]

        estadisticas = {
            "total_acciones": total_acciones,
            "total_usuarios": total_usuarios,
            "total_canjes": total_canjes,
            "total_analisis_ia": total_analisis_ia,
            "total_movimientos": total_movimientos
        }

        return render_template(
            "admin/admin_auditoria.html",
            auditorias=auditorias,
            usuarios_nuevos=usuarios_nuevos,
            canjes_recientes=canjes_recientes,
            analisis_recientes=analisis_recientes,
            movimientos_recientes=movimientos_recientes,
            ranking=ranking,
            estadisticas=estadisticas
        )

    except Exception as e:

        print("ERROR AUDITORIAS ADMIN:", e)

        return render_template(
            "admin/admin_auditoria.html",
            auditorias=[],
            usuarios_nuevos=[],
            canjes_recientes=[],
            analisis_recientes=[],
            movimientos_recientes=[],
            ranking=[],
            estadisticas={
                "total_acciones": 0,
                "total_usuarios": 0,
                "total_canjes": 0,
                "total_analisis_ia": 0,
                "total_movimientos": 0
            }
        )

    finally:

        if cursor:
            cursor.close()

        if conexion:
            conexion.close()


# ==========================================
# MOVIMIENTOS (INTERCAMBIO DE PUNTOS)
# ==========================================
@app.route("/admin/movimientos")
def admin_movimientos():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    conexion = obtener_conexion()

    if conexion is None:
        flash(
            "No fue posible conectar con la base de datos.",
            "danger"
        )
        return redirect(url_for("admin_dashboard"))

    cursor = conexion.cursor(dictionary=True)

    try:

        cursor.execute("""
            SELECT
                id_producto,
                nombre,
                descripcion,
                costo_puntos,
                stock
            FROM productos
            WHERE stock > 0
            ORDER BY costo_puntos ASC
        """)

        productos = cursor.fetchall()

        cursor.execute("""
            SELECT
                u.id_usuario,
                u.numero_identificacion,
                u.nombre_completo,

                COALESCE(
                    (
                        SELECT SUM(mp.puntos)
                        FROM movimientos_puntos mp
                        WHERE mp.id_usuario = u.id_usuario
                    ),
                    0
                ) AS puntos_totales

            FROM usuarios u
            WHERE u.tipo_usuario = 'USUARIO'
            ORDER BY u.nombre_completo ASC
        """)

        usuarios = cursor.fetchall()

        return render_template(
            "admin/movimientos.html",
            productos=productos,
            usuarios=usuarios
        )

    except Exception as e:

        print("ERROR MOVIMIENTOS:", e)
        flash(
            f"Error al cargar movimientos: {e}",
            "danger"
        )
        return redirect(url_for("admin_dashboard"))

    finally:
        cursor.close()
        conexion.close()


@app.route("/admin/api/usuario/<identificacion>")
def api_buscar_usuario(identificacion):

    if "id_usuario" not in session:
        return jsonify({"error": "Sesión no válida."}), 401

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return jsonify({"error": "No autorizado."}), 403

    conexion = obtener_conexion()

    if conexion is None:
        return jsonify({"error": "Error de conexión."}), 500

    cursor = conexion.cursor(dictionary=True)

    try:

        cursor.execute("""
            SELECT
                u.id_usuario,
                u.usuario,
                u.numero_identificacion,
                u.nombre_completo,
                u.correo,

                COALESCE(
                    (
                        SELECT SUM(mp.puntos)
                        FROM movimientos_puntos mp
                        WHERE mp.id_usuario = u.id_usuario
                    ),
                    0
                ) AS puntos_totales

            FROM usuarios u
            WHERE u.numero_identificacion = %s
        """, (identificacion,))

        usuario = cursor.fetchone()

        if usuario is None:
            return jsonify({
                "error": "No se encontró un usuario con esa identificación."
            }), 404

        return jsonify(usuario)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conexion.close()


@app.route("/admin/movimientos/canjar", methods=["POST"])
def admin_canjar_puntos():

    if "id_usuario" not in session:
        return redirect(url_for("login"))

    if session.get("tipo_usuario") != "ADMINISTRADOR":
        return redirect(url_for("dashboard"))

    datos = request.get_json()

    if not datos:
        return jsonify({"error": "No se recibieron datos."}), 400

    id_usuario = datos.get("id_usuario")
    id_producto = datos.get("id_producto")

    if not id_usuario or not id_producto:
        return jsonify({
            "error": "Faltan datos: id_usuario e id_producto son obligatorios."
        }), 400

    conexion = obtener_conexion()

    if conexion is None:
        return jsonify({"error": "Error de conexión."}), 500

    cursor = conexion.cursor(dictionary=True)

    try:

        # ==========================================
        # 1. OBTENER PRODUCTO
        # ==========================================

        cursor.execute("""
            SELECT id_producto, nombre, costo_puntos, stock
            FROM productos
            WHERE id_producto = %s
        """, (id_producto,))

        producto = cursor.fetchone()

        if producto is None:
            return jsonify({
                "error": "El producto no existe."
            }), 404

        if producto["stock"] <= 0:
            return jsonify({
                "error": "El producto no tiene stock disponible."
            }), 400

        # ==========================================
        # 2. OBTENER SALDO DEL USUARIO
        # ==========================================

        cursor.execute("""
            SELECT COALESCE(SUM(puntos), 0) AS saldo
            FROM movimientos_puntos
            WHERE id_usuario = %s
        """, (id_usuario,))

        resultado = cursor.fetchone()
        saldo_actual = int(resultado["saldo"] or 0)

        costo = int(producto["costo_puntos"])

        if saldo_actual < costo:
            return jsonify({
                "error": (
                    f"El usuario tiene {saldo_actual} AtlasPuntos "
                    f"pero el producto cuesta {costo}. "
                    f"Faltan {costo - saldo_actual} puntos."
                )
            }), 400

        # ==========================================
        # 3. INSERTAR CANJE
        # ==========================================

        cursor.execute("""
            INSERT INTO canjes (id_usuario, total_puntos)
            VALUES (%s, %s)
        """, (id_usuario, costo))

        id_canje = cursor.lastrowid

        # ==========================================
        # 4. INSERTAR MOVIMIENTO DE PUNTOS
        # ==========================================

        motivo = (
            f"Canje de {costo} AtlasPuntos "
            f"por producto: {producto['nombre']}"
        )

        cursor.execute("""
            INSERT INTO movimientos_puntos (
                id_usuario,
                id_canje,
                tipo_movimiento,
                puntos,
                motivo,
                id_usuario_admin
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            id_usuario,
            id_canje,
            "CANJE",
            -costo,
            motivo,
            session["id_usuario"]
        ))

        # ==========================================
        # 5. ACTUALIZAR STOCK DEL PRODUCTO
        # ==========================================

        cursor.execute("""
            UPDATE productos
            SET stock = stock - 1
            WHERE id_producto = %s
        """, (id_producto,))

        # ==========================================
        # 6. REGISTRAR AUDITORÍA
        # ==========================================

        cursor.execute("""
            INSERT INTO auditoria (
                id_usuario,
                accion,
                tabla_afectada,
                id_registro,
                descripcion,
                ip
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            session["id_usuario"],
            "CANJE",
            "productos",
            id_producto,
            (
                f"Canje de {costo} AtlasPuntos por "
                f"'{producto['nombre']}' "
                f"(Usuario ID: {id_usuario})."
            ),
            request.remote_addr
        ))

        # ==========================================
        # 7. CONFIRMAR
        # ==========================================

        conexion.commit()

        saldo_nuevo = saldo_actual - costo

        return jsonify({
            "success": True,
            "mensaje": (
                f"Canje exitoso: {producto['nombre']} "
                f"por {costo} AtlasPuntos."
            ),
            "saldo_anterior": saldo_actual,
            "saldo_nuevo": saldo_nuevo,
            "puntos_gastados": costo,
            "producto": producto["nombre"],
            "numero_canje": f"CANJ-{str(id_canje).zfill(6)}"
        })

    except Exception as e:

        conexion.rollback()
        print("ERROR CANJANDO:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conexion.close()


# ==========================================
# EJECUTAR
# ==========================================
if __name__ == "__main__":
    app.run(
        debug=True
    )

