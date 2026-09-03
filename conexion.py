import mysql.connector
from mysql.connector import Error

def obtener_conexion():
    try:
        conexion = mysql.connector.connect(
            host="bcbuhydxsy4laalvcgvo-mysql.services.clever-cloud.com",
            user="uedp1u5czob44lvw", # El usuario que te muestra Clever Cloud en la pestaña Information
            password="2sbxorADR3e7iOgS7ZH3", # La contraseña que te muestra Clever Cloud en la pestaña Information
            database="bcbuhydxsy4laalvcgvo",
            port=3306
        )
        return conexion
    except Error as e:
        print(f"Error al conectar con MySQL: {e}")
        return None
