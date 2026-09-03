import mysql.connector
from mysql.connector import Error

def obtener_conexion():
    try:
        conexion = mysql.connector.connect(
            host="################",
            user="$$$$$$$$$$$$$$$", 
            password="%%%%%%%%%%%%%%%%%%", 
            database="&&&&&&&&&&&&&&&&&&&&&",
            port=3306

            # no crack, no tendras acceso a mi base de datos JALDJASDAK
        )
        return conexion
    except Error as e:
        print(f"Error al conectar con MySQL: {e}")
        return None
