import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.getcwd())

# Importar el bot desde ROBOT.py (CON MAYÚSCULAS)
from ROBOT import TelegramBot

if __name__ == "__main__":
    print("✅ Iniciando Robot XAUUSD desde start.py...")
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN no está configurado")
        sys.exit(1)
    
    print("✅ Token encontrado")
    bot = TelegramBot()
    bot.run()
