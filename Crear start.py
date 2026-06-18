from robot import TelegramBot
import os

if __name__ == "__main__":
    print("✅ Iniciando Robot XAUUSD...")
    
    # Verificar que el token existe
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN no está configurado")
        exit(1)
    
    print("✅ Token encontrado")
    bot = TelegramBot()
    bot.run()
