from robot import TelegramBot
import os

if __name__ == "__main__":
    # Verificar que el token existe
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        print("❌ ERROR: TELEGRAM_BOT_TOKEN no está configurado")
        exit(1)
    
    print("✅ Iniciando Robot XAUUSD...")
    bot = TelegramBot()
    bot.run()
