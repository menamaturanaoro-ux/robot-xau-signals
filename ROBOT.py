import asyncio
import logging
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
import json
import os
warnings.filterwarnings('ignore')

# ================= CONFIGURACIÓN =================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
# ========== CONFIGURACIÓN DE METATRADER 5 (VANTAGE) ==========
MT5_TERMINAL_PATH = r"C:\Program Files\Vantage International MT5\terminal64.exe"
SYMBOL = "XAUUSD"

# ========== TEMPORALIDAD ==========
TIMEFRAME = mt5.TIMEFRAME_M15
LIMIT = 150

# ========== CONFIGURACIÓN DE SEÑALES ==========
ENVIO_AUTOMATICO = True
INTERVALO_MINUTOS = 15
CHAT_ID = None

# ========== CONFIGURACIÓN DE EXPIRACIÓN ==========
DIAS_PRUEBA_GRATIS = 3

# ================= ARCHIVO DE USUARIOS =================
ARCHIVO_USUARIOS = "usuarios.json"

def cargar_usuarios():
    if os.path.exists(ARCHIVO_USUARIOS):
        with open(ARCHIVO_USUARIOS, 'r') as f:
            return json.load(f)
    return {}

def guardar_usuarios(usuarios):
    with open(ARCHIVO_USUARIOS, 'w') as f:
        json.dump(usuarios, f, indent=4)

# ================= LOGGING =================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================= CLASE DEL ROBOT =================
class TradingSignalRobot:
    def __init__(self):
        self.last_signal_time = None
        self.signal_cooldown = 15
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.mt5_connected = False
        self._initialize_mt5()
        
    def _initialize_mt5(self):
        try:
            if not mt5.initialize(MT5_TERMINAL_PATH):
                if not mt5.initialize():
                    logger.error("No se pudo inicializar MT5")
                    return False
            
            if not mt5.symbol_select(SYMBOL, True):
                logger.error(f"Símbolo {SYMBOL} no disponible en MT5")
                mt5.shutdown()
                return False
            
            self.mt5_connected = True
            logger.info(f"✅ Conectado a MT5 - Símbolo: {SYMBOL}")
            logger.info("🧠 IA LSTM + RSI")
            logger.info("📡 Señales cada: 15 minutos")
            return True
            
        except Exception as e:
            logger.error(f"Error en _initialize_mt5: {e}")
            return False
    
    def get_market_data(self):
        if not self.mt5_connected:
            logger.error("MT5 no está conectado")
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LIMIT)
            
            if rates is None or len(rates) == 0:
                logger.error("No se obtuvieron datos de MT5")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            df = df[['open', 'high', 'low', 'close', 'tick_volume']]
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error en get_market_data: {e}")
            return None

    # ========== RSI ==========
    def calcular_rsi(self, df, periodo=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ========== RED NEURONAL LSTM ==========
    def entrenar_modelo(self, df):
        try:
            data = df['close'].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data)
            
            seq_length = 20
            X, y = [], []
            for i in range(seq_length, len(scaled_data)):
                X.append(scaled_data[i-seq_length:i, 0])
                y.append(scaled_data[i, 0])
            
            if len(X) < 10:
                return df['close'].iloc[-1]
            
            X = np.array(X)
            y = np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            if self.model is None:
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                self.model = model
                self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
                self.is_trained = True
            else:
                self.model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            
            last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
            next_price_scaled = self.model.predict(last_sequence, verbose=0)
            next_price = self.scaler.inverse_transform(next_price_scaled)[0][0]
            
            return next_price
        except:
            return df['close'].iloc[-1]

    # ========== ANÁLISIS COMPLETO - SOLO IA + RSI ==========
    def analyze_market(self):
        df = self.get_market_data()
        if df is None:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # ========== 1. IA LSTM ==========
        try:
            predicted_price = self.entrenar_modelo(df)
        except:
            predicted_price = current_price
        
        # Señal IA
        if predicted_price > current_price * 1.003:
            ia_signal = 'BUY'
            ia_score = 85
        elif predicted_price < current_price * 0.997:
            ia_signal = 'SELL'
            ia_score = 85
        else:
            ia_signal = 'NEUTRAL'
            ia_score = 50
        
        # ========== 2. RSI ==========
        rsi = self.calcular_rsi(df, 14)
        rsi_actual = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # Puntuación RSI
        if rsi_actual < 30:
            rsi_signal = 'BUY'  # Sobreventa → Comprar
            rsi_score = 85
        elif rsi_actual > 70:
            rsi_signal = 'SELL'  # Sobrecompra → Vender
            rsi_score = 85
        elif rsi_actual < 40:
            rsi_signal = 'BUY'
            rsi_score = 65
        elif rsi_actual > 60:
            rsi_signal = 'SELL'
            rsi_score = 65
        else:
            rsi_signal = 'NEUTRAL'
            rsi_score = 50
        
        # ========== 3. DECISIÓN FINAL ==========
        # ¿IA y RSI están alineados?
        alineados = (ia_signal == rsi_signal) and ia_signal != 'NEUTRAL'
        
        if alineados:
            # Si están alineados, la señal es fuerte
            if ia_signal == 'BUY':
                final_signal = 'BUY'
                confidence = 85
            elif ia_signal == 'SELL':
                final_signal = 'SELL'
                confidence = 85
            else:
                final_signal = 'NEUTRAL'
                confidence = 0
        else:
            # Si no están alineados, ponderamos
            # 50% IA + 50% RSI
            weighted_score = (ia_score * 0.50 + rsi_score * 0.50)
            
            if weighted_score >= 65:
                final_signal = 'BUY'
                confidence = min(weighted_score, 85)
            elif weighted_score <= 35:
                final_signal = 'SELL'
                confidence = min(100 - weighted_score, 85)
            else:
                final_signal = 'NEUTRAL'
                confidence = 0
        
        # ========== 4. NIVELES (ATR) ==========
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high[1:] - low[1:]
        tr2 = abs(high[1:] - close[:-1])
        tr3 = abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_val = np.mean(tr[-14:])
        
        if final_signal == 'BUY':
            entry = current_price
            sl = entry - atr_val * 1.3
            tp1 = entry + atr_val * 1.3
            tp2 = entry + atr_val * 2.0
            tp3 = entry + atr_val * 3.0
        elif final_signal == 'SELL':
            entry = current_price
            sl = entry + atr_val * 1.3
            tp1 = entry - atr_val * 1.3
            tp2 = entry - atr_val * 2.0
            tp3 = entry - atr_val * 3.0
        else:
            entry = sl = tp1 = tp2 = tp3 = current_price
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'ia_signal': ia_signal,
            'ia_score': ia_score,
            'rsi': rsi_actual,
            'rsi_signal': rsi_signal,
            'rsi_score': rsi_score,
            'alineados': alineados,
            'atr': atr_val
        }
    
    def __del__(self):
        try:
            if self.mt5_connected:
                mt5.shutdown()
        except:
            pass

# ================= BOT DE TELEGRAM =================
class TelegramBot:
    def __init__(self):
        self.robot = TradingSignalRobot()
        self.application = Application.builder().token(TELEGRAM_TOKEN).build()
        self.last_signal = None
        self.signal_time = None
        self.chat_id = None
        self.processing = False
        self.usuarios = cargar_usuarios()
        
    def usuario_activo(self, user_id):
        if str(user_id) not in self.usuarios:
            fecha_inicio = datetime.now()
            fecha_expiracion = fecha_inicio + timedelta(days=DIAS_PRUEBA_GRATIS)
            self.usuarios[str(user_id)] = {
                "fecha_inicio": fecha_inicio.isoformat(),
                "fecha_expiracion": fecha_expiracion.isoformat(),
                "activo": DIAS_PRUEBA_GRATIS > 0
            }
            guardar_usuarios(self.usuarios)
            return DIAS_PRUEBA_GRATIS > 0
        
        usuario = self.usuarios[str(user_id)]
        fecha_expiracion = datetime.fromisoformat(usuario["fecha_expiracion"])
        return datetime.now() < fecha_expiracion
    
    def obtener_tiempo_restante(self, user_id):
        usuario = self.usuarios.get(str(user_id))
        if not usuario:
            return 0
        fecha_expiracion = datetime.fromisoformat(usuario["fecha_expiracion"])
        delta = fecha_expiracion - datetime.now()
        return max(0, delta.total_seconds() / 86400)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        self.chat_id = update.effective_chat.id
        
        if not self.usuario_activo(user_id):
            await update.message.reply_text(
                "⏰ *TU PERÍODO DE PRUEBA HA EXPIRADO*\n\nContacta al administrador: @TU_USUARIO_TELEGRAM",
                parse_mode='Markdown'
            )
            return
        
        dias_restantes = self.obtener_tiempo_restante(user_id)
        mt5_status = "✅ Conectado" if self.robot.mt5_connected else "❌ Desconectado"
        
        mensaje = (
            "🤖 *ROBOT XAUUSD - IA + RSI*\n\n"
            "🧠 *SOLO DOS FACTORES:*\n\n"
            "✅ IA LSTM (Predicción de precios)\n"
            "✅ RSI (Sobrecompra/Sobreventa)\n\n"
            "📊 *LÓGICA:*\n"
            "• IA y RSI alineados → SEÑAL FUERTE\n"
            "• Desalineados → Ponderación 50/50\n\n"
            f"📊 *Símbolo:* {SYMBOL}\n"
            f"⏱️ *Timeframe:* M15\n"
            f"📡 *MT5:* {mt5_status}\n"
            f"⏰ *Días:* {dias_restantes:.1f}\n\n"
            "📱 /signal - Señal manual\n"
            "📱 /stop - Detener señales"
        )
        
        await update.message.reply_text(mensaje, parse_mode='Markdown')
        
        if ENVIO_AUTOMATICO and not self.processing and self.robot.mt5_connected:
            self.processing = True
            asyncio.create_task(self.enviar_senales_automaticas())
    
    async def signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        self.chat_id = update.effective_chat.id
        
        if not self.usuario_activo(user_id):
            await update.message.reply_text("⏰ *SUSCRIPCIÓN EXPIRADA*", parse_mode='Markdown')
            return
        
        if self.signal_time:
            elapsed = (datetime.now() - self.signal_time).total_seconds() / 60
            if elapsed < INTERVALO_MINUTOS:
                remaining = int(INTERVALO_MINUTOS - elapsed)
                await update.message.reply_text(
                    f"⏳ *Espera {remaining} min*",
                    parse_mode='Markdown'
                )
                return
        
        await self.enviar_analisis(update.message)
    
    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.processing = False
        await update.message.reply_text("⏸️ *Señales DETENIDAS*", parse_mode='Markdown')
    
    async def enviar_analisis(self, message):
        processing_msg = await message.reply_text(
            "🧠 *PROCESANDO IA + RSI...*",
            parse_mode='Markdown'
        )
        
        try:
            analysis = self.robot.analyze_market()
            await self.enviar_mensaje_senal(analysis, processing_msg)
        except Exception as e:
            logger.error(f"Error en análisis: {e}")
            await processing_msg.edit_text("❌ *ERROR*", parse_mode='Markdown')
    
    async def enviar_senales_automaticas(self):
        while self.processing and ENVIO_AUTOMATICO:
            try:
                if self.chat_id:
                    await asyncio.sleep(INTERVALO_MINUTOS * 60)
                    
                    if self.signal_time:
                        elapsed = (datetime.now() - self.signal_time).total_seconds() / 60
                        if elapsed < INTERVALO_MINUTOS:
                            continue
                    
                    if not self.usuario_activo(self.chat_id):
                        await self.application.bot.send_message(
                            chat_id=self.chat_id,
                            text="⏰ *SUSCRIPCIÓN EXPIRADA*",
                            parse_mode='Markdown'
                        )
                        self.processing = False
                        break
                    
                    processing_msg = await self.application.bot.send_message(
                        chat_id=self.chat_id,
                        text="🧠 *SEÑAL AUTOMÁTICA*",
                        parse_mode='Markdown'
                    )
                    
                    analysis = self.robot.analyze_market()
                    await self.enviar_mensaje_senal(analysis, processing_msg)
                    
                    self.signal_time = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(30)
    
    async def enviar_mensaje_senal(self, analysis, processing_msg=None):
        if analysis is None:
            mensaje = "❌ *ERROR* - No se pudieron obtener datos"
            if processing_msg:
                await processing_msg.edit_text(mensaje, parse_mode='Markdown')
            return
        
        if analysis['signal'] == 'NEUTRAL':
            mensaje = (
                f"⚠️ *NEUTRAL - XAUUSD*\n\n"
                f"📊 IA: {analysis['ia_signal']} (Score: {analysis['ia_score']:.0f})\n"
                f"📊 RSI: {analysis['rsi']:.1f} ({analysis['rsi_signal']}) (Score: {analysis['rsi_score']:.0f})\n"
                f"🔗 Alineación: {'✅' if analysis['alineados'] else '❌'}\n\n"
                f"⏰ Próxima en {INTERVALO_MINUTOS} min"
            )
            if processing_msg:
                await processing_msg.edit_text(mensaje, parse_mode='Markdown')
            return
        
        emoji = "🟢" if analysis['signal'] == 'BUY' else "🔴"
        
        conf = analysis['confidence']
        if conf >= 70:
            calidad = "🟢 ALTA"
        elif conf >= 40:
            calidad = "🟡 MEDIA"
        else:
            calidad = "🔴 BAJA"
        
        mensaje = f"""
{emoji} *SEÑAL {analysis['signal']} - XAUUSD*
━━━━━━━━━━━━━━━━━━━━━━━

💰 *PRECIO:* ${analysis['current_price']:.2f}
🎯 *ENTRADA:* ${analysis['entry']:.2f}
🛑 *STOP LOSS:* ${analysis['sl']:.2f}

🎯 *TP1:* ${analysis['tp1']:.2f}
🎯 *TP2:* ${analysis['tp2']:.2f}
🎯 *TP3:* ${analysis['tp3']:.2f}

🧠 *ANÁLISIS (IA + RSI):*

🤖 *IA LSTM:*
• Predicción: ${analysis['predicted_price']:.2f}
• Señal: {analysis['ia_signal']}
• Score: {analysis['ia_score']:.0f}/100

📊 *RSI:*
• Valor: {analysis['rsi']:.1f}
• Señal: {analysis['rsi_signal']}
• Score: {analysis['rsi_score']:.0f}/100

🔗 *ALINEACIÓN:*
• IA y RSI alineados: {'✅ SI' if analysis['alineados'] else '❌ NO'}

📊 *RESULTADO:*
• Confianza: {analysis['confidence']:.0f}% - {calidad}

⏰ *Próxima:* {INTERVALO_MINUTOS} min
━━━━━━━━━━━━━━━━━━━━━━━
🧠 IA LSTM + RSI
"""
        
        keyboard = [[InlineKeyboardButton("📊 Señal Manual", callback_data='manual_signal')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if processing_msg:
            await processing_msg.edit_text(mensaje, parse_mode='Markdown', reply_markup=reply_markup)
        else:
            await self.application.bot.send_message(chat_id=self.chat_id, text=mensaje, parse_mode='Markdown', reply_markup=reply_markup)
        
        self.signal_time = datetime.now()
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        if query.data == 'manual_signal':
            await self.signal(update, context)
    
    def run(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("signal", self.signal))
        self.application.add_handler(CommandHandler("stop", self.stop))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        print("=" * 60)
        print("🤖 ROBOT XAUUSD - IA + RSI")
        print("=" * 60)
        print(f"📊 Símbolo: {SYMBOL}")
        print(f"⏱️ Timeframe: M15")
        print(f"📡 Señales: {INTERVALO_MINUTOS} min")
        print(f"📡 MT5: {'✅' if self.robot.mt5_connected else '❌'}")
        print("-" * 60)
        print("🧠 FACTORES:")
        print("   • IA LSTM (Predicción)")
        print("   • RSI (Sobrecompra/Sobreventa)")
        print("=" * 60)
        
        self.application.run_polling()

if __name__ == "__main__":
    bot = TelegramBot()
    bot.run()