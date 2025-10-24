# Binance Crypto Day-Trading Bot

Bot de trading diario de criptomonedas diseñado para operar 24/7 en Binance con análisis técnico avanzado, modelo LSTM y agente DQN para aprendizaje por refuerzo.

## ⚠️ Descargo de responsabilidad
Este repositorio se proporciona con fines educativos. Operar con criptomonedas implica un riesgo elevado de pérdida de capital. Úsalo bajo tu propia responsabilidad.

## 🚀 Características principales
- Integración con Binance utilizando claves almacenadas en variables de entorno (.env + python-dotenv).
- Cálculo de indicadores profesionales (RSI, divergencias, MACD, SuperTrend, Bandas de Bollinger, EMAs, ADX y patrones de velas con TA-Lib) sobre velas de 5m y 15m.
- Salidas normalizadas listas para IA y generación de señales discretas de compra/venta.
- Modelo LSTM en TensorFlow para clasificación direccional (+1, 0, -1) con secuencias de 60 pasos.
- Entorno de aprendizaje por refuerzo con DQN (stable-baselines3) y recompensas basadas en el desempeño del portafolio.
- Gestión de riesgo con stop-loss (-1.5%), take-profit (+3%), comisiones de Binance (0.1%) y protección contra órdenes duplicadas.
- Modos de operación: entrenamiento LSTM, entrenamiento RL, backtesting, paper trading y trading en vivo.
- Logging detallado en archivos rotativos para auditoría continua.

## 📁 Estructura
```
bot/
├── backtester.py      # Backtesting y métricas
├── config.py          # Configuración global y logging
├── data_loader.py     # Descarga y cacheo de datos OHLCV
├── indicators.py      # Indicadores técnicos y señales
├── lstm_model.py      # Entrenamiento e inferencia LSTM
├── main.py            # CLI principal
├── rl_agent.py        # Entorno Gym y entrenamiento DQN
├── strategy.py        # Estrategia ensamblada
└── trader.py          # Gestión de posiciones y órdenes
```

## 📦 Instalación
1. Crea y activa un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Instala dependencias:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Copia el archivo `.env.example` a `.env` y agrega tus credenciales de Binance:
   ```bash
   cp .env.example .env
   ```

## 🔐 Variables de entorno
| Variable | Descripción |
|----------|-------------|
| `BINANCE_API_KEY` | Clave API de Binance con permisos de lectura y trading (según el modo). |
| `BINANCE_API_SECRET` | Secreto de la API. |

**No subas tus claves reales a ningún repositorio.**

## 🧠 Entrenamiento del modelo LSTM
```bash
python -m bot.main --train-lstm
```
El modelo entrenado se guardará en `models/lstm_classifier.h5` e incluye métricas de accuracy, precision y recall en los logs.

## 🤖 Entrenamiento del agente DQN
Asegúrate de tener un modelo LSTM entrenado previamente. Luego ejecuta:
```bash
python -m bot.main --train-rl
```
La política aprendida se almacenará en `models/dqn_policy.zip`.

## 📈 Backtesting
```bash
python -m bot.main --backtest
```
Mostrará métricas de PnL, win rate y últimos trades registrados.

## 🧪 Paper trading
```bash
python -m bot.main --paper
```
Este modo simula la operativa en tiempo real sin enviar órdenes a Binance. Requiere un modelo LSTM entrenado.

## 💹 Trading en vivo
```bash
python -m bot.main --live
```
**Advertencia:** prueba exhaustivamente los modos de entrenamiento y paper trading antes de activar este modo. Verifica límites de la API y protección de claves.

## 🛠️ Desarrollo y extensiones
- El código está organizado para escalar a múltiples pares y estrategias adicionales.
- Los módulos incluyen manejo de excepciones para asegurar resiliencia 24/7.
- Los logs rotativos en `logs/` permiten auditar las sesiones.

## 🧾 Licencia
MIT
