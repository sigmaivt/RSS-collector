# RSS → LLM → Telegram

Локальный сервис автоматического сбора, фильтрации и суммаризации технологических новостей с доставкой в Telegram.

## Архитектура

```
RSS Feeds → Feed Poller → Normalizer → LLM Classifier → LLM Summarizer → Telegram
                                            ↑                    ↑
                                      prompts/*.txt         prompts/*.txt
                                     (hot reload)          (hot reload)
```

## Каналы (темы)

| Канал | Описание | Источники |
|-------|----------|-----------|
| `ai_coding` | AI-инструменты для кодинга | TechCrunch, The Verge, HN, GitHub, Wired, Ars, InfoQ, Habr, OpenNet, The Register |
| `ai_models` | Модели и исследования AI/ML | HF Papers, HN, Google Research, OpenAI, VentureBeat, HF Blog, Habr ML |
| `erp_mes` | ERP/MES и автоматизация | ERP Software Blog, Automation.com, Automation World, TAdviser, CNews |

## Быстрый старт

### 1. Инфраструктура

```bash
# RSSHub + Redis
docker compose up -d

# LM Studio — запустить GUI, загрузить модель, Start Server (порт 1234)

# Telegram Bot — через @BotFather, записать токен в settings.env
```

### 2. Настройка

```bash
cp config/settings.env .env
# Заполнить: TG_BOT_TOKEN, GITHUB_ACCESS_TOKEN, chat_id для каналов
```

### 3. Запуск

```bash
pip install -r requirements.txt
python src/main.py
```

## Стек

- **Python 3.11+** — основной runtime
- **RSSHub + Redis** — агрегация RSS (Docker)
- **LM Studio** — локальная LLM (OpenAI-совместимый API)
- **SQLite (WAL)** — хранилище, очередь, метрики
- **Telegram Bot API** — доставка через httpx

## Мониторинг

- Health checks каждые 5 минут (LM Studio, RSSHub, feeds, queue, Telegram, disk)
- Алерты в admin-чат Telegram (с cooldown 30 мин)
- Дневной дайджест: статистика по каналам
- Метрики в SQLite (pipeline_metrics)

## Структура

```
rss-llm-filter/
├── docker-compose.yaml
├── config/
│   ├── channels.yaml       # каналы, источники, промпты
│   └── settings.env        # токены и настройки
├── prompts/
│   ├── classifier_ai_coding.txt
│   ├── classifier_ai_models.txt
│   ├── classifier_erp_mes.txt
│   └── summarizer_ru.txt
├── src/
│   ├── main.py
│   ├── config.py
│   ├── db.py
│   ├── models.py
│   ├── schema.sql
│   ├── feed_poller.py
│   ├── normalizer.py
│   ├── classifier.py
│   ├── summarizer.py
│   ├── telegram_sender.py
│   ├── prompt_manager.py
│   ├── health_monitor.py
│   ├── alert_manager.py
│   └── metrics.py
├── requirements.txt
└── tests/
```
