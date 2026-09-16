# Черновик. Образ приложения — ТОЛЬКО код поиска/API/evaluate.py.
# Модель эмбеддингов сюда НЕ входит — она в отдельном сервисе TEI
# (см. docker-compose.yml и ARCHITECTURE.md п.2: веса в отдельном
# именованном volume, независимом от этого образа, чтобы правки кода
# никогда не приводили к повторному скачиванию весов).

# Базовый образ: -slim, не -alpine, хоть alpine формально легче.
# Причина: numpy/pandas/rapidfuzz — пакеты с C/C++-расширениями; на Alpine
# (musl libc, не glibc) для них не всегда есть готовые musllinux-колёса под
# конкретную версию, и pip может тихо уйти в сборку из исходников (нужен
# компилятор в образе — размер съедается обратно, плюс медленная и хрупкая
# сборка). -slim гарантированно ставит бинарные wheels под glibc для всех
# наших зависимостей без компиляции — самый лёгкий вариант, который при
# этом надёжно и быстро собирается.
FROM python:3.12.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# python -m uvicorn, а не просто uvicorn — гарантирует, что WORKDIR (/app)
# добавлен в sys.path, иначе "from search.pipeline import ..." в api/main.py
# может не найти пакет search/ в зависимости от того, как разрешается путь
# у голого uvicorn-бинарника.
#
# Дефолтная команда — API-сервис; для evaluate.py переопределяется:
#   docker compose run app python evaluate.py
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
