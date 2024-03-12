FROM python:3.10-alpine AS builder

WORKDIR /app
COPY requirements.txt .
RUN set -x \
    && python3 -m venv venv \
    && venv/bin/pip install --no-cache-dir -r requirements.txt \
    && find /venv/lib/python3.*/ -name 'tests' -exec rm -r '{}' + \
    && find /venv/lib/python3.*/ -name '__pycache__' -exec rm -r '{}' + \
    && set +x

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -r requirements.txt
 
# Stage 2
FROM python:3-alpine AS runner
 
WORKDIR /app
 
COPY --from=builder /app/venv venv
COPY . .
 
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV FLASK_APP=app/app.py
 
EXPOSE 8080
 
CMD ["gunicorn", "--bind" , ":8080", "--workers", "2", "app:app"]