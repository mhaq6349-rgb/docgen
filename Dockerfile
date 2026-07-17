FROM python:3.12-slim
WORKDIR /app
COPY server.py .
COPY index.html .
EXPOSE 8326
ENV DOCGEN_SECRET=change-me-in-production
ENV DOCGEN_HOST=0.0.0.0
CMD ["python", "server.py"]
