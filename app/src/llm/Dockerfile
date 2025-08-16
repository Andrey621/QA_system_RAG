FROM ollama/ollama

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all

COPY start_ollama.sh /start_ollama.sh

RUN chmod +x /start_ollama.sh

ENTRYPOINT ["/start_ollama.sh"]