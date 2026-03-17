# Build Stage
FROM rust:1.85-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a shell script to download the model during build
RUN mkdir -p model && \
    curl -L -o model/config.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json && \
    curl -L -o model/tokenizer.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json && \
    curl -L -o model/model.safetensors https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors

# Copy source code and build
COPY . .
RUN cargo build --release

# Runtime Stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    openssl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary and required files
COPY --from=builder /app/target/release/rag-server .
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/static ./static
COPY --from=builder /app/model ./model

# Set environment variables
ENV PORT=3000
EXPOSE 3000

CMD ["./rag-server"]
