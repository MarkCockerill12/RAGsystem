use askama::Template;
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    response::{Html, IntoResponse},
    routing::{get, post},
    Form, Router,
};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::limit::RequestBodyLimitLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const SYSTEM_PROMPT: &str = "You are an agentic RAG (Retrieval-Augmented Generation) system built with Rust (Axum, SQLite). 'RAG' stands for Retrieval-Augmented Generation. This system features semantic chunking and a hybrid search engine. You MUST use the search_documents tool to look up information before answering ANY question. If a question is about the system itself, use this background info. If a question has multiple parts, perform multiple separate searches. After receiving tool results, synthesize a concise and highly accurate answer using ONLY the retrieved context. Do not waste tokens on filler; be extremely precise.";
const PRIMARY_MODEL: &str = "llama-3.3-70b-versatile";
const FALLBACK_MODEL: &str = "llama-4-scout-17b-instruct";

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    status: String,
    files: Vec<String>,
    primary_model: String,
}

#[derive(Template)]
#[template(path = "file_list.html")]
struct FileListTemplate {
    files: Vec<String>,
}

struct EmbeddingModel {
    model: BertModel,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    fn new() -> anyhow::Result<Self> {
        let device = Device::Cpu;
        
        let (config_filename, tokenizer_filename, weights_filename) = if std::path::Path::new("model/config.json").exists() {
            tracing::info!("Loading model from local 'model/' directory...");
            (
                std::path::PathBuf::from("model/config.json"),
                std::path::PathBuf::from("model/tokenizer.json"),
                std::path::PathBuf::from("model/model.safetensors"),
            )
        } else {
            tracing::info!("Downloading model from HuggingFace...");
            let api = Api::new()?;
            let repo = api.repo(Repo::model(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            ));
            (
                repo.get("config.json")?,
                repo.get("tokenizer.json")?,
                repo.get("model.safetensors")?,
            )
        };

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        
        let token_ids = tokens.get_ids();
        let attention_mask = tokens.get_attention_mask();

        let input_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;
        let mask = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;

        let embeddings = self.model.forward(&input_ids, &token_type_ids, Some(&mask))?;

        // Mean Pooling with attention mask
        let mask = mask.to_dtype(embeddings.dtype())?.unsqueeze(2)?; 
        let sum_embeddings = embeddings.broadcast_mul(&mask)?.sum(1)?;
        let sum_mask = mask.sum(1)?;
        let embeddings = sum_embeddings.broadcast_div(&sum_mask)?;
        let embeddings = embeddings.get(0)?;
        
        // L2 Normalization
        let norm = embeddings.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let normalized = (embeddings / (norm as f64))?;
        
        Ok(normalized.to_vec1::<f32>()?)
    }
}

#[derive(Deserialize)]
struct ChatQuery {
    query: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct GroqMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    tool_calls: Option<Vec<GroqToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct GroqToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: GroqFunctionCall,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct GroqFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GroqTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct GroqTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: GroqFunctionDefinition,
}

#[derive(Serialize)]
struct GroqFunctionDefinition {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct GroqResponse {
    #[serde(default)]
    choices: Vec<GroqChoice>,
    #[serde(default)]
    error: Option<GroqError>,
}

#[derive(Deserialize, Debug)]
struct GroqError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    failed_generation: Option<String>,
}

#[derive(Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

struct AppState {
    model: Arc<EmbeddingModel>,
    groq_api_key: String,
    db_pool: Pool<SqliteConnectionManager>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,rag_server=debug"));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(filter)
        .init();

    tracing::info!("Initializing database pool (rag.db)...");
    let manager = SqliteConnectionManager::file("rag.db");
    let db_pool = Pool::new(manager)?;

    // Initialize DB & Enable WAL
    init_db(&db_pool)?;

    let groq_api_key =
        std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY environment variable not set");

    tracing::info!("Loading embedding model...");
    let model = Arc::new(EmbeddingModel::new()?);
    let state = Arc::new(AppState {
        model,
        groq_api_key,
        db_pool,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health_check))
        .route("/upload", post(upload))
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024))
        .route("/chat", post(chat))
        .route("/files", get(get_files))
        .route("/clear", post(clear_index))
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("listening on {}", addr);

    let socket = tokio::net::TcpSocket::new_v4()?;
    socket.set_reuseaddr(true)?;
    socket.bind(addr)?;
    let listener = socket.listen(1024)?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, starting graceful shutdown...");
}

fn init_db(pool: &Pool<SqliteConnectionManager>) -> anyhow::Result<()> {
    let conn = pool.get()?;

    // Enable WAL mode for better concurrency
    conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB
        )",
        [],
    )?;
    Ok(())
}

async fn index(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let files = fetch_indexed_files(&state.db_pool).unwrap_or_default();
    IndexTemplate {
        status: "System Ready".to_string(),
        files,
        primary_model: PRIMARY_MODEL.to_string(),
    }
}

async fn get_files(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let files = fetch_indexed_files(&state.db_pool).unwrap_or_default();
    FileListTemplate { files }
}

async fn clear_index(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.db_pool.get() {
        Ok(conn) => {
            let _ = conn.execute("DELETE FROM chunks", []);
            tracing::info!("System database cleared.");
            (
                [("HX-Trigger", "file-uploaded")],
                Html("<p class='text-yellow-500 text-sm'>Context reset successfully!</p>"),
            ).into_response()
        }
        Err(e) => {
            tracing::error!("Failed to get DB connection for clearing: {}", e);
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Html("<p class='text-red-500 text-sm'>Error: Failed to reset system.</p>"),
            ).into_response()
        }
    }
}

fn fetch_indexed_files(pool: &Pool<SqliteConnectionManager>) -> anyhow::Result<Vec<String>> {
    let conn = pool.get()?;
    let mut stmt = conn.prepare("SELECT DISTINCT filename FROM chunks")?;
    let files = stmt.query_map([], |row| row.get(0))?
        .collect::<Result<Vec<String>, _>>()?;
    Ok(files)
}

async fn health_check() -> impl IntoResponse {
    axum::http::StatusCode::OK
}

async fn upload(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> impl IntoResponse {
    let mut files_count = 0;
    while let Ok(Some(field)) = multipart.next_field().await {
        files_count += 1;
        if files_count > 5 {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Html("<p class='text-red-500'>Error: Maximum 5 files allowed.</p>"),
            )
                .into_response();
        }

        let file_name = field.file_name().unwrap_or("unknown").to_string();
        let ext = file_name
            .split('.')
            .next_back()
            .unwrap_or("")
            .to_lowercase();
        if !["pdf", "txt", "docx"].contains(&ext.as_str()) {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Html(format!(
                    "<p class='text-red-500'>Error: File {} has invalid extension.</p>",
                    file_name
                )),
            )
                .into_response();
        }

        let data = match field.bytes().await {
            Ok(d) => d,
            Err(_) => {
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    Html("<p class='text-red-500'>Error: Failed to read file data.</p>"),
                )
                    .into_response()
            }
        };

        let text = match extract_text(&data, &ext) {
            Ok(t) => t,
            Err(_) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Html(format!(
                        "<p class='text-red-500'>Error: Failed to extract text from {}.</p>",
                        file_name
                    )),
                )
                    .into_response()
            }
        };

        tracing::info!("Extracted {} characters from {}", text.len(), file_name);

        let chunks = chunk_text(&text, 1000, 200);
        tracing::info!("Created {} chunks for {}", chunks.len(), file_name);

        let state_clone = Arc::clone(&state);
        let file_name_clone = file_name.clone();

        // Use spawn_blocking for CPU-intensive embedding and DB operations
        let handle = tokio::task::spawn_blocking(move || {
            store_chunks(
                &state_clone.db_pool,
                &state_clone.model,
                &file_name_clone,
                chunks,
            )
        });

        match handle.await {
            Ok(Ok(_)) => tracing::info!("Successfully indexed {}", file_name),
            Ok(Err(e)) => {
                tracing::error!("Failed to store chunks for {}: {}", file_name, e);
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Html(format!(
                        "<p class='text-red-500'>Error: Failed to store chunks for {}.</p>",
                        file_name
                    )),
                )
                    .into_response();
            }
            Err(e) => {
                tracing::error!("Task join error for {}: {}", file_name, e);
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Html(format!(
                        "<p class='text-red-500'>Error: Internal task error for {}.</p>",
                        file_name
                    )),
                )
                    .into_response();
            }
        }
    }

    (
        [("HX-Trigger", "file-uploaded")],
        Html("<p class='text-green-500 text-sm'>Files uploaded and indexed successfully!</p>"),
    ).into_response()
}

fn store_chunks(
    pool: &Pool<SqliteConnectionManager>,
    model: &EmbeddingModel,
    filename: &str,
    chunks: Vec<String>,
) -> anyhow::Result<()> {
    let mut conn = pool.get()?;
    let tx = conn.transaction()?;

    for chunk in chunks {
        tracing::debug!("Storing chunk ({} chars): {}", chunk.len(), &chunk[..chunk.len().min(50)]);
        let embedding = model.embed(&chunk)?;
        let embedding_bytes = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                embedding.len() * std::mem::size_of::<f32>(),
            )
        };

        tx.execute(
            "INSERT INTO chunks (filename, content, embedding) VALUES (?1, ?2, ?3)",
            (filename, chunk, embedding_bytes),
        )?;
    }

    tx.commit()?;
    Ok(())
}

fn extract_text(data: &[u8], ext: &str) -> anyhow::Result<String> {
    match ext {
        "txt" => Ok(String::from_utf8_lossy(data).to_string()),
        "pdf" => Ok(pdf_extract::extract_text_from_mem(data)?),
        "docx" => {
            use docx_rs::*;
            let doc = read_docx(data).map_err(|e| anyhow::anyhow!("DOCX parse error: {}", e))?;
            let mut text = String::new();

            fn handle_paragraph(p: &Paragraph, out: &mut String) {
                for p_child in &p.children {
                    match p_child {
                        ParagraphChild::Run(r) => {
                            for r_child in &r.children {
                                if let RunChild::Text(t) = r_child {
                                    out.push_str(&t.text);
                                }
                            }
                        }
                        ParagraphChild::Hyperlink(h) => {
                            for h_child in &h.children {
                                if let ParagraphChild::Run(r) = h_child {
                                    for r_child in &r.children {
                                        if let RunChild::Text(t) = r_child {
                                            out.push_str(&t.text);
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                out.push('\n');
            }

            fn handle_table(t: &Table, out: &mut String) {
                for row_child in &t.rows {
                    let TableChild::TableRow(row) = row_child;
                    for cell_child in &row.cells {
                        let TableRowChild::TableCell(cell) = cell_child;
                        for c_child in &cell.children {
                            match c_child {
                                TableCellContent::Paragraph(p) => handle_paragraph(p, out),
                                TableCellContent::Table(inner_t) => handle_table(inner_t, out),
                                _ => {}
                            }
                        }
                        out.push_str(" | ");
                    }
                    out.push('\n');
                }
            }

            for child in doc.document.children {
                match child {
                    DocumentChild::Paragraph(p) => handle_paragraph(&p, &mut text),
                    DocumentChild::Table(t) => handle_table(&t, &mut text),
                    _ => {}
                }
            }
            Ok(text)
        }
        _ => anyhow::bail!("Unsupported extension"),
    }
}

fn chunk_text(text: &str, target_chunk_size: usize, _overlap_size: usize) -> Vec<String> {
    // Phase 5: Split by single newline to keep rows/lines as distinct structural units
    let blocks: Vec<String> = text
        .split('\n')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_length = 0;

    for block in blocks {
        // If adding this block exceeds target size, push current chunk
        if current_length + block.len() > target_chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.join("\n"));
            current_chunk.clear();
            current_length = 0;
        }
        
        // If a single block is still too large (rare with \n split), split it further
        if block.len() > target_chunk_size {
            let sentences: Vec<&str> = block
                .split_inclusive(['.', '!', '?'])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            
            for sentence in sentences {
                if current_length + sentence.len() > target_chunk_size && !current_chunk.is_empty() {
                    chunks.push(current_chunk.join(" "));
                    current_chunk.clear();
                    current_length = 0;
                }
                current_chunk.push(sentence.to_string());
                current_length += sentence.len();
            }
        } else {
            current_chunk.push(block.to_string());
            current_length += block.len();
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join("\n"));
    }

    chunks
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Form(form): Form<ChatQuery>,
) -> impl IntoResponse {
    let mut messages = vec![
        GroqMessage {
            role: "system".to_string(),
            content: Some(SYSTEM_PROMPT.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        GroqMessage {
            role: "user".to_string(),
            content: Some(form.query.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();
    let mut iterations = 0;
    let max_iterations = 5;
    let tool_use_fallback_re = regex::Regex::new(r#"(?i)<function=search_documents[^>]*>\{.*?"query":\s*"([^"]+)"#).unwrap();

    let mut current_model = PRIMARY_MODEL.to_string();
    let mut is_fallback = false;

    tracing::info!("Processing query: '{}'", form.query);

    while iterations < max_iterations {
        iterations += 1;
        
        let mut groq_request = GroqRequest {
            model: current_model.clone(),
            messages: messages.clone(),
            tools: Some(vec![GroqTool {
                tool_type: "function".to_string(),
                function: GroqFunctionDefinition {
                    name: "search_documents".to_string(),
                    description: "Search through uploaded document chunks using semantic similarity.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"]
                    }),
                },
            }]),
            tool_choice: Some(serde_json::json!({"type": "function", "function": {"name": "search_documents"}})),
        };

        // On subsequent calls (after tool results), remove tools and tool_choice
        if messages.iter().any(|m| m.role == "tool") {
            groq_request.tools = None;
            groq_request.tool_choice = None;
        }

        let mut retry_count = 0;
        let max_retries = 3;
        let mut res = None;

        while retry_count <= max_retries {
            let response = client.post("https://api.groq.com/openai/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", state.groq_api_key))
                .json(&groq_request)
                .send()
                .await;

            match response {
                Ok(r) => {
                    // Log Gas Gauge
                    if let Some(tokens) = r.headers().get("x-ratelimit-remaining-tokens") {
                        tracing::info!("Gas Gauge (Tokens): {}", tokens.to_str().unwrap_or("?"));
                    }
                    if let Some(requests) = r.headers().get("x-ratelimit-remaining-requests") {
                        tracing::info!("Gas Gauge (Requests): {}", requests.to_str().unwrap_or("?"));
                    }

                    if r.status().as_u16() == 429 {
                        let body = r.text().await.unwrap_or_default();
                        tracing::warn!("Rate limit hit (429). Body: {}", body);
                        
                        // Check for Daily Limit (TPD/RPD)
                        if body.contains("Daily limit") || body.contains("RPD") || body.contains("TPD") {
                            if !is_fallback {
                                tracing::info!("Daily limit reached. Switching to fallback model: {}", FALLBACK_MODEL);
                                current_model = FALLBACK_MODEL.to_string();
                                groq_request.model = current_model.clone();
                                is_fallback = true;
                                retry_count = 0; // Reset retries for the new model
                                continue;
                            } else {
                                return Html("<div class='bg-red-500/10 border border-red-500/20 p-4 rounded-xl max-w-[80%]'><p class='text-sm'>Error: All models reached daily rate limits.</p></div>").into_response();
                            }
                        }

                        // Check for Momentary Limit (TPM/RPM)
                        if retry_count < max_retries {
                            let wait_secs = if body.contains("retry-after") {
                                // Simple extraction of "retry-after": "X.Xs" or similar
                                if let Some(cap) = regex::Regex::new(r#"retry-after":\s*"([\d.]+)"#).unwrap().captures(&body) {
                                    cap[1].parse::<f32>().unwrap_or(1.0).ceil() as u64
                                } else {
                                    1
                                }
                            } else {
                                1
                            };

                            tracing::info!("Momentary limit hit. Retrying in {}s (Attempt {}/{})", wait_secs, retry_count + 1, max_retries);
                            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
                            retry_count += 1;
                            continue;
                        } else {
                            return Html("<div class='bg-red-500/10 border border-red-500/20 p-4 rounded-xl max-w-[80%]'><p class='text-sm'>Error: Too many retries for momentary rate limit.</p></div>").into_response();
                        }
                    }

                    if !r.status().is_success() {
                        let status = r.status();
                        let body = r.text().await.unwrap_or_default();
                        
                        // Check if this is a tool use failure we can recover from
                        if status.as_u16() == 400 {
                            if let Ok(groq_err) = serde_json::from_str::<GroqResponse>(&body) {
                                if groq_err.error.as_ref().map(|e| e.code.as_deref()) == Some(Some("tool_use_failed")) {
                                    tracing::warn!("Caught 400 tool_use_failed, proceeding to fallback.");
                                    res = Some((status, body)); // Use a tuple to pass body along
                                    break;
                                }
                            }
                        }

                        tracing::error!("API Error ({}): {}", status, body);
                        return Html(format!("<div class='bg-red-500/10 border border-red-500/20 p-4 rounded-xl max-w-[80%]'><p class='text-sm'>Error: API returned status {}. {}</p></div>", status, body)).into_response();
                    }

                    res = Some((r.status(), r.text().await.unwrap_or_default()));
                    break;
                }
                Err(e) => {
                    tracing::error!("Network error: {}", e);
                    return Html("<div class='bg-red-500/10 border border-red-500/20 p-4 rounded-xl max-w-[80%]'><p class='text-sm'>Error: API request failed due to network error.</p></div>").into_response();
                }
            }
        }

        let (status, body) = res.unwrap();
        tracing::debug!("Groq Raw Response ({}): {}", status, body);

        let groq_res: GroqResponse = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to parse Groq response: {}. Body: {}", e, body);
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Html(format!("<div class='bg-red-500/10 border border-red-500/20 p-4 rounded-xl max-w-[80%]'><p class='text-sm text-red-500 font-bold'>Groq API Parse Error: {}</p><pre class='mt-2 text-[10px] text-gray-400 overflow-x-auto'>{}</pre></div>", e, body)),
                ).into_response();
            }
        };

        if let Some(err) = groq_res.error {
            if err.code.as_deref() == Some("tool_use_failed") {
                if let Some(ref failed_gen) = err.failed_generation {
                    if let Some(caps) = tool_use_fallback_re.captures(failed_gen) {
                        let extracted_query = &caps[1];
                        tracing::info!("Fallback: Extracted query from failed_generation: {}", extracted_query);
                        
                        let query_embedding = match state.model.embed(extracted_query) {
                            Ok(e) => e,
                            Err(_) => vec![0.0; 384],
                        };
                        let context = retrieve_context(&state.db_pool, query_embedding, extracted_query).unwrap_or_default();
                        let context_str = context.join("\n---\n");
                        
                        messages.push(GroqMessage {
                            role: "system".to_string(),
                            content: Some(format!("Use the following context from uploaded documents to answer the user's question:\n\n---\n{}\n---\nIf the context does not contain the answer, say you don't have enough information.", context_str)),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                        continue;
                    }
                }
            }
            
            tracing::error!("Groq API returned an error: {:?}", err);
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Html(format!("<div class='bg-red-500/10 border border-red-500/20 p-4 rounded-xl max-w-[80%]'><p class='text-sm text-red-500 font-bold'>Groq API Error: {}</p><p class='text-xs text-gray-400'>Type: {}</p></div>", err.message, err.error_type)),
            ).into_response();
        }

        if groq_res.choices.is_empty() {
             return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Html("<div class='bg-red-500/10 border border-red-500/20 p-4 rounded-xl max-w-[80%]'><p class='text-sm text-red-500 font-bold'>Groq API Error: No choices returned</p></div>".to_string()),
            ).into_response();
        }

        let assistant_msg = groq_res.choices[0].message.clone();
        messages.push(assistant_msg.clone());

        if let Some(tool_calls) = assistant_msg.tool_calls {
            for tool_call in tool_calls {
                if tool_call.function.name == "search_documents" {
                    let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments).unwrap_or_default();
                    let query = args["query"].as_str().unwrap_or("");
                    let query_embedding = match state.model.embed(query) {
                        Ok(e) => e,
                        Err(_) => vec![0.0; 384],
                    };

                    let context = retrieve_context(&state.db_pool, query_embedding, query).unwrap_or_default();
                    let context_str = if context.is_empty() {
                        "No relevant results found.".to_string()
                    } else {
                        context.join("\n---\n")
                    };

                    messages.push(GroqMessage {
                        role: "tool".to_string(),
                        content: Some(context_str),
                        tool_calls: None,
                        tool_call_id: Some(tool_call.id),
                    });
                }
            }
        } else if let Some(content) = assistant_msg.content {
            let escaped_query = escape_html(&form.query);
            let escaped_content = escape_html(&content);
            let model_badge = if is_fallback { "Llama 4 (Fallback)" } else { "Llama 3.3" };
            return Html(format!(
                "<div class='bg-white/5 border border-white/10 p-4 rounded-xl max-w-[90%] ml-auto mb-4'>\
                    <p class='text-xs text-gray-500 mb-1'>You</p>\
                    <p class='text-sm'>{}</p>\
                </div>\
                <div class='bg-blue-600/10 border border-blue-500/20 p-4 rounded-xl max-w-[90%] mb-4'>\
                    <p class='text-xs text-blue-400 mb-1'>Assistant ({})</p>\
                    <div class='text-sm prose prose-invert max-w-none'>{}</div>\
                </div>", escaped_query, model_badge, escaped_content)).into_response();
        } else {
            break;
        }
    }

    Html("<div class='bg-red-500/10 border border-red-500/20 p-4 rounded-xl max-w-[80%]'><p class='text-sm'>Error: LLM failed to provide an answer after multiple steps.</p></div>").into_response()
}

fn retrieve_context(
    pool: &Pool<SqliteConnectionManager>,
    query_embedding: Vec<f32>,
    query_text: &str,
) -> anyhow::Result<Vec<String>> {
    let conn = pool.get()?;
    
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
    tracing::info!("Database has {} chunks available for search.", count);

    let mut stmt = conn.prepare("SELECT content, embedding FROM chunks")?;
    let rows = stmt.query_map([], |row| {
        let content: String = row.get(0)?;
        let embedding_bytes: Vec<u8> = row.get(1)?;
        
        let embedding: Vec<f32> = embedding_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(chunk);
                f32::from_ne_bytes(bytes)
            })
            .collect();

        Ok((content, embedding))
    })?;

    let query_lower = query_text.to_lowercase();
    let mut similarities: Vec<(String, f32)> = Vec::new();

    for row in rows {
        let (content, embedding) = row?;
        let semantic_similarity = cosine_similarity(&query_embedding, &embedding);
        
        // Hybrid Search: Keyword Boosting
        // If the query text (or a significant part of it) appears literally in the chunk, boost it.
        let mut final_score = semantic_similarity;
        if content.to_lowercase().contains(&query_lower) && query_lower.len() > 1 {
            final_score += 0.5; // Significant boost for literal matches
            tracing::debug!("Keyword boost applied (+0.5) for query '{}'", query_text);
        }

        similarities.push((content, final_score));
    }
    
    if !similarities.is_empty() {
        let max_score = similarities.iter().map(|(_, s)| *s).fold(0.0, f32::max);
        tracing::debug!("Found {} similarities. Max score: {:.4}", similarities.len(), max_score);
    }

    // Retain matches with high scores (after boost) or reasonable similarity
    similarities.retain(|(_, score)| *score > 0.05);
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    for (i, (content, score)) in similarities.iter().take(10).enumerate() {
        tracing::debug!("Top result {}: score={:.4}, preview='{}'", i, score, &content[..content.len().min(80)]);
    }

    Ok(similarities.into_iter().take(10).map(|(c, _)| c).collect())
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
     .replace('\'', "&#x27;")
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();
    dot_product / (norm1 * norm2)
}
