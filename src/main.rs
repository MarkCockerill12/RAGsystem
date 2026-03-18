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
use uuid::Uuid;

const SYSTEM_PROMPT: &str = "\
You are a highly intelligent document analysis assistant. Your goal is to answer questions using the provided context from the user's uploaded documents.\n\n\
VITAL INTELLIGENCE RULES:\n\
1. **Implicit Context**: If the user asks about 'the document', 'the txt file', or similar, and you have context from a file, use it. Do not be pedantically restrictive.\n\
2. **Reasoning**: Use your full reasoning capabilities. If a question requires counting (e.g., '8th letter'), find the relevant text block in the context and perform the task step-by-step.\n\
3. **Source Attribution**: Always mention which file your information comes from using the [Source: filename] tag.\n\
4. **Fallback**: If the provided context is insufficient or contains 'NO RELEVANT DOCUMENTS FOUND', but you can see chunks from a relevantly named file, try to answer based on what you have. If you truly cannot find the answer, say: 'The uploaded documents do not appear to contain this specific information.'\n\
5. **Precision**: When asked for specific items, quotes, or counts, be exact. Show your work for complex lookups.\n\n\
SEARCH STRATEGY:\n\
6. Use the `search_documents` tool to find information. If your first search is too narrow, try a broader search.\n\
7. If the user mentions a specific file, prioritize searching in that file.";
const PRIMARY_MODEL: &str = "llama-3.3-70b-versatile";
const FALLBACK_MODEL: &str = "llama-3.1-8b-instant";
const FALLBACK_BRANDING: &str = "Llama Scout";


use std::sync::OnceLock;

fn retry_after_regex() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"try again in ([\d.]+)s").unwrap())
}

fn function_in_content_regex() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(
            r#"(?i)(?:<function|search_documents\s*\(|<tool_call>|\{"name"\s*:\s*"search_documents")"#
        ).unwrap()
    })
}

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
            (std::path::PathBuf::from("model/config.json"), std::path::PathBuf::from("model/tokenizer.json"), std::path::PathBuf::from("model/model.safetensors"))
        } else {
            let api = Api::new()?;
            let repo = api.repo(Repo::model("sentence-transformers/all-MiniLM-L6-v2".to_string()));
            (repo.get("config.json")?, repo.get("tokenizer.json")?, repo.get("model.safetensors")?)
        };
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;
        Ok(Self { model, tokenizer, device })
    }

    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;
        let mask = Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;
        let embeddings = self.model.forward(&input_ids, &token_type_ids, Some(&mask))?;
        let mask = mask.to_dtype(embeddings.dtype())?.unsqueeze(2)?; 
        let sum_embeddings = embeddings.broadcast_mul(&mask)?.sum(1)?;
        let sum_mask = mask.sum(1)?;
        let embeddings = sum_embeddings.broadcast_div(&sum_mask)?.get(0)?;
        let norm = embeddings.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let normalized = (embeddings / (norm as f64))?;
        Ok(normalized.to_vec1::<f32>()?)
    }
}

#[derive(Deserialize)]
struct ChatQuery { query: String }

#[derive(Serialize, Deserialize, Clone, Debug)]
struct GroqMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<GroqToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
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
#[allow(dead_code)]
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
struct GroqChoice { message: GroqMessage }

struct AppState {
    model: Arc<EmbeddingModel>,
    groq_api_key: String,
    db_pool: Pool<SqliteConnectionManager>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,rag_server=debug"));
    tracing_subscriber::registry().with(tracing_subscriber::fmt::layer()).with(filter).init();
    let manager = SqliteConnectionManager::file("rag.db");
    let db_pool = Pool::new(manager)?;
    init_db(&db_pool)?;
    let groq_api_key = std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY environment variable not set");
    let model = Arc::new(EmbeddingModel::new()?);
    let state = Arc::new(AppState { model, groq_api_key, db_pool });
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
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(3000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("listening on {}", addr);
    let socket = tokio::net::TcpSocket::new_v4()?;
    socket.set_reuseaddr(true)?;
    socket.bind(addr)?;
    let listener = socket.listen(1024)?;
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;
    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.expect("failed");
}

fn init_db(pool: &Pool<SqliteConnectionManager>) -> anyhow::Result<()> {
    let conn = pool.get()?;
    conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")?;
    conn.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, filename TEXT NOT NULL, content TEXT NOT NULL, embedding BLOB)", [])?;
    Ok(())
}

async fn index(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let files = fetch_indexed_files(&state.db_pool).unwrap_or_default();
    IndexTemplate { status: "System Ready".to_string(), files, primary_model: PRIMARY_MODEL.to_string() }
}

async fn get_files(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let files = fetch_indexed_files(&state.db_pool).unwrap_or_default();
    FileListTemplate { files }
}
fn fetch_indexed_files(pool: &Pool<SqliteConnectionManager>) -> anyhow::Result<Vec<String>> {
    let conn = pool.get()?;
    let mut stmt = conn.prepare("SELECT DISTINCT filename FROM chunks ORDER BY filename ASC")?;
    let filenames = stmt.query_map([], |row| row.get(0))?
        .collect::<Result<Vec<String>, _>>()?;
    Ok(filenames)
}
async fn clear_index(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let conn = state.db_pool.get().unwrap();
    let _ = conn.execute("DELETE FROM chunks", []);
    ( [("HX-Trigger", "file-uploaded")], Html("<p class='text-yellow-500 text-sm'>Context reset successfully!</p>") ).into_response()
}
async fn health_check() -> impl IntoResponse { axum::http::StatusCode::OK }

async fn upload(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> impl IntoResponse {
    while let Ok(Some(field)) = multipart.next_field().await {
        let file_name = field.file_name().unwrap_or("unknown").to_string();
        let ext = file_name.split('.').next_back().unwrap_or("").to_lowercase();
        let data = field.bytes().await.unwrap();
        let text = extract_text(&data, &ext).unwrap();
        let chunks = chunk_text(&text, 1000, 200);
        let state_clone = Arc::clone(&state);
        let file_name_clone = file_name.clone();
        tokio::task::spawn_blocking(move || {
            store_chunks(&state_clone.db_pool, &state_clone.model, &file_name_clone, chunks)
        }).await.unwrap().unwrap();
    }
    ( [("HX-Trigger", "file-uploaded")], Html("<p class='text-green-500 text-sm'>Files uploaded successfully!</p>") ).into_response()
}

fn store_chunks(
    pool: &Pool<SqliteConnectionManager>,
    model: &EmbeddingModel,
    filename: &str,
    chunks: Vec<String>,
) -> anyhow::Result<()> {
    let mut conn = pool.get()?;
    let tx = conn.transaction()?;

    // Delete existing chunks for this file to prevent duplicates
    tx.execute("DELETE FROM chunks WHERE filename = ?1", [filename])?;

    for chunk in &chunks {
        let embedding = model.embed(chunk)?;
        let embedding_bytes = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                embedding.len() * std::mem::size_of::<f32>(),
            )
        };
        tx.execute(
            "INSERT INTO chunks (filename, content, embedding) VALUES (?1, ?2, ?3)",
            (filename, chunk.as_str(), embedding_bytes),
        )?;
    }

    tx.commit()?;
    tracing::info!("Stored {} chunks for file '{}'", chunks.len(), filename);
    Ok(())
}

fn extract_query_from_failed_generation(failed_gen: &str, fallback_query: &str) -> String {
    // Strategy 1: Find JSON with "query" key
    if let Some(start) = failed_gen.find('{') {
        let mut depth = 0i32;
        let mut end = start;
        for (i, ch) in failed_gen[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 { end = start + i + 1; break; }
                }
                _ => {}
            }
        }
        if depth == 0 {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&failed_gen[start..end]) {
                if let Some(q) = val.get("query").and_then(|v| v.as_str()) {
                    if !q.trim().is_empty() { return q.trim().to_string(); }
                }
                if let Some(args) = val.get("arguments") {
                    if let Some(q) = args.get("query").and_then(|v| v.as_str()) {
                        if !q.trim().is_empty() { return q.trim().to_string(); }
                    }
                    if let Some(args_str) = args.as_str() {
                        if let Ok(inner) = serde_json::from_str::<serde_json::Value>(args_str) {
                            if let Some(q) = inner.get("query").and_then(|v| v.as_str()) {
                                if !q.trim().is_empty() { return q.trim().to_string(); }
                            }
                        }
                    }
                }
            }
        }
    }

    // Strategy 2: "query": "value" pattern
    static QUERY_VALUE_RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = QUERY_VALUE_RE.get_or_init(|| regex::Regex::new(r#""query"\s*:\s*"([^"]+)""#).unwrap());
    if let Some(caps) = re.captures(failed_gen) {
        let q = caps[1].trim();
        if !q.is_empty() { return q.to_string(); }
    }

    // Strategy 3: search_documents(query="value")
    static FUNC_CALL_RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = FUNC_CALL_RE.get_or_init(|| {
        regex::Regex::new(r#"(?i)search_documents\s*\(\s*query\s*=\s*"([^"]+)""#).unwrap()
    });
    if let Some(caps) = re.captures(failed_gen) {
        let q = caps[1].trim();
        if !q.is_empty() { return q.to_string(); }
    }

    // Strategy 4: XML-style <function>content</function>
    static XML_RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = XML_RE.get_or_init(|| {
        regex::Regex::new(r#"(?s)<function[^>]*>(.*?)</function>"#).unwrap()
    });
    if let Some(caps) = re.captures(failed_gen) {
        let inner = caps[1].trim();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(inner) {
            if let Some(q) = val.get("query").and_then(|v| v.as_str()) {
                if !q.trim().is_empty() { return q.trim().to_string(); }
            }
        }
        if !inner.is_empty() && inner.len() < 300 { return inner.to_string(); }
    }

    tracing::warn!(
        "All extraction strategies failed. Input: '{}'. Falling back to user query.",
        &failed_gen[..failed_gen.len().min(200)]
    );
    fallback_query.to_string()
}

fn extract_text(data: &[u8], ext: &str) -> anyhow::Result<String> {
    match ext {
        "txt" | "md" => Ok(String::from_utf8_lossy(data).to_string()),
        "pdf" => Ok(pdf_extract::extract_text_from_mem(data)?),
        "docx" => {
            use docx_rs::*;
            let doc = read_docx(data).unwrap();
            let mut text = String::new();
            fn handle_paragraph(p: &Paragraph, out: &mut String) {
                for p_child in &p.children {
                    match p_child {
                        ParagraphChild::Run(r) => { for r_child in &r.children { if let RunChild::Text(t) = r_child { out.push_str(&t.text); } } }
                        ParagraphChild::Hyperlink(h) => { for h_child in &h.children { if let ParagraphChild::Run(r) = h_child { for r_child in &r.children { if let RunChild::Text(t) = r_child { out.push_str(&t.text); } } } } }
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
                        for c_child in &cell.children { match c_child { TableCellContent::Paragraph(p) => handle_paragraph(p, out), TableCellContent::Table(inner_t) => handle_table(inner_t, out), _ => {} } }
                        out.push_str(" | ");
                    }
                    out.push('\n');
                }
            }
            for child in doc.document.children { match child { DocumentChild::Paragraph(p) => handle_paragraph(&p, &mut text), DocumentChild::Table(t) => handle_table(&t, &mut text), _ => {} } }
            Ok(text)
        }
        _ => anyhow::bail!("Unsupported"),
    }
}

fn chunk_text(text: &str, target_chunk_size: usize, overlap_size: usize) -> Vec<String> {
    let blocks: Vec<String> = text.split('\n')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    
    if blocks.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current_blocks: Vec<String> = Vec::new();
    let mut current_len = 0;

    for block in &blocks {
        if current_len + block.len() > target_chunk_size && !current_blocks.is_empty() {
            chunks.push(current_blocks.join("\n"));

            // Keep trailing blocks that fit within overlap_size
            let mut overlap_blocks = Vec::new();
            let mut overlap_len = 0;
            for b in current_blocks.iter().rev() {
                if overlap_len + b.len() > overlap_size {
                    break;
                }
                overlap_len += b.len();
                overlap_blocks.push(b.clone());
            }
            overlap_blocks.reverse();
            
            current_blocks = overlap_blocks;
            current_len = current_blocks.iter().map(|b| b.len()).sum();
        }
        current_len += block.len();
        current_blocks.push(block.clone());
    }

    if !current_blocks.is_empty() {
        chunks.push(current_blocks.join("\n"));
    }

    chunks
}
async fn chat(State(state): State<Arc<AppState>>, Form(form): Form<ChatQuery>) -> impl IntoResponse {
    let indexed_files = fetch_indexed_files(&state.db_pool).unwrap_or_default();
    let file_list_str = if indexed_files.is_empty() {
        "No files are currently indexed.".to_string()
    } else {
        format!(
            "Currently indexed files: {}",
            indexed_files.iter()
                .map(|f| format!("'{}'", f))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    let dynamic_system_prompt = format!("{}\n\nAVAILABLE FILES:\n{}", SYSTEM_PROMPT, file_list_str);

    let mut messages = vec![
        GroqMessage {
            role: "system".to_string(),
            content: Some(dynamic_system_prompt),
            tool_calls: None,
            tool_call_id: None,
        },
        GroqMessage {
            role: "user".to_string(),
            content: Some(form.query.clone()),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(40))
        .build()
        .unwrap_or_default();

    let mut iterations = 0;
    let mut current_model = PRIMARY_MODEL.to_string();
    let mut is_fallback = false;

    while iterations < 5 {
        iterations += 1;

        let mut groq_request = GroqRequest {
            model: current_model.clone(),
            messages: messages.clone(),
            tools: Some(vec![GroqTool {
                tool_type: "function".to_string(),
                function: GroqFunctionDefinition {
                    name: "search_documents".to_string(),
                    description: "Search through uploaded document chunks using semantic similarity. \
                        Use the 'query' parameter for the search terms. \
                        Use the optional 'filename' parameter to restrict search to a specific file \
                        when the user asks about a particular document.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search keywords or question."
                            },
                            "filename": {
                                "type": "string",
                                "description": "Optional. Specific file to search."
                            }
                        },
                        "required": ["query"]
                    }),
                },
            }]),
            tool_choice: Some(serde_json::json!("auto")),
        };

        // If we already have tool results, don't ask for more tools in this turn
        if messages.iter().any(|m| m.role == "tool") {
            groq_request.tools = None;
            groq_request.tool_choice = None;
        }

        let mut retry_count = 0u32;
        let mut res = None;

        while retry_count <= 10 {
            let response = client.post("https://api.groq.com/openai/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", state.groq_api_key))
                .json(&groq_request)
                .send()
                .await;

            match response {
                Ok(r) => {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();

                    if status.as_u16() == 429 {
                        let body_lower = body.to_lowercase();
                        let is_daily_limit = body_lower.contains("daily")
                            || body_lower.contains("rpd")
                            || body_lower.contains("tpd");

                        if is_daily_limit && !is_fallback {
                            tracing::warn!(
                                "Daily limit on {}. Switching to {}",
                                current_model, FALLBACK_MODEL
                            );
                            current_model = FALLBACK_MODEL.to_string();
                            groq_request.model = current_model.clone();
                            is_fallback = true;
                            retry_count = 0;
                            continue;
                        }

                        if is_daily_limit && is_fallback {
                            return error_response(
                                &form.query,
                                "Both models have exceeded daily rate limits. Please try again tomorrow."
                            ).into_response();
                        }

                        if retry_count < 10 {
                            let base_wait_secs = retry_after_regex()
                                .captures(&body)
                                .and_then(|cap| cap[1].parse::<f64>().ok())
                                .map(|w| w.ceil().max(1.0))
                                .unwrap_or_else(|| (2.0f64).powi(retry_count.min(3) as i32));

                            let jitter_ms = (std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .subsec_millis() % 500) as u64;

                            let wait_ms = (base_wait_secs * 1000.0) as u64 + jitter_ms;

                            tracing::info!(
                                "Rate limited. Retry {}/10 in {}ms (model: {})",
                                retry_count + 1, wait_ms, current_model
                            );
                            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
                            retry_count += 1;
                            continue;
                        }

                        return error_response(
                            &form.query,
                            "Rate limit exceeded after maximum retries. Please wait a moment."
                        ).into_response();
                    }

                    if !status.is_success() {
                        if status.as_u16() == 400 && body.contains("tool_use_failed") {
                            tracing::warn!("Caught tool_use_failed, entering recovery.");
                            res = Some((status, body));
                            break;
                        }
                        tracing::error!("Groq error {}: {}", status, &body[..body.len().min(500)]);
                        return error_response(
                            &form.query,
                            &format!("API returned error status {}.", status)
                        ).into_response();
                    }

                    res = Some((status, body));
                    break;
                }
                Err(e) => {
                    if retry_count < 3 {
                        tracing::warn!("Network error (attempt {}): {}", retry_count + 1, e);
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        retry_count += 1;
                        continue;
                    }
                    return error_response(
                        &form.query,
                        "Network error connecting to the AI service. Please try again."
                    ).into_response();
                }
            }
        }

        let (status, body) = match res {
            Some(v) => v,
            None => break,
        };

        let groq_res: GroqResponse = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    "Failed to parse Groq response: {}. Status: {}. Body: {}",
                    e, status, &body[..body.len().min(500)]
                );
                return error_response(&form.query, "Failed to parse API response.").into_response();
            }
        };

        if groq_res.error.is_none() && groq_res.choices.is_empty() {
            tracing::error!("Empty Groq response. Body: {}", &body[..body.len().min(500)]);
            return error_response(&form.query, "Received empty response from API.").into_response();
        }

        if let Some(err) = groq_res.error {
            if err.code.as_deref() == Some("tool_use_failed") {
                if let Some(ref fg) = err.failed_generation {
                    tracing::warn!("Failed generation content: '{}'", fg);
                }
                let search_query = err.failed_generation.as_deref()
                    .map(|fg| extract_query_from_failed_generation(fg, &form.query))
                    .unwrap_or_else(|| form.query.clone());

                tracing::warn!("tool_use_failed recovered. Extracted query: '{}'", search_query);

                let query_embedding = match state.model.embed(&search_query) {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::error!("Embedding failed: {}", e);
                        return error_response(&form.query, "Search processing failed.").into_response();
                    }
                };

                let context_budget = if is_fallback { 6000 } else { 20000 };
                let context = retrieve_context(
                    &state.db_pool, query_embedding, &search_query, None, context_budget
                ).unwrap_or_default();

                let fake_tool_call_id = format!("call_{}", Uuid::new_v4());

                messages.push(GroqMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![GroqToolCall {
                        id: fake_tool_call_id.clone(),
                        tool_type: "function".to_string(),
                        function: GroqFunctionCall {
                            name: "search_documents".to_string(),
                            arguments: serde_json::json!({"query": search_query}).to_string(),
                        },
                    }]),
                    tool_call_id: None,
                });

                messages.push(GroqMessage {
                    role: "tool".to_string(),
                    content: Some(context.join("\n\n")),
                    tool_calls: None,
                    tool_call_id: Some(fake_tool_call_id),
                });

                continue;
            }

            tracing::error!("Groq API error: {:?}", err);
            return error_response(&form.query, &format!("API error: {}", err.message)).into_response();
        }

        let assistant_msg = groq_res.choices[0].message.clone();
        messages.push(assistant_msg.clone());

        if let Some(tool_calls) = assistant_msg.tool_calls {
            for tool_call in tool_calls {
                let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
                    .unwrap_or_default();
                let query = args["query"].as_str().unwrap_or(&form.query);
                let filename_filter = args["filename"].as_str();

                let query_embedding = match state.model.embed(query) {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::error!("Embedding error: {}", e);
                        return error_response(&form.query, "Failed to process search query.").into_response();
                    }
                };

                let context_budget = if is_fallback { 6000 } else { 20000 };
                let context = retrieve_context(
                    &state.db_pool,
                    query_embedding,
                    query,
                    filename_filter,
                    context_budget,
                ).unwrap_or_default();

                messages.push(GroqMessage {
                    role: "tool".to_string(),
                    content: Some(context.join("\n\n")),
                    tool_calls: None,
                    tool_call_id: Some(tool_call.id),
                });
            }
        } else if let Some(content) = assistant_msg.content {
            if function_in_content_regex().is_match(&content) {
                let extracted_query = extract_query_from_failed_generation(&content, &form.query);

                let query_embedding = match state.model.embed(&extracted_query) {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::error!("Embedding failed: {}", e);
                        return error_response(&form.query, "Search processing failed.").into_response();
                    }
                };

                let context_budget = if is_fallback { 6000 } else { 20000 };
                let context = retrieve_context(
                    &state.db_pool, query_embedding, &extracted_query, None, context_budget
                ).unwrap_or_default();

                messages.pop();

                let fake_tool_call_id = format!("call_{}", Uuid::new_v4());

                messages.push(GroqMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![GroqToolCall {
                        id: fake_tool_call_id.clone(),
                        tool_type: "function".to_string(),
                        function: GroqFunctionCall {
                            name: "search_documents".to_string(),
                            arguments: serde_json::json!({"query": extracted_query}).to_string(),
                        },
                    }]),
                    tool_call_id: None,
                });

                messages.push(GroqMessage {
                    role: "tool".to_string(),
                    content: Some(context.join("\n\n")),
                    tool_calls: None,
                    tool_call_id: Some(fake_tool_call_id),
                });

                continue;
            }

            // FORCE SEARCH HEURISTIC: If the first iteration produces NO tool calls 
            // and NO results were retrieved, but the query is document-related, force a search.
            if iterations == 1 && !messages.iter().any(|m| m.role == "tool") {
                let q_lower = form.query.to_lowercase();
                let doc_keywords = ["what", "how", "find", "search", "document", "file", "index", "section", "item", "letter", "word"];
                if doc_keywords.iter().any(|k| q_lower.contains(k)) {
                    tracing::info!("Heuristic triggered: forcing search for query '{}'", form.query);
                    let query_embedding = match state.model.embed(&form.query) {
                        Ok(e) => e,
                        Err(_) => vec![0.0; 384],
                    };
                    let context = retrieve_context(&state.db_pool, query_embedding, &form.query, None, 20000).unwrap_or_default();
                    
                    let fake_id = format!("call_{}", Uuid::new_v4());
                    messages.push(GroqMessage {
                        role: "assistant".to_string(),
                        content: None,
                        tool_calls: Some(vec![GroqToolCall {
                            id: fake_id.clone(),
                            tool_type: "function".to_string(),
                            function: GroqFunctionCall {
                                name: "search_documents".to_string(),
                                arguments: serde_json::json!({"query": form.query}).to_string(),
                            },
                        }]),
                        tool_call_id: None,
                    });
                    messages.push(GroqMessage {
                        role: "tool".to_string(),
                        content: Some(context.join("\n\n")),
                        tool_calls: None,
                        tool_call_id: Some(fake_id),
                    });
                    continue;
                }
            }

            let assistant_name = if is_fallback { FALLBACK_BRANDING } else { "Llama 3.3" };
            return Html(format!(
                r#"<div class="flex flex-col gap-4 w-full">
                    <div class="flex justify-end">
                        <div class="bg-blue-600/20 border border-blue-500/30 p-3 rounded-2xl rounded-tr-none px-4 max-w-[85%]">
                            <p class="text-[10px] font-bold text-blue-400 mb-1 uppercase tracking-widest text-right">You</p>
                            <p class="text-sm text-blue-100 pr-1">{}</p>
                        </div>
                    </div>
                    <div class="flex justify-start">
                        <div class="bg-white/5 border border-white/10 p-4 rounded-2xl rounded-tl-none max-w-[85%] shadow-xl">
                            <p class="text-[10px] font-bold text-gray-400 mb-2 uppercase tracking-widest italic">Assistant ({})</p>
                            <p class="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">{}</p>
                        </div>
                    </div>
                </div>"#,
                escape_html(&form.query),
                assistant_name,
                escape_html(content.trim())
            )).into_response();
        }
    }
    error_response(&form.query, "TIMEOUT: Maximum iterations reached without a final answer.").into_response()
}

fn retrieve_context(
    pool: &Pool<SqliteConnectionManager>,
    query_embedding: Vec<f32>,
    query_text: &str,
    filter_filename: Option<&str>,
    max_context_chars: usize,
) -> anyhow::Result<Vec<String>> {
    let conn = pool.get()?;
    let mut stmt = conn.prepare("SELECT filename, content, embedding FROM chunks")?;
    let rows = stmt.query_map([], |row| {
        let f: String = row.get(0)?;
        let c: String = row.get(1)?;
        let b: Vec<u8> = row.get(2)?;
        let e: Vec<f32> = b.chunks_exact(4)
            .map(|ch| {
                let mut bs = [0u8; 4];
                bs.copy_from_slice(ch);
                f32::from_ne_bytes(bs)
            })
            .collect();
        Ok((f, c, e))
    })?;

    let query_lower = query_text.to_lowercase();
    let query_tokens: Vec<String> = query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    // Identify implicit file type mentions to boost relevant documents
    let mentions_txt = query_lower.contains("txt") || query_lower.contains("text file");
    let mentions_pdf = query_lower.contains("pdf");
    let mentions_docx = query_lower.contains("docx") || query_lower.contains("word");
    let mentions_md = query_lower.contains("md") || query_lower.contains("markdown");

    tracing::info!(
        "Searching with query: '{}', tokens: {:?}, file_filter: {:?}",
        query_text, query_tokens, filter_filename
    );

    let mut scored: Vec<(String, String, f32)> = Vec::new();

    for row in rows {
        let (filename, content, embedding) = row?;
        let filename_lower = filename.to_lowercase();

        // If a filename filter is specified, skip chunks from other files
        if let Some(filter) = filter_filename {
            if !filename_lower.contains(&filter.to_lowercase()) {
                continue;
            }
        }

        let semantic_score = cosine_similarity(&query_embedding, &embedding);
        let content_lower = content.to_lowercase();

        // Keyword boost: enhanced for multiple matches
        let mut keyword_boost: f32 = 0.0;
        for token in &query_tokens {
            if content_lower.contains(token) {
                if token.len() <= 3
                    && token.chars().any(|c| c.is_numeric())
                    && token.chars().any(|c| c.is_alphabetic())
                {
                    let pattern = format!(r"(?i)\b{}\b", regex::escape(token));
                    if let Ok(re) = regex::Regex::new(&pattern) {
                        if re.is_match(&content) { keyword_boost += 0.35; }
                    }
                } else if token.len() >= 3 {
                    keyword_boost += 0.15;
                }
            }
        }

        // Implicit file type boost: if the query mentions "txt" and this is a .txt file
        let mut intent_boost = 0.0;
        if (mentions_txt && filename_lower.contains(".txt")) ||
           (mentions_pdf && filename_lower.contains(".pdf")) ||
           (mentions_docx && filename_lower.contains(".docx")) ||
           (mentions_md && filename_lower.contains(".md")) {
            intent_boost = 0.4;
        }

        // Cap keyword boost so it doesn't mask semantic relevance entirely but still helps a lot
        keyword_boost = keyword_boost.min(0.6);
        let final_score = semantic_score + keyword_boost + intent_boost;

        // Dynamic threshold: lower for targeted files, generally more permissive (0.15)
        let threshold = if filter_filename.is_some() || intent_boost > 0.0 { 0.05 } else { 0.15 };

        if final_score > threshold {
            scored.push((filename, content, final_score));
        }
    }

    // Sort by score descending
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Budget-aware selection
    let mut results = Vec::new();
    let mut total_chars = 0;

    for (i, (filename, content, score)) in scored.iter().enumerate() {
        let formatted = format!(
            "--- CHUNK {} [Source: {}] [Relevance: {:.0}%] ---\n{}",
            i + 1,
            filename,
            (score.min(1.0)) * 100.0,
            content
        );

        if total_chars + formatted.len() > max_context_chars && !results.is_empty() {
            break;
        }

        total_chars += formatted.len();
        results.push(formatted);

        if results.len() >= 12 { break; }
    }

    if results.is_empty() {
        tracing::warn!("No chunks passed threshold for query: '{}'", query_text);
        results.push("NO RELEVANT DOCUMENTS FOUND. Please try rephrasing or check if files are uploaded.".to_string());
    }

    tracing::info!(
        "Returning {} chunks ({} chars) for query '{}'",
        results.len(), total_chars, query_text
    );

    Ok(results)
}

fn error_response(user_query: &str, error_msg: &str) -> impl IntoResponse {
    Html(format!(
        r#"<div class="flex flex-col gap-4 w-full">
            <div class="flex justify-end">
                <div class="bg-blue-600/20 border border-blue-500/30 p-3 rounded-2xl rounded-tr-none px-4 max-w-[85%]">
                    <p class="text-[10px] font-bold text-blue-400 mb-1 uppercase tracking-widest text-right">You</p>
                    <p class="text-sm text-blue-100 pr-1">{}</p>
                </div>
            </div>
            <div class="flex justify-start">
                <div class="bg-red-500/10 border border-red-500/20 p-4 rounded-2xl rounded-tl-none max-w-[85%] shadow-xl">
                    <p class="text-[10px] font-bold text-red-400 mb-2 uppercase tracking-widest">System Error</p>
                    <p class="text-sm text-red-300 leading-relaxed">{}</p>
                </div>
            </div>
        </div>"#,
        escape_html(user_query),
        escape_html(error_msg)
    ))
}

fn escape_html(s: &str) -> String { s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#x27;") }

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let n1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let n2: f32 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();
    dot / (n1 * n2)
}
