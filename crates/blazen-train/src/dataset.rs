//! Training datasets.
//!
//! Provides [`JsonlDataset`], a tokenizer + chat-template-aware loader for
//! JSONL files in either the OpenAI `messages` shape or the legacy
//! `prompt`/`completion` shape. The renderer masks every non-completion
//! token in `labels` with `-100` so the training loss is only computed
//! on assistant outputs (SFT convention).

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use minijinja::{Environment, context};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::error::BlazenTrainError;
use crate::trainer::{TrainingBatch, TrainingDataset};

/// PEFT/HF convention: label positions set to this id are skipped by
/// the cross-entropy loss reduction.
pub const DEFAULT_IGNORE_INDEX: i64 = -100;

/// A single chat message in OpenAI shape.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    /// `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Message body.
    pub content: String,
}

#[derive(Debug, Clone)]
enum RawExample {
    Messages(Vec<ChatMessage>),
    PromptCompletion { prompt: String, completion: String },
}

#[derive(Debug, Deserialize)]
struct JsonlRow {
    #[serde(default)]
    messages: Option<Vec<ChatMessage>>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
}

/// JSONL-backed [`TrainingDataset`].
///
/// Each line of the input file must deserialize to either
/// `{"messages": [{"role": ..., "content": ...}, ...]}` (OpenAI) or
/// `{"prompt": "...", "completion": "..."}` (legacy SFT). Both shapes
/// may coexist in the same file.
pub struct JsonlDataset {
    examples: Vec<RawExample>,
    tokenizer: Arc<Tokenizer>,
    chat_env: Option<Arc<Environment<'static>>>,
    max_seq_len: usize,
    device: Device,
    pad_token_id: u32,
    ignore_index: i64,
}

impl JsonlDataset {
    /// Load and parse the JSONL file at `path`.
    ///
    /// `chat_template` is a Jinja2 string straight from a HuggingFace
    /// `tokenizer_config.json`'s `chat_template` field; it must accept a
    /// `messages` variable. If `None`, only `prompt`/`completion` rows
    /// are supported (rows in `messages` shape will fail at batch time).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Dataset`] for I/O failures, JSON parse
    /// failures, or empty files. Returns [`BlazenTrainError::Dataset`]
    /// if `chat_template` is supplied but minijinja rejects it.
    pub fn from_path(
        path: &Path,
        tokenizer: Arc<Tokenizer>,
        chat_template: Option<&str>,
        max_seq_len: usize,
        device: Device,
        pad_token_id: u32,
    ) -> Result<Self, BlazenTrainError> {
        if max_seq_len == 0 {
            return Err(BlazenTrainError::Dataset(
                "max_seq_len must be > 0".to_string(),
            ));
        }

        let text = std::fs::read_to_string(path).map_err(|e| {
            BlazenTrainError::Dataset(format!("failed to read jsonl at {}: {e}", path.display()))
        })?;

        let mut examples = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let row: JsonlRow = serde_json::from_str(trimmed).map_err(|e| {
                BlazenTrainError::Dataset(format!(
                    "jsonl parse error at {}:{}: {e}",
                    path.display(),
                    lineno + 1
                ))
            })?;
            examples.push(row_to_example(row, lineno + 1, path)?);
        }

        if examples.is_empty() {
            return Err(BlazenTrainError::Dataset(format!(
                "jsonl file at {} contains zero examples",
                path.display()
            )));
        }

        let chat_env = if let Some(tpl) = chat_template {
            let mut env = Environment::new();
            env.add_template_owned("chat", tpl.to_string())
                .map_err(|e| {
                    BlazenTrainError::Dataset(format!("chat_template compile failed: {e}"))
                })?;
            Some(Arc::new(env))
        } else {
            None
        };

        Ok(Self {
            examples,
            tokenizer,
            chat_env,
            max_seq_len,
            device,
            pad_token_id,
            ignore_index: DEFAULT_IGNORE_INDEX,
        })
    }

    /// Override the label-mask sentinel id (default `-100`).
    #[must_use]
    pub fn with_ignore_index(mut self, ignore_index: i64) -> Self {
        self.ignore_index = ignore_index;
        self
    }

    fn render(&self, ex: &RawExample) -> Result<(String, String), BlazenTrainError> {
        match ex {
            RawExample::PromptCompletion { prompt, completion } => {
                Ok((prompt.clone(), completion.clone()))
            }
            RawExample::Messages(msgs) => {
                let env = self.chat_env.as_ref().ok_or_else(|| {
                    BlazenTrainError::Dataset(
                        "chat-template messages row encountered but no chat_template was provided"
                            .to_string(),
                    )
                })?;
                let tpl = env.get_template("chat").map_err(|e| {
                    BlazenTrainError::Dataset(format!("chat_template lookup failed: {e}"))
                })?;

                let prompt_msgs: Vec<&ChatMessage> =
                    msgs.iter().take_while(|m| m.role != "assistant").collect();
                if prompt_msgs.len() == msgs.len() {
                    return Err(BlazenTrainError::Dataset(
                        "messages row has no trailing assistant turn to train on".to_string(),
                    ));
                }
                let assistant = msgs
                    .iter()
                    .skip(prompt_msgs.len())
                    .find(|m| m.role == "assistant")
                    .ok_or_else(|| {
                        BlazenTrainError::Dataset(
                            "messages row has no assistant turn to train on".to_string(),
                        )
                    })?;

                let prompt_view: Vec<TemplateMsg> =
                    prompt_msgs.iter().map(|m| TemplateMsg::from(*m)).collect();
                let prompt_render = tpl
                    .render(context! {
                        messages => prompt_view,
                        add_generation_prompt => true,
                    })
                    .map_err(|e| {
                        BlazenTrainError::Dataset(format!(
                            "chat_template render (prompt) failed: {e}"
                        ))
                    })?;

                let full_view: Vec<TemplateMsg> = msgs.iter().map(TemplateMsg::from).collect();
                let full_render = tpl
                    .render(context! {
                        messages => full_view,
                        add_generation_prompt => false,
                    })
                    .map_err(|e| {
                        BlazenTrainError::Dataset(format!(
                            "chat_template render (full) failed: {e}"
                        ))
                    })?;

                let completion = full_render
                    .strip_prefix(&prompt_render)
                    .map_or_else(|| assistant.content.clone(), str::to_string);

                Ok((prompt_render, completion))
            }
        }
    }

    fn tokenize_pair(
        &self,
        prompt: &str,
        completion: &str,
    ) -> Result<(Vec<u32>, usize), BlazenTrainError> {
        let prompt_enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| BlazenTrainError::Dataset(format!("prompt tokenize failed: {e}")))?;
        let full_enc = self
            .tokenizer
            .encode(format!("{prompt}{completion}"), false)
            .map_err(|e| BlazenTrainError::Dataset(format!("full tokenize failed: {e}")))?;

        let prompt_len = prompt_enc.get_ids().len();
        let mut full_ids = full_enc.get_ids().to_vec();

        if full_ids.len() > self.max_seq_len {
            full_ids.truncate(self.max_seq_len);
        }
        let prompt_len = prompt_len.min(full_ids.len());
        Ok((full_ids, prompt_len))
    }
}

#[derive(serde::Serialize)]
struct TemplateMsg<'a> {
    role: &'a str,
    content: &'a str,
}

impl<'a> From<&'a ChatMessage> for TemplateMsg<'a> {
    fn from(m: &'a ChatMessage) -> Self {
        Self {
            role: &m.role,
            content: &m.content,
        }
    }
}

fn row_to_example(
    row: JsonlRow,
    lineno: usize,
    path: &Path,
) -> Result<RawExample, BlazenTrainError> {
    if let Some(msgs) = row.messages {
        if msgs.is_empty() {
            return Err(BlazenTrainError::Dataset(format!(
                "empty messages array at {}:{lineno}",
                path.display()
            )));
        }
        return Ok(RawExample::Messages(msgs));
    }
    match (row.prompt, row.completion) {
        (Some(p), Some(c)) => Ok(RawExample::PromptCompletion {
            prompt: p,
            completion: c,
        }),
        _ => Err(BlazenTrainError::Dataset(format!(
            "row at {}:{lineno} must have either `messages` or both `prompt` and `completion`",
            path.display()
        ))),
    }
}

#[async_trait]
impl TrainingDataset for JsonlDataset {
    fn len(&self) -> usize {
        self.examples.len()
    }

    async fn batch(
        &self,
        batch_size: usize,
        idx: usize,
    ) -> Result<TrainingBatch, BlazenTrainError> {
        if batch_size == 0 {
            return Err(BlazenTrainError::Dataset(
                "batch_size must be > 0".to_string(),
            ));
        }
        if self.examples.is_empty() {
            return Err(BlazenTrainError::Dataset("dataset is empty".to_string()));
        }

        let mut row_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut row_prompt_lens: Vec<usize> = Vec::with_capacity(batch_size);
        let n = self.examples.len();

        for slot in 0..batch_size {
            // Why: cyclic indexing so callers can drive arbitrary epoch counts
            // by ticking `idx` upward without bounds-checking against len().
            let pos = (idx.saturating_mul(batch_size) + slot) % n;
            let ex = &self.examples[pos];
            let (prompt, completion) = self.render(ex)?;
            let (ids, prompt_len) = self.tokenize_pair(&prompt, &completion)?;
            row_ids.push(ids);
            row_prompt_lens.push(prompt_len);
        }

        let max_len = row_ids.iter().map(Vec::len).max().unwrap_or(0).max(1);

        let mut input_ids_flat: Vec<u32> = Vec::with_capacity(batch_size * max_len);
        let mut attn_flat: Vec<u32> = Vec::with_capacity(batch_size * max_len);
        let mut labels_flat: Vec<i64> = Vec::with_capacity(batch_size * max_len);

        for (ids, prompt_len) in row_ids.iter().zip(row_prompt_lens.iter()) {
            let real_len = ids.len();
            for (pos, &tok) in ids.iter().enumerate() {
                input_ids_flat.push(tok);
                attn_flat.push(1);
                if pos < *prompt_len {
                    labels_flat.push(self.ignore_index);
                } else {
                    labels_flat.push(i64::from(tok));
                }
            }
            for _ in real_len..max_len {
                input_ids_flat.push(self.pad_token_id);
                attn_flat.push(0);
                labels_flat.push(self.ignore_index);
            }
        }

        let shape = (batch_size, max_len);
        let input_ids =
            Tensor::from_vec(input_ids_flat, shape, &self.device)?.to_dtype(DType::U32)?;
        let attention_mask =
            Tensor::from_vec(attn_flat, shape, &self.device)?.to_dtype(DType::U32)?;
        let labels = Tensor::from_vec(labels_flat, shape, &self.device)?.to_dtype(DType::I64)?;

        Ok(TrainingBatch {
            input_ids,
            attention_mask,
            labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn build_tokenizer() -> Arc<Tokenizer> {
        let vocab_terms = [
            "[UNK]",
            "hello",
            "world",
            "foo",
            "bar",
            "baz",
            "qux",
            "quux",
            "corge",
            "grault",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "you",
            "are",
            "a",
            "bot",
            "the",
            "answer",
            "is",
            "42",
            "what",
            "meaning",
            "of",
            "life",
        ];
        let vocab: ahash::AHashMap<String, u32> = vocab_terms
            .iter()
            .enumerate()
            .map(|(i, t)| ((*t).to_string(), u32::try_from(i).unwrap()))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tk = Tokenizer::new(model);
        tk.with_pre_tokenizer(Some(Whitespace {}));
        Arc::new(tk)
    }

    fn simple_template() -> &'static str {
        "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
    }

    fn write_jsonl(dir: &TempDir, lines: &[&str]) -> std::path::PathBuf {
        let path = dir.path().join("ds.jsonl");
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    #[tokio::test]
    async fn jsonl_loads_prompt_completion_format() {
        let dir = TempDir::new().unwrap();
        let line = r#"{"prompt": "hello world ", "completion": "foo bar"}"#;
        let path = write_jsonl(&dir, &[line]);

        let tk = build_tokenizer();
        let ds = JsonlDataset::from_path(&path, tk, None, 32, Device::Cpu, 0).unwrap();
        assert_eq!(ds.len(), 1);

        let batch = ds.batch(1, 0).await.unwrap();
        assert_eq!(batch.input_ids.dims().len(), 2);
        assert_eq!(batch.input_ids.dim(0).unwrap(), 1);
    }

    #[tokio::test]
    async fn jsonl_loads_messages_format() {
        let dir = TempDir::new().unwrap();
        let line = r#"{"messages": [{"role":"user","content":"what is the meaning of life"},{"role":"assistant","content":"the answer is 42"}]}"#;
        let path = write_jsonl(&dir, &[line]);

        let tk = build_tokenizer();
        let ds = JsonlDataset::from_path(&path, tk, Some(simple_template()), 64, Device::Cpu, 0)
            .unwrap();
        assert_eq!(ds.len(), 1);
        let batch = ds.batch(1, 0).await.unwrap();
        assert_eq!(batch.input_ids.dim(0).unwrap(), 1);
    }

    #[tokio::test]
    async fn batch_pads_correctly() {
        let dir = TempDir::new().unwrap();
        let lines = [
            r#"{"prompt": "hello ", "completion": "foo"}"#,
            r#"{"prompt": "hello world foo bar baz qux quux ", "completion": "corge grault"}"#,
        ];
        let path = write_jsonl(&dir, &lines);

        let tk = build_tokenizer();
        let ds = JsonlDataset::from_path(&path, tk, None, 64, Device::Cpu, 999).unwrap();
        let batch = ds.batch(2, 0).await.unwrap();
        let (b, seq) = (
            batch.input_ids.dim(0).unwrap(),
            batch.input_ids.dim(1).unwrap(),
        );
        assert_eq!(b, 2);
        assert!(seq >= 2);

        let attn: Vec<Vec<u32>> = batch
            .attention_mask
            .to_dtype(DType::U32)
            .unwrap()
            .to_vec2()
            .unwrap();
        let total_ones: u32 = attn.iter().flatten().sum();
        let total: u32 = u32::try_from(b * seq).unwrap();
        assert!(total_ones < total, "expected at least one padding cell");

        let ids: Vec<Vec<u32>> = batch.input_ids.to_vec2().unwrap();
        for (row_idx, row) in ids.iter().enumerate() {
            for (col_idx, &cell) in row.iter().enumerate() {
                if attn[row_idx][col_idx] == 0 {
                    assert_eq!(cell, 999, "padded cell must equal pad_token_id");
                }
            }
        }
    }

    #[tokio::test]
    async fn batch_masks_prompt_tokens_in_labels() {
        let dir = TempDir::new().unwrap();
        let line = r#"{"prompt": "hello world foo bar ", "completion": "baz qux"}"#;
        let path = write_jsonl(&dir, &[line]);

        let tk = build_tokenizer();
        let ds = JsonlDataset::from_path(&path, tk, None, 64, Device::Cpu, 0).unwrap();
        let batch = ds.batch(1, 0).await.unwrap();

        let labels: Vec<Vec<i64>> = batch.labels.to_vec2().unwrap();
        let masked = labels[0]
            .iter()
            .filter(|&&v| v == DEFAULT_IGNORE_INDEX)
            .count();
        let kept = labels[0]
            .iter()
            .filter(|&&v| v != DEFAULT_IGNORE_INDEX)
            .count();
        assert!(masked > 0, "prompt portion must be masked");
        assert!(kept > 0, "completion portion must be kept");
    }
}
