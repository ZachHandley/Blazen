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
use crate::trainer::{
    KtoBatch, PreferenceBatch, PreferenceDataset, RatedDataset, TrainingBatch, TrainingDataset,
};

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
        let (full_ids, labels) = tokenize_prompt_completion(
            prompt,
            completion,
            &self.tokenizer,
            self.max_seq_len,
            self.ignore_index,
        )?;
        // SFT path needs prompt_len for its label-masking loop; recover it
        // from the labels vector (count leading ignore-index sentinels).
        let prompt_len = labels
            .iter()
            .take_while(|&&v| v == self.ignore_index)
            .count();
        Ok((full_ids, prompt_len))
    }
}

/// Tokenize a `(prompt, completion)` pair into a single id sequence and a
/// parallel SFT-style label vector (prompt positions set to `ignore_index`,
/// completion positions carry the real token id as `i64`).
///
/// Both sequences are truncated to `max_seq_len`. Special tokens are not
/// added (`add_special_tokens = false`) so callers control BOS/EOS via the
/// chat template / prompt string itself.
///
/// # Errors
///
/// Returns [`BlazenTrainError::Dataset`] if either tokenizer call fails.
pub fn tokenize_prompt_completion(
    prompt: &str,
    completion: &str,
    tokenizer: &Tokenizer,
    max_seq_len: usize,
    ignore_index: i64,
) -> Result<(Vec<u32>, Vec<i64>), BlazenTrainError> {
    let prompt_enc = tokenizer
        .encode(prompt, false)
        .map_err(|e| BlazenTrainError::Dataset(format!("prompt tokenize failed: {e}")))?;
    let full_enc = tokenizer
        .encode(format!("{prompt}{completion}"), false)
        .map_err(|e| BlazenTrainError::Dataset(format!("full tokenize failed: {e}")))?;

    let prompt_len = prompt_enc.get_ids().len();
    let mut full_ids = full_enc.get_ids().to_vec();

    if full_ids.len() > max_seq_len {
        full_ids.truncate(max_seq_len);
    }
    let prompt_len = prompt_len.min(full_ids.len());

    let labels: Vec<i64> = full_ids
        .iter()
        .enumerate()
        .map(|(i, &tok)| {
            if i < prompt_len {
                ignore_index
            } else {
                i64::from(tok)
            }
        })
        .collect();

    Ok((full_ids, labels))
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

// -----------------------------------------------------------------------------
// Preference-pair datasets (DPO / ORPO / SimPO)
// -----------------------------------------------------------------------------

/// A single preference-pair example.
///
/// Either `prompt` (legacy SFT-style string) or `messages` (OpenAI chat
/// shape, rendered through the chat template) supplies the prompt prefix;
/// `chosen` / `rejected` are the two assistant responses being compared.
#[derive(Debug, Clone, Deserialize)]
pub struct PreferenceExample {
    /// Plain-text prompt prefix.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Chat messages forming the prompt prefix (mutually exclusive with `prompt`).
    #[serde(default)]
    pub messages: Option<Vec<ChatMessage>>,
    /// The preferred completion.
    pub chosen: String,
    /// The dispreferred completion.
    pub rejected: String,
}

#[derive(Debug, Deserialize)]
struct PreferenceJsonlRow {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    messages: Option<Vec<ChatMessage>>,
    chosen: String,
    rejected: String,
}

/// JSONL-backed preference-pair dataset for DPO / ORPO / SimPO.
///
/// Each line must deserialize to `{"prompt": "...", "chosen": "...",
/// "rejected": "..."}` or `{"messages": [...], "chosen": "...",
/// "rejected": "..."}` where `messages` supplies the prompt prefix and is
/// rendered through `chat_template`.
pub struct PreferenceJsonlDataset {
    examples: Vec<PreferenceExample>,
    tokenizer: Arc<Tokenizer>,
    chat_env: Option<Arc<Environment<'static>>>,
    max_seq_len: usize,
    device: Device,
    pad_token_id: u32,
    ignore_index: i64,
}

impl PreferenceJsonlDataset {
    /// Load and parse a preference-pair JSONL file at `path`.
    ///
    /// `chat_template` is the same Jinja2 string accepted by
    /// [`JsonlDataset::from_path`]; if omitted, only `prompt`/`chosen`/`rejected`
    /// rows are supported.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Dataset`] for I/O failures, JSON parse
    /// failures, an empty file, or a chat-template compile failure.
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
            BlazenTrainError::Dataset(format!(
                "failed to read preference jsonl at {}: {e}",
                path.display()
            ))
        })?;

        let mut examples: Vec<PreferenceExample> = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let row: PreferenceJsonlRow = serde_json::from_str(trimmed).map_err(|e| {
                BlazenTrainError::Dataset(format!(
                    "preference jsonl parse error at {}:{}: {e}",
                    path.display(),
                    lineno + 1
                ))
            })?;
            if row.prompt.is_none() && row.messages.is_none() {
                return Err(BlazenTrainError::Dataset(format!(
                    "preference row at {}:{} must have either `prompt` or `messages`",
                    path.display(),
                    lineno + 1
                )));
            }
            if let Some(msgs) = row.messages.as_ref()
                && msgs.is_empty()
            {
                return Err(BlazenTrainError::Dataset(format!(
                    "empty messages array at {}:{}",
                    path.display(),
                    lineno + 1
                )));
            }
            examples.push(PreferenceExample {
                prompt: row.prompt,
                messages: row.messages,
                chosen: row.chosen,
                rejected: row.rejected,
            });
        }

        if examples.is_empty() {
            return Err(BlazenTrainError::Dataset(format!(
                "preference jsonl file at {} contains zero examples",
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

    /// Number of preference examples in the dataset.
    #[must_use]
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether the dataset contains zero examples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    fn render_prompt(&self, ex: &PreferenceExample) -> Result<String, BlazenTrainError> {
        if let Some(p) = &ex.prompt {
            return Ok(p.clone());
        }
        let msgs = ex.messages.as_ref().ok_or_else(|| {
            BlazenTrainError::Dataset(
                "preference example missing both `prompt` and `messages`".to_string(),
            )
        })?;
        let env = self.chat_env.as_ref().ok_or_else(|| {
            BlazenTrainError::Dataset(
                "preference messages row encountered but no chat_template was provided".to_string(),
            )
        })?;
        let tpl = env
            .get_template("chat")
            .map_err(|e| BlazenTrainError::Dataset(format!("chat_template lookup failed: {e}")))?;
        let view: Vec<TemplateMsg> = msgs.iter().map(TemplateMsg::from).collect();
        tpl.render(context! {
            messages => view,
            add_generation_prompt => true,
        })
        .map_err(|e| {
            BlazenTrainError::Dataset(format!("chat_template render (prompt) failed: {e}"))
        })
    }

    /// Build the `idx`-th preference batch of size `batch_size`.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Dataset`] for indexing / tokenization
    /// failures and [`BlazenTrainError`] from underlying tensor ops.
    pub fn batch(
        &self,
        batch_size: usize,
        idx: usize,
    ) -> Result<PreferenceBatch, BlazenTrainError> {
        if batch_size == 0 {
            return Err(BlazenTrainError::Dataset(
                "batch_size must be > 0".to_string(),
            ));
        }
        if self.examples.is_empty() {
            return Err(BlazenTrainError::Dataset(
                "preference dataset is empty".to_string(),
            ));
        }

        let n = self.examples.len();
        let mut chosen_rows: Vec<(Vec<u32>, Vec<i64>)> = Vec::with_capacity(batch_size);
        let mut rejected_rows: Vec<(Vec<u32>, Vec<i64>)> = Vec::with_capacity(batch_size);

        for slot in 0..batch_size {
            let pos = (idx.saturating_mul(batch_size) + slot) % n;
            let ex = &self.examples[pos];
            let prompt = self.render_prompt(ex)?;
            chosen_rows.push(tokenize_prompt_completion(
                &prompt,
                &ex.chosen,
                &self.tokenizer,
                self.max_seq_len,
                self.ignore_index,
            )?);
            rejected_rows.push(tokenize_prompt_completion(
                &prompt,
                &ex.rejected,
                &self.tokenizer,
                self.max_seq_len,
                self.ignore_index,
            )?);
        }

        let (chosen_input_ids, chosen_attn, chosen_labels) = stack_padded(
            &chosen_rows,
            self.pad_token_id,
            self.ignore_index,
            &self.device,
        )?;
        let (rejected_input_ids, rejected_attn, rejected_labels) = stack_padded(
            &rejected_rows,
            self.pad_token_id,
            self.ignore_index,
            &self.device,
        )?;

        Ok(PreferenceBatch {
            chosen_input_ids,
            chosen_labels,
            chosen_attn,
            rejected_input_ids,
            rejected_labels,
            rejected_attn,
        })
    }
}

// -----------------------------------------------------------------------------
// Rated single-completion datasets (KTO)
// -----------------------------------------------------------------------------

/// A single KTO example: a `(prompt, completion)` pair with a binary
/// desirability label.
#[derive(Debug, Clone, Deserialize)]
pub struct RatedExample {
    /// Plain-text prompt prefix.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Chat messages forming the prompt prefix (mutually exclusive with `prompt`).
    #[serde(default)]
    pub messages: Option<Vec<ChatMessage>>,
    /// The assistant response being rated.
    pub completion: String,
    /// `true` = desirable, `false` = undesirable.
    pub label: bool,
}

#[derive(Debug, Deserialize)]
struct RatedJsonlRow {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    messages: Option<Vec<ChatMessage>>,
    completion: String,
    label: bool,
}

/// JSONL-backed rated dataset for KTO.
///
/// Each line must deserialize to `{"prompt": "...", "completion": "...",
/// "label": true|false}` or `{"messages": [...], "completion": "...",
/// "label": ...}`.
pub struct RatedJsonlDataset {
    examples: Vec<RatedExample>,
    tokenizer: Arc<Tokenizer>,
    chat_env: Option<Arc<Environment<'static>>>,
    max_seq_len: usize,
    device: Device,
    pad_token_id: u32,
    ignore_index: i64,
}

impl RatedJsonlDataset {
    /// Load and parse a rated JSONL file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Dataset`] for I/O failures, JSON parse
    /// failures, an empty file, or a chat-template compile failure.
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
            BlazenTrainError::Dataset(format!(
                "failed to read rated jsonl at {}: {e}",
                path.display()
            ))
        })?;

        let mut examples: Vec<RatedExample> = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let row: RatedJsonlRow = serde_json::from_str(trimmed).map_err(|e| {
                BlazenTrainError::Dataset(format!(
                    "rated jsonl parse error at {}:{}: {e}",
                    path.display(),
                    lineno + 1
                ))
            })?;
            if row.prompt.is_none() && row.messages.is_none() {
                return Err(BlazenTrainError::Dataset(format!(
                    "rated row at {}:{} must have either `prompt` or `messages`",
                    path.display(),
                    lineno + 1
                )));
            }
            if let Some(msgs) = row.messages.as_ref()
                && msgs.is_empty()
            {
                return Err(BlazenTrainError::Dataset(format!(
                    "empty messages array at {}:{}",
                    path.display(),
                    lineno + 1
                )));
            }
            examples.push(RatedExample {
                prompt: row.prompt,
                messages: row.messages,
                completion: row.completion,
                label: row.label,
            });
        }

        if examples.is_empty() {
            return Err(BlazenTrainError::Dataset(format!(
                "rated jsonl file at {} contains zero examples",
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

    /// Number of rated examples in the dataset.
    #[must_use]
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether the dataset contains zero examples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    fn render_prompt(&self, ex: &RatedExample) -> Result<String, BlazenTrainError> {
        if let Some(p) = &ex.prompt {
            return Ok(p.clone());
        }
        let msgs = ex.messages.as_ref().ok_or_else(|| {
            BlazenTrainError::Dataset(
                "rated example missing both `prompt` and `messages`".to_string(),
            )
        })?;
        let env = self.chat_env.as_ref().ok_or_else(|| {
            BlazenTrainError::Dataset(
                "rated messages row encountered but no chat_template was provided".to_string(),
            )
        })?;
        let tpl = env
            .get_template("chat")
            .map_err(|e| BlazenTrainError::Dataset(format!("chat_template lookup failed: {e}")))?;
        let view: Vec<TemplateMsg> = msgs.iter().map(TemplateMsg::from).collect();
        tpl.render(context! {
            messages => view,
            add_generation_prompt => true,
        })
        .map_err(|e| {
            BlazenTrainError::Dataset(format!("chat_template render (prompt) failed: {e}"))
        })
    }

    /// Build the `idx`-th rated batch of size `batch_size`.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Dataset`] for indexing / tokenization
    /// failures and [`BlazenTrainError`] from underlying tensor ops.
    pub fn batch(&self, batch_size: usize, idx: usize) -> Result<KtoBatch, BlazenTrainError> {
        if batch_size == 0 {
            return Err(BlazenTrainError::Dataset(
                "batch_size must be > 0".to_string(),
            ));
        }
        if self.examples.is_empty() {
            return Err(BlazenTrainError::Dataset(
                "rated dataset is empty".to_string(),
            ));
        }

        let n = self.examples.len();
        let mut rows: Vec<(Vec<u32>, Vec<i64>)> = Vec::with_capacity(batch_size);
        let mut desirability: Vec<f32> = Vec::with_capacity(batch_size);

        for slot in 0..batch_size {
            let pos = (idx.saturating_mul(batch_size) + slot) % n;
            let ex = &self.examples[pos];
            let prompt = self.render_prompt(ex)?;
            rows.push(tokenize_prompt_completion(
                &prompt,
                &ex.completion,
                &self.tokenizer,
                self.max_seq_len,
                self.ignore_index,
            )?);
            desirability.push(if ex.label { 1.0 } else { 0.0 });
        }

        let (input_ids, attn, labels) =
            stack_padded(&rows, self.pad_token_id, self.ignore_index, &self.device)?;
        let desirable_mask =
            Tensor::from_vec(desirability, (batch_size,), &self.device)?.to_dtype(DType::F32)?;

        Ok(KtoBatch {
            input_ids,
            labels,
            attn,
            desirable_mask,
        })
    }
}

#[async_trait]
impl PreferenceDataset for PreferenceJsonlDataset {
    fn len(&self) -> usize {
        self.examples.len()
    }

    async fn batch(
        &self,
        batch_size: usize,
        idx: usize,
    ) -> Result<PreferenceBatch, BlazenTrainError> {
        self.batch(batch_size, idx)
    }
}

#[async_trait]
impl RatedDataset for RatedJsonlDataset {
    fn len(&self) -> usize {
        self.examples.len()
    }

    async fn batch(&self, batch_size: usize, idx: usize) -> Result<KtoBatch, BlazenTrainError> {
        self.batch(batch_size, idx)
    }
}

// -----------------------------------------------------------------------------
// Prompt-only datasets (GRPO sampling)
// -----------------------------------------------------------------------------

/// A single GRPO prompt: just the prompt text (plain or chat-shaped). The
/// completions are sampled at train time by the GRPO trainer's policy.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptExample {
    /// Plain-text prompt prefix.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Chat messages forming the prompt prefix (mutually exclusive with `prompt`).
    #[serde(default)]
    pub messages: Option<Vec<ChatMessage>>,
}

#[derive(Debug, Deserialize)]
struct PromptJsonlRow {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    messages: Option<Vec<ChatMessage>>,
}

/// JSONL-backed prompt-only dataset for GRPO.
///
/// Each line must deserialize to `{"prompt": "..."}` or
/// `{"messages": [...]}`. The reader tokenizes lazily — only the raw rows
/// are held in memory. Tokenization (and chat-template rendering, if a
/// template was supplied) happens inside [`Self::prompt_text`], called by
/// the GRPO sampler at batch-build time.
///
/// The dataset is intentionally split from the GRPO trainer's batching:
/// completion sampling requires running the policy model, which lives
/// inside the trainer, so the dataset only owns the prompt rendering.
pub struct PromptJsonlDataset {
    examples: Vec<PromptExample>,
    chat_env: Option<Arc<Environment<'static>>>,
}

impl PromptJsonlDataset {
    /// Load and parse a prompt-only JSONL file.
    ///
    /// `chat_template` is the same Jinja2 string accepted by
    /// [`JsonlDataset::from_path`]; if omitted, only `prompt` rows are
    /// supported.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Dataset`] for I/O failures, JSON parse
    /// failures, an empty file, or a chat-template compile failure.
    pub fn from_path(path: &Path, chat_template: Option<&str>) -> Result<Self, BlazenTrainError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            BlazenTrainError::Dataset(format!(
                "failed to read prompt jsonl at {}: {e}",
                path.display()
            ))
        })?;

        let mut examples: Vec<PromptExample> = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let row: PromptJsonlRow = serde_json::from_str(trimmed).map_err(|e| {
                BlazenTrainError::Dataset(format!(
                    "prompt jsonl parse error at {}:{}: {e}",
                    path.display(),
                    lineno + 1
                ))
            })?;
            if row.prompt.is_none() && row.messages.is_none() {
                return Err(BlazenTrainError::Dataset(format!(
                    "prompt row at {}:{} must have either `prompt` or `messages`",
                    path.display(),
                    lineno + 1
                )));
            }
            if let Some(msgs) = row.messages.as_ref()
                && msgs.is_empty()
            {
                return Err(BlazenTrainError::Dataset(format!(
                    "empty messages array at {}:{}",
                    path.display(),
                    lineno + 1
                )));
            }
            examples.push(PromptExample {
                prompt: row.prompt,
                messages: row.messages,
            });
        }

        if examples.is_empty() {
            return Err(BlazenTrainError::Dataset(format!(
                "prompt jsonl file at {} contains zero examples",
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

        Ok(Self { examples, chat_env })
    }

    /// Number of prompts loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether the dataset is empty (only true if file had no rows).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Borrow the rendered prompt text for the `idx`-th row.
    ///
    /// For `prompt` rows this is the raw string; for `messages` rows the
    /// chat template renders the message list with `add_generation_prompt
    /// = true`. The GRPO sampler appends sampled completions onto this
    /// string before tokenization.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenTrainError::Dataset`] if `idx` is out of bounds,
    /// if a `messages` row arrives without a chat template, or if the
    /// template render itself fails.
    pub fn prompt_text(&self, idx: usize) -> Result<String, BlazenTrainError> {
        let ex = self.examples.get(idx).ok_or_else(|| {
            BlazenTrainError::Dataset(format!(
                "prompt index {idx} out of range (len={})",
                self.examples.len()
            ))
        })?;
        if let Some(p) = &ex.prompt {
            return Ok(p.clone());
        }
        let msgs = ex.messages.as_ref().ok_or_else(|| {
            BlazenTrainError::Dataset("prompt example missing both fields".to_string())
        })?;
        let env = self.chat_env.as_ref().ok_or_else(|| {
            BlazenTrainError::Dataset(
                "prompt messages row encountered but no chat_template was provided".to_string(),
            )
        })?;
        let tpl = env
            .get_template("chat")
            .map_err(|e| BlazenTrainError::Dataset(format!("chat_template lookup failed: {e}")))?;
        let view: Vec<TemplateMsg> = msgs.iter().map(TemplateMsg::from).collect();
        tpl.render(context! {
            messages => view,
            add_generation_prompt => true,
        })
        .map_err(|e| {
            BlazenTrainError::Dataset(format!("chat_template render (prompt) failed: {e}"))
        })
    }
}

/// Pad a batch of `(ids, labels)` rows to a common length and stack into
/// `(input_ids, attn, labels)` tensors. Used by both preference and rated
/// datasets so their padding semantics match [`JsonlDataset`] exactly.
fn stack_padded(
    rows: &[(Vec<u32>, Vec<i64>)],
    pad_token_id: u32,
    ignore_index: i64,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor), BlazenTrainError> {
    let batch = rows.len();
    let max_len = rows
        .iter()
        .map(|(ids, _)| ids.len())
        .max()
        .unwrap_or(0)
        .max(1);

    let mut ids_flat: Vec<u32> = Vec::with_capacity(batch * max_len);
    let mut attn_flat: Vec<u32> = Vec::with_capacity(batch * max_len);
    let mut labels_flat: Vec<i64> = Vec::with_capacity(batch * max_len);

    for (ids, labels) in rows {
        let real_len = ids.len();
        debug_assert_eq!(real_len, labels.len());
        for (&tok, &lab) in ids.iter().zip(labels.iter()) {
            ids_flat.push(tok);
            attn_flat.push(1);
            labels_flat.push(lab);
        }
        for _ in real_len..max_len {
            ids_flat.push(pad_token_id);
            attn_flat.push(0);
            labels_flat.push(ignore_index);
        }
    }

    let shape = (batch, max_len);
    let input_ids = Tensor::from_vec(ids_flat, shape, device)?.to_dtype(DType::U32)?;
    let attn = Tensor::from_vec(attn_flat, shape, device)?.to_dtype(DType::U32)?;
    let labels = Tensor::from_vec(labels_flat, shape, device)?.to_dtype(DType::I64)?;
    Ok((input_ids, attn, labels))
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

    // -------------------------------------------------------------------------
    // PreferenceJsonlDataset / RatedJsonlDataset tests (PR8 Wave 1)
    // -------------------------------------------------------------------------

    #[test]
    fn preference_jsonl_loads_three_field_format() {
        let dir = TempDir::new().unwrap();
        let lines = [
            r#"{"prompt": "hello world ", "chosen": "foo bar", "rejected": "baz qux"}"#,
            r#"{"prompt": "hello ", "chosen": "foo", "rejected": "bar"}"#,
            r#"{"prompt": "world ", "chosen": "qux", "rejected": "baz"}"#,
        ];
        let path = write_jsonl(&dir, &lines);

        let tk = build_tokenizer();
        let ds = PreferenceJsonlDataset::from_path(&path, tk, None, 32, Device::Cpu, 0).unwrap();
        assert_eq!(ds.len(), 3);

        let batch = ds.batch(2, 0).unwrap();
        assert_eq!(batch.chosen_input_ids.dim(0).unwrap(), 2);
        assert_eq!(batch.rejected_input_ids.dim(0).unwrap(), 2);
        // Chosen and rejected may have different T but both must be 2D [B, T].
        assert_eq!(batch.chosen_input_ids.dims().len(), 2);
        assert_eq!(batch.rejected_input_ids.dims().len(), 2);
        assert_eq!(batch.chosen_labels.dim(0).unwrap(), 2);
        assert_eq!(batch.rejected_labels.dim(0).unwrap(), 2);
        assert_eq!(batch.chosen_attn.dim(0).unwrap(), 2);
        assert_eq!(batch.rejected_attn.dim(0).unwrap(), 2);
    }

    #[test]
    fn preference_jsonl_masks_prompt_in_labels() {
        let dir = TempDir::new().unwrap();
        let line =
            r#"{"prompt": "hello world foo bar ", "chosen": "baz qux", "rejected": "the answer"}"#;
        let path = write_jsonl(&dir, &[line]);

        let tk = build_tokenizer();
        let ds = PreferenceJsonlDataset::from_path(&path, tk, None, 32, Device::Cpu, 0).unwrap();
        let batch = ds.batch(1, 0).unwrap();

        let chosen_labels: Vec<Vec<i64>> = batch.chosen_labels.to_vec2().unwrap();
        let rejected_labels: Vec<Vec<i64>> = batch.rejected_labels.to_vec2().unwrap();

        let masked = chosen_labels[0]
            .iter()
            .filter(|&&v| v == DEFAULT_IGNORE_INDEX)
            .count();
        let kept = chosen_labels[0]
            .iter()
            .filter(|&&v| v != DEFAULT_IGNORE_INDEX)
            .count();
        assert!(masked > 0, "chosen prompt prefix must be masked");
        assert!(kept > 0, "chosen completion must be kept");

        // Prompt prefix should produce identical leading -100 spans on both
        // sides since the prompt is shared.
        let chosen_prefix = chosen_labels[0]
            .iter()
            .take_while(|&&v| v == DEFAULT_IGNORE_INDEX)
            .count();
        let rejected_prefix = rejected_labels[0]
            .iter()
            .take_while(|&&v| v == DEFAULT_IGNORE_INDEX)
            .count();
        assert_eq!(
            chosen_prefix, rejected_prefix,
            "shared prompt should yield equal label-mask prefixes"
        );
    }

    #[test]
    fn preference_jsonl_handles_messages_format() {
        let dir = TempDir::new().unwrap();
        let line = r#"{"messages": [{"role":"user","content":"hello world"}], "chosen": "the answer is 42", "rejected": "foo bar"}"#;
        let path = write_jsonl(&dir, &[line]);

        let tk = build_tokenizer();
        let ds = PreferenceJsonlDataset::from_path(
            &path,
            tk,
            Some(simple_template()),
            64,
            Device::Cpu,
            0,
        )
        .unwrap();
        assert_eq!(ds.len(), 1);
        let batch = ds.batch(1, 0).unwrap();
        assert_eq!(batch.chosen_input_ids.dim(0).unwrap(), 1);
        assert_eq!(batch.rejected_input_ids.dim(0).unwrap(), 1);

        // Without a chat template, the messages row must fail.
        let ds_no_tpl =
            PreferenceJsonlDataset::from_path(&path, build_tokenizer(), None, 64, Device::Cpu, 0)
                .unwrap();
        assert!(ds_no_tpl.batch(1, 0).is_err());
    }

    #[test]
    fn rated_jsonl_loads_mixed_labels() {
        let dir = TempDir::new().unwrap();
        let lines = [
            r#"{"prompt": "hello world ", "completion": "foo bar", "label": true}"#,
            r#"{"prompt": "hello ", "completion": "baz qux", "label": false}"#,
            r#"{"prompt": "world ", "completion": "the answer", "label": true}"#,
        ];
        let path = write_jsonl(&dir, &lines);

        let tk = build_tokenizer();
        let ds = RatedJsonlDataset::from_path(&path, tk, None, 32, Device::Cpu, 0).unwrap();
        assert_eq!(ds.len(), 3);

        let batch = ds.batch(3, 0).unwrap();
        assert_eq!(batch.input_ids.dim(0).unwrap(), 3);
        assert_eq!(batch.desirable_mask.dims(), &[3]);

        let mask: Vec<f32> = batch.desirable_mask.to_vec1().unwrap();
        assert!((mask[0] - 1.0).abs() < f32::EPSILON);
        assert!(mask[1].abs() < f32::EPSILON);
        assert!((mask[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn rated_jsonl_pads_batch() {
        let dir = TempDir::new().unwrap();
        let lines = [
            r#"{"prompt": "hello ", "completion": "foo", "label": true}"#,
            r#"{"prompt": "hello world foo bar baz qux quux ", "completion": "corge grault", "label": false}"#,
        ];
        let path = write_jsonl(&dir, &lines);

        let tk = build_tokenizer();
        let ds = RatedJsonlDataset::from_path(&path, tk, None, 64, Device::Cpu, 999).unwrap();
        let batch = ds.batch(2, 0).unwrap();

        let b = batch.input_ids.dim(0).unwrap();
        let seq = batch.input_ids.dim(1).unwrap();
        assert_eq!(b, 2);
        assert!(seq >= 2);

        let attn: Vec<Vec<u32>> = batch.attn.to_vec2().unwrap();
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

        // Padding positions in labels must be DEFAULT_IGNORE_INDEX.
        let labels: Vec<Vec<i64>> = batch.labels.to_vec2().unwrap();
        for (row_idx, row) in labels.iter().enumerate() {
            for (col_idx, &lab) in row.iter().enumerate() {
                if attn[row_idx][col_idx] == 0 {
                    assert_eq!(lab, DEFAULT_IGNORE_INDEX);
                }
            }
        }
    }
}
