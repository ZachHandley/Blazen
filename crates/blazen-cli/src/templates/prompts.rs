pub const TEMPLATE: &str = r#"# Blazen Prompts — Usage Guide

## Installation

### Rust

```bash
cargo add blazen --registry forgejo --features prompts
```

### Python

```bash
pip install blazen --index-url https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/pypi/simple/
```

### TypeScript / Node.js

```bash
npm install blazen --registry https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/npm/
```

## Core Concepts

- **PromptTemplate** — A reusable template with `{{variable}}` placeholders, a role, and versioning.
- **PromptRegistry** — A collection of named, versioned templates with lookup and rendering.
- **PromptFile** — YAML or JSON file format for storing prompt template collections.
- **Variable interpolation** — `{{var_name}}` placeholders are replaced at render time.

## Directory Structure

```
prompts/
  example.yaml      # Example prompt templates (created by blazen init prompts)
  summarise.yaml    # Add more YAML files as needed
  classify.yaml
```

## Quick Start (Rust)

```rust
use blazen::prompts::{PromptRegistry, PromptTemplate, TemplateRole};
use std::collections::HashMap;

// Load all prompt files from the prompts/ directory
let registry = PromptRegistry::from_dir("./prompts")?;

// Render a prompt with variables
let mut vars = HashMap::new();
vars.insert("topic".to_owned(), "quantum computing".to_owned());
vars.insert("style".to_owned(), "concise".to_owned());

let message = registry.render("summarise", &vars)?;
println!("Role: {:?}, Content: {}", message.role, message.content);
```

### Inline Templates

```rust
use blazen::prompts::{PromptTemplate, TemplateRole};

let template = PromptTemplate::new(
    "greet",
    TemplateRole::User,
    "Hello {{name}}, tell me about {{topic}}.",
);

let message = template.render_with(&[
    ("name", "Alice"),
    ("topic", "Rust"),
])?;
```

### Versioned Prompts

```rust
let mut registry = PromptRegistry::new();

registry.register(
    PromptTemplate::new("greet", TemplateRole::User, "Hello {{name}}!")
        .with_version("1.0"),
);
registry.register(
    PromptTemplate::new("greet", TemplateRole::User, "Hey {{name}}, welcome!")
        .with_version("2.0"),
);

// get() returns latest (2.0)
let latest = registry.get("greet").unwrap();

// Pin to a specific version
let v1 = registry.get_version("greet", "1.0").unwrap();
```

### Building CompletionRequests

```rust
use blazen::prompts::PromptRegistry;

let registry = PromptRegistry::from_dir("./prompts")?;

let request = registry.build_request(&[
    ("system_prompt", [("domain".into(), "medical".into())].into()),
    ("user_query", [("question".into(), "What causes headaches?".into())].into()),
])?;

// Pass request.messages to your CompletionModel
```

## YAML File Format

```yaml
prompts:
  - name: summarise
    version: "1.0"
    role: system
    description: "System prompt for summarisation tasks"
    template: |
      You are an expert summariser. Summarise the following {{doc_type}}
      in a {{style}} style. Focus on key insights.
    metadata:
      author: team
      category: summarisation

  - name: user_query
    role: user
    template: "{{question}}"
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier for lookup |
| `role` | Yes | `system`, `user`, or `assistant` |
| `template` | Yes | Text with `{{variable}}` placeholders |
| `version` | No | Version string (default: `"1.0"`) |
| `description` | No | Human-readable purpose |
| `metadata` | No | Arbitrary key-value pairs |

## Saving Prompts

```rust
// Save registry to a single file
registry.to_file("./prompts/all.yaml")?;

// Also supports JSON
registry.to_file("./prompts/all.json")?;
```
"#;

pub const EXAMPLE_YAML: &str = r#"prompts:
  - name: summarise
    version: "1.0"
    role: system
    description: "System prompt for summarisation tasks"
    template: |
      You are an expert summariser. Summarise the following {{doc_type}}
      in a {{style}} style. Focus on the key insights and main arguments.
    metadata:
      author: blazen-cli
      category: summarisation

  - name: classify
    version: "1.0"
    role: system
    description: "System prompt for text classification"
    template: |
      You are a text classifier. Classify the following text into one of
      these categories: {{categories}}.

      Respond with only the category name.
    metadata:
      author: blazen-cli
      category: classification

  - name: user_query
    version: "1.0"
    role: user
    description: "Generic user query template"
    template: "{{question}}"
"#;
