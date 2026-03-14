//! Integration tests for the `blazen-prompts` crate.

use std::collections::HashMap;

use blazen_llm::Role;
use blazen_prompts::{PromptError, PromptRegistry, PromptTemplate, TemplateRole};

// ---------------------------------------------------------------------------
// Template rendering
// ---------------------------------------------------------------------------

#[test]
fn test_template_render() {
    let template = PromptTemplate::new(
        "greet",
        TemplateRole::User,
        "Hello {{name}}, welcome to {{place}}!",
    );

    let mut vars = HashMap::new();
    vars.insert("name".to_owned(), "Alice".to_owned());
    vars.insert("place".to_owned(), "Wonderland".to_owned());

    let message = template.render(&vars).unwrap();
    assert_eq!(message.role, Role::User);

    let text = message.content.as_text().unwrap();
    assert_eq!(text, "Hello Alice, welcome to Wonderland!");
}

#[test]
fn test_missing_variable_errors() {
    let template = PromptTemplate::new("greet", TemplateRole::User, "Hello {{name}}, age {{age}}!");

    let mut vars = HashMap::new();
    vars.insert("name".to_owned(), "Bob".to_owned());
    // "age" is missing

    let err = template.render(&vars).unwrap_err();
    match err {
        PromptError::MissingVariable { template, name } => {
            assert_eq!(template, "greet");
            assert_eq!(name, "age");
        }
        other => panic!("expected MissingVariable, got: {other}"),
    }
}

#[test]
fn test_extra_variables_ignored() {
    let template = PromptTemplate::new("simple", TemplateRole::User, "Hello {{name}}!");

    let mut vars = HashMap::new();
    vars.insert("name".to_owned(), "Alice".to_owned());
    vars.insert("extra".to_owned(), "ignored".to_owned());

    let message = template.render(&vars).unwrap();
    let text = message.content.as_text().unwrap();
    assert_eq!(text, "Hello Alice!");
}

#[test]
fn test_variable_extraction() {
    let template = PromptTemplate::new(
        "info",
        TemplateRole::User,
        "Hello {{name}}, you are {{age}}",
    );
    let vars = template.variables();
    assert_eq!(vars, &["age", "name"]);
}

// ---------------------------------------------------------------------------
// Registry versioning
// ---------------------------------------------------------------------------

#[test]
fn test_registry_latest_version() {
    let mut registry = PromptRegistry::new();

    registry.register(
        PromptTemplate::new("greet", TemplateRole::User, "Hello {{name}}!").with_version("1.0"),
    );
    registry.register(
        PromptTemplate::new("greet", TemplateRole::User, "Hey {{name}}, welcome!")
            .with_version("2.0"),
    );

    let latest = registry.get("greet").unwrap();
    assert_eq!(latest.version, "2.0");
    assert!(latest.template.contains("Hey"));
}

#[test]
fn test_registry_specific_version() {
    let mut registry = PromptRegistry::new();

    registry.register(
        PromptTemplate::new("greet", TemplateRole::User, "Hello {{name}}!").with_version("1.0"),
    );
    registry.register(
        PromptTemplate::new("greet", TemplateRole::User, "Hey {{name}}, welcome!")
            .with_version("2.0"),
    );

    let v1 = registry.get_version("greet", "1.0").unwrap();
    assert_eq!(v1.version, "1.0");
    assert!(v1.template.contains("Hello"));
}

// ---------------------------------------------------------------------------
// Build request
// ---------------------------------------------------------------------------

#[test]
fn test_build_request() {
    let mut registry = PromptRegistry::new();

    registry.register(PromptTemplate::new(
        "system_prompt",
        TemplateRole::System,
        "You are a {{role}} assistant.",
    ));
    registry.register(PromptTemplate::new(
        "user_prompt",
        TemplateRole::User,
        "Analyse this {{doc_type}}.",
    ));

    let mut system_vars = HashMap::new();
    system_vars.insert("role".to_owned(), "helpful".to_owned());

    let mut user_vars = HashMap::new();
    user_vars.insert("doc_type".to_owned(), "image".to_owned());

    let request = registry
        .build_request(&[("system_prompt", system_vars), ("user_prompt", user_vars)])
        .unwrap();

    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.messages[0].role, Role::System);
    assert_eq!(request.messages[1].role, Role::User);

    let sys_text = request.messages[0].content.as_text().unwrap();
    assert_eq!(sys_text, "You are a helpful assistant.");

    let user_text = request.messages[1].content.as_text().unwrap();
    assert_eq!(user_text, "Analyse this image.");
}

// ---------------------------------------------------------------------------
// File I/O roundtrips
// ---------------------------------------------------------------------------

#[test]
fn test_yaml_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prompts.yaml");

    let mut registry = PromptRegistry::new();
    registry.register(
        PromptTemplate::new(
            "analyze",
            TemplateRole::System,
            "Analyze the {{image_type}} focusing on {{focus}}.",
        )
        .with_version("1.0")
        .with_description("Image analysis prompt")
        .with_metadata("author", "test"),
    );
    registry.register(
        PromptTemplate::new("query", TemplateRole::User, "What is {{question}}?")
            .with_version("1.0"),
    );

    registry.to_file(&path).unwrap();

    let loaded = PromptRegistry::from_file(&path).unwrap();

    // Verify both prompts survived the roundtrip.
    let analyze = loaded.get("analyze").unwrap();
    assert_eq!(analyze.version, "1.0");
    assert_eq!(
        analyze.description.as_deref(),
        Some("Image analysis prompt")
    );
    assert_eq!(
        analyze.metadata.get("author").map(String::as_str),
        Some("test")
    );
    assert_eq!(analyze.variables(), &["focus", "image_type"]);

    let query = loaded.get("query").unwrap();
    assert_eq!(query.version, "1.0");
    assert_eq!(query.variables(), &["question"]);
}

#[test]
fn test_json_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prompts.json");

    let mut registry = PromptRegistry::new();
    registry.register(
        PromptTemplate::new("greet", TemplateRole::User, "Hello {{name}}!").with_version("1.0"),
    );

    registry.to_file(&path).unwrap();

    let loaded = PromptRegistry::from_file(&path).unwrap();
    let greet = loaded.get("greet").unwrap();
    assert_eq!(greet.version, "1.0");
    assert_eq!(greet.variables(), &["name"]);

    // Verify rendering still works after deserialization.
    let mut vars = HashMap::new();
    vars.insert("name".to_owned(), "World".to_owned());
    let msg = greet.render(&vars).unwrap();
    assert_eq!(msg.content.as_text().unwrap(), "Hello World!");
}

// ---------------------------------------------------------------------------
// Convenience methods
// ---------------------------------------------------------------------------

#[test]
fn test_render_with_tuples() {
    let template = PromptTemplate::new("greet", TemplateRole::User, "Hello {{name}}!");

    let message = template.render_with(&[("name", "Alice")]).unwrap();
    let text = message.content.as_text().unwrap();
    assert_eq!(text, "Hello Alice!");
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_registry_not_found() {
    let registry = PromptRegistry::new();
    assert!(registry.get("nonexistent").is_none());
}

#[test]
fn test_empty_template() {
    let template = PromptTemplate::new("static", TemplateRole::System, "No variables here.");

    assert!(template.variables().is_empty());

    let message = template.render(&HashMap::new()).unwrap();
    let text = message.content.as_text().unwrap();
    assert_eq!(text, "No variables here.");
}
