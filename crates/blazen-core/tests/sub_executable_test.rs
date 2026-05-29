//! Integration test for [`SubExecutable`] + [`SubPipelineStep`].
//!
//! Mirrors `subworkflow_tests.rs` but embeds a `blazen_pipeline::Pipeline`
//! (which gets its `SubExecutable` impl from `blazen-pipeline`) inside a
//! parent `Workflow` via the trait-object dispatch path on
//! `StepKind::SubPipeline`.

use std::sync::Arc;

use blazen_core::{
    StepFn, StepOutput, StepRegistration, SubExecutable, SubPipelineStep, SubWorkflowInputMapper,
    SubWorkflowOutputMapper, WorkflowBuilder,
};
use blazen_events::{Event, StartEvent, StopEvent};
use blazen_pipeline::PipelineBuilder;

/// Default input mapper: forwards the parent `StartEvent` payload as-is.
fn passthrough_input_mapper() -> SubWorkflowInputMapper {
    Arc::new(|event| {
        if let Some(start) = event.as_any().downcast_ref::<StartEvent>() {
            start.data.clone()
        } else {
            event.to_json()
        }
    })
}

/// Default output mapper: wraps the child's terminal JSON in a `StopEvent`
/// so the parent workflow exits cleanly.
fn stop_event_output_mapper() -> SubWorkflowOutputMapper {
    Arc::new(|json| Box::new(StopEvent { result: json }))
}

#[tokio::test]
async fn subpipeline_step_runs_pipeline_and_emits_final_output() {
    // Build a tiny single-stage pipeline that doubles a number.
    let pipeline = PipelineBuilder::<serde_json::Value>::new("doubler")
        .stage_async("double", |i: i32| async move {
            Ok::<_, blazen_llm::BlazenError>(i * 2)
        })
        .build()
        .expect("doubler pipeline must build");

    // Wrap as an Arc<dyn SubExecutable> and embed in a parent workflow.
    let executable: Arc<dyn SubExecutable> = Arc::new(pipeline);

    let sub = SubPipelineStep {
        name: "outer_sub".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        executable,
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: None,
        retry_config: None,
    };

    let parent = WorkflowBuilder::new("parent-doubler")
        .add_subpipeline_step(sub)
        .no_timeout()
        .build()
        .expect("parent workflow must build");

    let handler = parent
        .run(serde_json::json!(21))
        .await
        .expect("parent run() must succeed");
    let result = handler.result().await.expect("workflow must complete");
    let stop = result
        .event
        .as_any()
        .downcast_ref::<StopEvent>()
        .expect("terminal event must be StopEvent");

    // The pipeline's final_output is the doubled integer; the parent's
    // output_mapper just wraps it in a StopEvent.
    let n: i32 = serde_json::from_value(stop.result.clone()).expect("integer");
    assert_eq!(n, 42, "parent should emit the pipeline's doubled output");
}

#[tokio::test]
async fn subpipeline_step_with_workflow_blanket_impl_works() {
    // Build a child workflow that tags its input with `{"child": "wf"}`.
    let handler: StepFn = Arc::new(|event, _ctx| {
        Box::pin(async move {
            let start = event
                .as_any()
                .downcast_ref::<StartEvent>()
                .expect("expected StartEvent");
            let mut data = start.data.clone();
            if let Some(obj) = data.as_object_mut() {
                obj.insert("child".to_owned(), serde_json::json!("wf"));
            } else {
                data = serde_json::json!({ "input": data, "child": "wf" });
            }
            Ok(StepOutput::Single(Box::new(StopEvent { result: data })))
        })
    });
    let child = WorkflowBuilder::new("child-wf")
        .step(StepRegistration::new(
            "tag".to_owned(),
            vec![StartEvent::event_type()],
            vec![StopEvent::event_type()],
            handler,
            0,
        ))
        .no_timeout()
        .build()
        .expect("child workflow must build");

    // The blanket `impl SubExecutable for Workflow` lets us embed any
    // workflow through the SubPipeline path too.
    let exec: Arc<dyn SubExecutable> = Arc::new(child);
    let sub = SubPipelineStep {
        name: "outer_wf_sub".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        executable: exec,
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: None,
        retry_config: None,
    };

    let parent = WorkflowBuilder::new("parent-wf-sub")
        .add_subpipeline_step(sub)
        .no_timeout()
        .build()
        .expect("parent workflow must build");

    let h = parent.run(serde_json::json!({"n": 3})).await.unwrap();
    let res = h.result().await.expect("workflow must complete");
    let stop = res
        .event
        .as_any()
        .downcast_ref::<StopEvent>()
        .expect("terminal event must be StopEvent");

    assert_eq!(
        stop.result,
        serde_json::json!({"n": 3, "child": "wf"}),
        "parent should emit the workflow's tagged output via trait-object dispatch",
    );
}
