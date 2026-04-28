use napi_derive::napi;

#[napi(js_name = "PipelineEvent")]
pub struct JsPipelineEvent {
    pub(crate) stage_name: String,
    pub(crate) branch_name: Option<String>,
    pub(crate) workflow_run_id: String,
    pub(crate) event: serde_json::Value,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsPipelineEvent {
    #[napi(getter, js_name = "stageName")]
    pub fn stage_name(&self) -> String {
        self.stage_name.clone()
    }

    #[napi(getter, js_name = "branchName")]
    pub fn branch_name(&self) -> Option<String> {
        self.branch_name.clone()
    }

    #[napi(getter, js_name = "workflowRunId")]
    pub fn workflow_run_id(&self) -> String {
        self.workflow_run_id.clone()
    }

    #[napi(getter)]
    pub fn event(&self) -> serde_json::Value {
        self.event.clone()
    }
}
