//! Current-request output-length shadow prediction.

use super::prediction_profile::{OutputLengthKey, OutputPrediction};
use super::ProgramIdentity;
use crate::config::types::PredictionProfileConfig;

pub(crate) fn keys_for_request(
    identity: &ProgramIdentity,
    estimated_context_tokens: usize,
    config: &PredictionProfileConfig,
) -> Vec<OutputLengthKey> {
    let input_bucket = config
        .input_length_bucket_bounds_tokens
        .iter()
        .copied()
        .find(|upper| estimated_context_tokens <= *upper)
        .unwrap_or(usize::MAX);
    let exact = OutputLengthKey {
        workflow_profile_id: identity.workflow_profile_id().map(str::to_string),
        profile_version: identity.profile_version().map(str::to_string),
        model_pool: Some(identity.model_pool().to_string()),
        agent_role: identity.agent_role().map(str::to_string),
        input_bucket: Some(input_bucket),
    };
    let mut keys = vec![exact.clone()];

    let mut without_input = exact.clone();
    without_input.input_bucket = None;
    push_unique(&mut keys, without_input);

    let model_role = OutputLengthKey {
        workflow_profile_id: None,
        profile_version: None,
        model_pool: exact.model_pool.clone(),
        agent_role: exact.agent_role.clone(),
        input_bucket: None,
    };
    push_unique(&mut keys, model_role);

    push_unique(
        &mut keys,
        OutputLengthKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: exact.model_pool,
            agent_role: None,
            input_bucket: None,
        },
    );
    push_unique(
        &mut keys,
        OutputLengthKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: None,
            agent_role: None,
            input_bucket: None,
        },
    );
    keys
}

fn push_unique(keys: &mut Vec<OutputLengthKey>, key: OutputLengthKey) {
    if !keys.contains(&key) {
        keys.push(key);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PendingOutputPrediction {
    pub keys: Vec<OutputLengthKey>,
    pub prediction: Option<OutputPrediction>,
}
