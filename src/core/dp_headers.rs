//! Data-parallel rank header plumbing, shared by every component that talks to
//! a DP-aware worker.
//!
//! Both the proxied user paths ([`crate::routers`]) and the core probe paths
//! (the health checker in [`crate::core::worker`] and the observational
//! inference probe in [`crate::core::inference_probe`]) must pin a logical rank
//! the same way. The helper therefore lives at the innermost consuming layer —
//! `core` — and [`crate::routers::http::dp_utils`] re-exports it, so no
//! `core → routers` dependency (and no `core ⇄ routers` module cycle) is
//! needed.

/// Header a DP-aware worker reads to select the logical engine to serve a
/// request from.
pub const DP_RANK_HEADER: &str = "X-data-parallel-rank";

/// Add the [`DP_RANK_HEADER`] header to a request when `dp_rank` is present.
///
/// A worker without a rank is left untouched, which keeps non-DP deployments on
/// the exact request they had before.
///
/// # Example
/// ```
/// use vllm_router_rs::core::dp_headers::add_dp_rank_header;
///
/// let client = reqwest::Client::new();
/// let request = client.post("https://worker:8000/v1/generate");
/// let request = add_dp_rank_header(request, Some(3));
/// // Request now has "X-data-parallel-rank: 3" header
/// ```
pub fn add_dp_rank_header(
    mut request: reqwest::RequestBuilder,
    dp_rank: Option<usize>,
) -> reqwest::RequestBuilder {
    if let Some(rank) = dp_rank {
        request = request.header(DP_RANK_HEADER, rank.to_string());
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header name is part of the wire contract with vLLM; a rename here
    /// silently un-pins every DP-aware request.
    #[test]
    fn dp_rank_header_name_is_stable() {
        assert_eq!(DP_RANK_HEADER, "X-data-parallel-rank");
    }

    #[test]
    fn add_dp_rank_header_pins_a_rank_and_leaves_none_untouched() {
        let client = reqwest::Client::new();

        let request = add_dp_rank_header(client.post("https://worker:8000"), Some(3))
            .build()
            .expect("request builds");
        assert_eq!(request.headers().get(DP_RANK_HEADER).unwrap(), "3");

        let request = add_dp_rank_header(client.post("https://worker:8000"), None)
            .build()
            .expect("request builds");
        assert!(request.headers().get(DP_RANK_HEADER).is_none());
    }
}
