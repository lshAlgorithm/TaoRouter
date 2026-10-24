//! Evidence that a worker really produced generated tokens for a proxied user
//! request.
//!
//! The observational inference probe ([`crate::core::inference_probe`]) exists
//! to catch a worker that answers but cannot infer. A worker that is already
//! serving real traffic has answered that question by itself, so
//! [`crate::core::WorkerRegistry::start_inference_probe`] skips it while its
//! evidence is fresh, instead of spending a real one-token generation on it
//! every interval.
//!
//! Two pieces live here:
//!
//! * the monotonic evidence clock ([`now_millis`]) — milliseconds since this
//!   process started, derived from an [`Instant`], never from `SystemTime`, so
//!   a wall-clock jump cannot make evidence look fresh forever;
//! * the bounded body scan ([`body_shows_generated_tokens`]) used by the
//!   non-streaming relay paths to decide whether a 2xx response actually
//!   carries a generation. It applies the same bar as the probe itself: a bare
//!   2xx with an empty `choices: []` is *not* evidence.
//!
//! Evidence is recorded per worker (see `Worker::note_generation_evidence`).
//! A hung stream stops producing chunks, its evidence goes stale, and the
//! worker is probed again on a later sweep — that self-healing property is why
//! skipping a busy worker is safe.

use std::sync::OnceLock;
use std::time::Instant;

/// Process-start reference for the monotonic evidence clock.
fn process_start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Monotonic milliseconds since this process started, never `0`.
///
/// The evidence clock must not be a wall-clock timestamp: `SystemTime` can jump
/// backwards (NTP step, suspend/resume), which would make stale evidence look
/// fresh and silently stop probing a worker that has stopped generating.
///
/// It must also never return `0`, because `0` is the "no generation seen yet"
/// sentinel (see `Worker::last_generation_ts`): a worker that first served
/// within the process's first millisecond would otherwise be indistinguishable
/// from one that has never served. Clamping the first millisecond to `1` keeps
/// the clock monotonic non-decreasing and the sentinel unambiguous.
pub fn now_millis() -> u64 {
    let millis: u64 = process_start()
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    millis.max(1)
}

/// Upper bound on how much of a response body the evidence scan inspects.
///
/// A completion puts `choices` first and `usage` last, so the head carries
/// either a non-empty choice (long real generation) or the whole document (a
/// short one). A body longer than this cap is already unambiguously served
/// traffic. Truncating the scan can therefore only produce a false *negative* —
/// one extra probe — never a false positive.
const MAX_SCAN_BYTES: usize = 64 * 1024;

/// Whether a non-streaming completion body demonstrates that the worker
/// generated tokens.
///
/// This is deliberately a bounded byte scan rather than a full JSON parse: it
/// runs on every proxied non-streaming response, including large ones. It
/// accepts a response when it carries a non-empty `choices` array plus either
///
/// * `usage.completion_tokens` present and `>= 1`, or
/// * a non-empty `text` / `content` string in the first choice **and no
///   `completion_tokens` field at all**,
///
/// and, for a body with no `choices` key at all (a bare usage document), when
/// `usage.completion_tokens` is present and `>= 1`.
///
/// A 2xx with `choices: []`, a zero token count, or an error-shaped body is not
/// evidence — the same bar `probe_inference` uses for a successful probe. In
/// particular an **explicit** empty `choices` array is authoritative and stays
/// a failure even alongside a positive token count: that is exactly the shape
/// of the usage-only terminal chunk of an SSE stream, and of the probe's own
/// `no_choice` outcome, neither of which is a completed generation on its own.
/// An **explicit** `completion_tokens: 0` is equally authoritative: it vetoes
/// the text fallback, because `probe_inference` classifies exactly that body
/// (`text: "."`, `finish_reason: "length"`, `completion_tokens: 0`) as
/// `zero_tokens`. Counting it as evidence would let a worker that never
/// generates tokens keep its sweep skipped and its `inference_ok` pinned to 1
/// for as long as it keeps answering.
pub fn body_shows_generated_tokens(body: &[u8]) -> bool {
    let body = &body[..body.len().min(MAX_SCAN_BYTES)];
    let tokens = completion_tokens_state(body);
    if !choices_present(body) {
        // A bare usage document is only evidence when it reports generation.
        return tokens == TokenCount::AtLeastOne;
    }
    if !choices_is_non_empty(body) {
        return false;
    }
    match tokens {
        TokenCount::AtLeastOne => true,
        TokenCount::Zero => false,
        TokenCount::Absent => choice_carries_text(body),
    }
}

/// How a body reports the number of tokens the worker generated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TokenCount {
    /// No `usage.completion_tokens` field is present at all.
    Absent,
    /// The field is present and reports zero, or is present but unreadable.
    Zero,
    /// The field reports at least one token.
    AtLeastOne,
}

/// Whether the body has a `"choices"` key at all, regardless of its value.
fn choices_present(body: &[u8]) -> bool {
    find(body, b"\"choices\"", 0).is_some()
}

/// Classify the `completion_tokens` field, tolerating arbitrary whitespace
/// around the separator (serde_json may be pretty-printed).
///
/// Every occurrence is inspected: a positive count anywhere wins, otherwise a
/// present field is [`TokenCount::Zero`]. A field that is present but does not
/// hold a number (a string, `null`) counts as zero — it is not a count this
/// scan can trust, and treating it as absent would let the text fallback accept
/// a body whose token count is unreadable.
fn completion_tokens_state(body: &[u8]) -> TokenCount {
    const KEY: &[u8] = b"\"completion_tokens\"";
    let mut saw_field = false;
    let mut search_from = 0;
    while let Some(offset) = find(body, KEY, search_from) {
        search_from = offset + KEY.len();
        let mut index = skip_whitespace(body, search_from);
        if body.get(index) != Some(&b':') {
            continue;
        }
        saw_field = true;
        index = skip_whitespace(body, index + 1);
        let digits_start = index;
        while index < body.len() && body[index].is_ascii_digit() {
            index += 1;
        }
        if index == digits_start {
            continue;
        }
        let value = body[digits_start..index].iter().fold(0u64, |acc, digit| {
            acc.saturating_mul(10)
                .saturating_add(u64::from(digit - b'0'))
        });
        if value >= 1 {
            return TokenCount::AtLeastOne;
        }
    }
    if saw_field {
        TokenCount::Zero
    } else {
        TokenCount::Absent
    }
}

/// A non-empty `"choices"` array whose first object carries a non-empty `"text"`
/// or `"content"` string.
///
/// `choices` being an array with at least one element is the structural half
/// (an empty `choices: []` must not count); the string half is what
/// distinguishes a completed generation from a metadata-only or error body.
fn choice_carries_text(body: &[u8]) -> bool {
    if !choices_is_non_empty(body) {
        return false;
    }
    [b"\"text\"".as_slice(), b"\"content\"".as_slice()]
        .iter()
        .any(|key| has_non_empty_string_value(body, key))
}

/// Whether the `"choices"` array is present, non-empty, and non-empty after
/// arbitrary whitespace (e.g. `"choices": [ {...} ]`).
fn choices_is_non_empty(body: &[u8]) -> bool {
    const KEY: &[u8] = b"\"choices\"";
    let Some(offset) = find(body, KEY, 0) else {
        return false;
    };
    let mut index = skip_whitespace(body, offset + KEY.len());
    if body.get(index) != Some(&b':') {
        return false;
    }
    index = skip_whitespace(body, index + 1);
    if body.get(index) != Some(&b'[') {
        return false;
    }
    !matches!(
        body.get(skip_whitespace(body, index + 1)),
        None | Some(&b']')
    )
}

/// Whether `key` is followed by a JSON string whose value is not empty.
///
/// An escaped first byte (e.g. `"\""`) counts as non-empty; the scan only has
/// to avoid calling an explicitly empty `""` a generation.
fn has_non_empty_string_value(body: &[u8], key: &[u8]) -> bool {
    let Some(offset) = find(body, key, 0) else {
        return false;
    };
    let mut index = skip_whitespace(body, offset + key.len());
    if body.get(index) != Some(&b':') {
        return false;
    }
    index = skip_whitespace(body, index + 1);
    if body.get(index) != Some(&b'"') {
        return false;
    }
    body.get(index + 1).is_some_and(|byte| *byte != b'"')
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|position| position + from)
}

fn skip_whitespace(body: &[u8], mut index: usize) -> usize {
    while body
        .get(index)
        .is_some_and(|byte| byte.is_ascii_whitespace())
    {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_well_formed_completion_with_usage() {
        let body = br#"{"id":"cmpl-1","choices":[{"text":".","index":0,"finish_reason":"length"}],"usage":{"completion_tokens":1,"prompt_tokens":1}}"#;
        assert!(body_shows_generated_tokens(body));
    }

    #[test]
    fn accepts_a_positive_token_count_with_empty_text() {
        // A legitimate end-of-sequence token can yield empty text while still
        // being a completed generation (the probe accepts this too).
        let body =
            br#"{"choices":[{"text":"","finish_reason":"stop"}],"usage":{"completion_tokens":3}}"#;
        assert!(body_shows_generated_tokens(body));
    }

    #[test]
    fn tolerates_whitespace_around_the_separator() {
        let body = b"{\"usage\": {\"completion_tokens\" : 12 }}";
        assert!(body_shows_generated_tokens(body));
    }

    #[test]
    fn accepts_a_chat_completion_without_usage() {
        let body = br#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#;
        assert!(body_shows_generated_tokens(body));
    }

    #[test]
    fn rejects_a_bare_2xx_body() {
        assert!(!body_shows_generated_tokens(b"{}"));
        assert!(!body_shows_generated_tokens(br#"{"status":"ok"}"#));
    }

    #[test]
    fn rejects_empty_choices() {
        // This is exactly the probe's own `no_choice` outcome: 200, no token.
        let body = br#"{"id":"cmpl-1","choices":[],"usage":{"completion_tokens":1}}"#;
        assert!(!body_shows_generated_tokens(body));
    }

    #[test]
    fn rejects_zero_tokens() {
        let body = br#"{"choices":[{"text":"","index":0,"finish_reason":"length"}],"usage":{"completion_tokens":0}}"#;
        assert!(!body_shows_generated_tokens(body));
    }

    #[test]
    fn rejects_an_explicit_zero_even_when_a_choice_carries_text() {
        // The probe tests' `Reply::ZeroTokens` fixture: a non-empty text with
        // `completion_tokens: 0`. `probe_inference` classifies this body as
        // `zero_tokens`, so it must not count as evidence either -- otherwise a
        // worker that never generates tokens could keep its sweep skipped and
        // `inference_ok` pinned to 1 for as long as it keeps answering.
        let body = br#"{"choices":[{"text":".","finish_reason":"length"}],"usage":{"completion_tokens":0}}"#;
        assert!(!body_shows_generated_tokens(body));
    }

    #[test]
    fn accepts_text_only_when_the_token_count_is_absent() {
        // Same shape with the usage field removed: no count to veto, so the
        // text fallback applies (a streamed chunk, or a worker that omits usage).
        let body = br#"{"choices":[{"text":".","finish_reason":"length"}]}"#;
        assert!(body_shows_generated_tokens(body));
    }

    #[test]
    fn rejects_an_error_shaped_body() {
        let body = br#"{"error":{"message":"engine cannot infer","type":"internal"}}"#;
        assert!(!body_shows_generated_tokens(body));
    }

    #[test]
    fn rejects_a_choice_without_a_text_field() {
        let body = br#"{"choices":[{"index":0,"finish_reason":"error"}]}"#;
        assert!(!body_shows_generated_tokens(body));
    }

    #[test]
    fn rejects_a_completion_tokens_field_that_is_not_a_number() {
        let body = br#"{"usage":{"completion_tokens":"1"}}"#;
        assert!(!body_shows_generated_tokens(body));
    }

    #[test]
    fn only_inspects_the_bounded_head() {
        // A `completion_tokens` marker beyond the bound is ignored: truncation
        // may only cause an extra probe, never a fabricated success.
        let mut body = Vec::new();
        body.extend_from_slice(br#"{"choices":[]"#);
        body.resize(MAX_SCAN_BYTES + 8, b' ');
        body.extend_from_slice(br#","usage":{"completion_tokens":5}}"#);
        assert!(!body_shows_generated_tokens(&body));
    }

    #[test]
    fn evidence_clock_is_monotonic_and_nonzero_after_start() {
        let first = now_millis();
        let second = now_millis();
        assert!(second >= first);
        // `0` is the "never served" sentinel, so the clock must not hand it out
        // even when this test runs inside the process's first millisecond.
        assert!(
            first > 0,
            "the evidence clock must never return the sentinel"
        );
    }
}
