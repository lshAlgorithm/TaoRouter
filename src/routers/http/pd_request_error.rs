//! Request-level P/D failures. Raw upstream bytes never participate in formatting.
use axum::{
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use serde_json::json;
use std::{fmt, future::Future};

pub(super) const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) enum PdPhase {
    Prefill,
    Decode,
}
impl fmt::Display for PdPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TransportErrorKind {
    Timeout,
    Connection,
    Other,
}
pub(super) fn classify_transport(error: &reqwest::Error) -> TransportErrorKind {
    if error.is_timeout() {
        TransportErrorKind::Timeout
    } else if error.is_connect() {
        TransportErrorKind::Connection
    } else {
        TransportErrorKind::Other
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ErrorBodyState {
    Complete,
    LimitExceeded,
    ReadFailed { kind: TransportErrorKind },
}
impl ErrorBodyState {
    fn label(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::LimitExceeded => "limit_exceeded",
            Self::ReadFailed { .. } => "read_failed",
        }
    }
}
pub(super) struct UpstreamHttpError {
    phase: PdPhase,
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
    body_state: ErrorBodyState,
}
#[derive(Debug)]
pub(super) struct TransportError {
    phase: PdPhase,
    kind: TransportErrorKind,
}
#[derive(Debug)]
pub(super) enum InternalErrorReason {
    MoriIoTransferModeMissing,
    SerializePrefillRequest,
    SerializeDecodeRequest,
    SerializeRequest,
    SerializeResponse,
    BuildResponse,
    RequestBuilder { phase: PdPhase },
}
#[derive(Debug)]
pub(super) enum PdRequestError {
    UpstreamHttp(Box<UpstreamHttpError>),
    Transport(TransportError),
    Protocol { phase: PdPhase },
    Internal(InternalErrorReason),
}
impl PdRequestError {
    pub(super) fn from_reqwest(phase: PdPhase, error: reqwest::Error) -> Self {
        if error.is_builder() {
            Self::Internal(InternalErrorReason::RequestBuilder { phase })
        } else {
            Self::Transport(TransportError {
                phase,
                kind: classify_transport(&error),
            })
        }
    }
    pub(super) fn protocol(phase: PdPhase) -> Self {
        Self::Protocol { phase }
    }
    /// These construction/serialization exits were not included in the old
    /// request counters. Errors from HTTP send/build still use their old exit.
    pub(super) fn records_legacy_request(&self) -> bool {
        !matches!(
            self,
            Self::Internal(
                InternalErrorReason::MoriIoTransferModeMissing
                    | InternalErrorReason::SerializePrefillRequest
                    | InternalErrorReason::SerializeDecodeRequest
                    | InternalErrorReason::SerializeRequest
            )
        )
    }
    pub(super) fn record_worker_error(&self, prefill: &str, decode: &str) {
        let phase = match self {
            Self::UpstreamHttp(e) => e.phase,
            Self::Transport(e) => e.phase,
            Self::Protocol { phase } => *phase,
            Self::Internal(InternalErrorReason::RequestBuilder { phase }) => *phase,
            Self::Internal(_) => return,
        };
        match phase {
            PdPhase::Prefill => crate::metrics::RouterMetrics::record_pd_prefill_error(prefill),
            PdPhase::Decode => crate::metrics::RouterMetrics::record_pd_decode_error(decode),
        }
    }
    fn status(&self) -> StatusCode {
        match self {
            Self::UpstreamHttp(error) => error.status,
            Self::Transport(TransportError {
                kind: TransportErrorKind::Timeout,
                ..
            }) => StatusCode::GATEWAY_TIMEOUT,
            Self::Transport(_) => StatusCode::BAD_GATEWAY,
            Self::Protocol { .. } | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    fn public_fields(&self) -> (&'static str, String) {
        match self {
            Self::UpstreamHttp(e) => (
                "upstream_http_error",
                format!(
                    "Upstream {} request failed (HTTP {}).",
                    e.phase,
                    e.status.as_u16()
                ),
            ),
            Self::Transport(TransportError {
                phase,
                kind: TransportErrorKind::Timeout,
            }) => (
                "upstream_timeout",
                format!("Upstream {phase} request timed out."),
            ),
            Self::Transport(TransportError { phase, .. }) => (
                "upstream_transport_error",
                format!("Unable to communicate with the upstream {phase} service."),
            ),
            Self::Protocol { phase } => (
                "upstream_protocol_error",
                format!("Upstream {phase} response did not satisfy the required protocol."),
            ),
            Self::Internal(_) => (
                "internal_error",
                "The router could not complete the request.".into(),
            ),
        }
    }
}
impl fmt::Display for PdRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.public_fields().1)
    }
}
// No raw source chain: generic error reporters must remain safe as well.
impl std::error::Error for PdRequestError {}
impl fmt::Debug for UpstreamHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpstreamHttpError")
            .field("phase", &self.phase)
            .field("status", &self.status.as_u16())
            .field("saved_bytes", &self.body.len())
            .field("body_state", &self.body_state)
            .finish()
    }
}
impl IntoResponse for PdRequestError {
    fn into_response(self) -> Response {
        tracing::warn!(error = %self, details = ?self, "P/D request failed");
        tracing::debug!(details = ?self, "P/D error classification");
        let (kind, message) = self.public_fields();
        let phase = match &self {
            Self::UpstreamHttp(e) => e.phase.to_string(),
            Self::Transport(e) => e.phase.to_string(),
            Self::Protocol { phase } => phase.to_string(),
            Self::Internal(_) => "router".to_owned(),
        };
        metrics::counter!("vllm_router_pd_phase_errors_total", "phase" => phase, "error_type" => kind).increment(1);
        let mut response = (self.status(), Json(json!({"error": {"message": message, "type": kind, "code": self.status().as_u16()}}))).into_response();
        if let Self::UpstreamHttp(error) = self {
            if let Some(value) = error.headers.get(header::RETRY_AFTER) {
                response
                    .headers_mut()
                    .insert(header::RETRY_AFTER, value.clone());
            }
        }
        response
    }
}

// Accept only delay-seconds or a round-tripped HTTP-date. chrono is already a
// direct dependency; no dependency/lockfile change is needed for this header.
fn valid_retry_after(value: &HeaderValue) -> bool {
    let Ok(text) = value.to_str() else {
        return false;
    };
    if !text.is_empty() && text.bytes().all(|c| c.is_ascii_digit()) {
        return true;
    }
    [
        "%a, %d %b %Y %H:%M:%S GMT",
        "%A, %d-%b-%y %H:%M:%S GMT",
        "%a %b %e %H:%M:%S %Y",
    ]
    .iter()
    .any(|format| {
        chrono::NaiveDateTime::parse_from_str(text, format)
            .is_ok_and(|date| date.format(format).to_string() == text)
    })
}

async fn capture_body<S>(stream: S) -> (Bytes, ErrorBodyState)
where
    S: Stream<Item = Result<Bytes, TransportErrorKind>>,
{
    futures_util::pin_mut!(stream);
    let mut body = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(kind) => return (body.freeze(), ErrorBodyState::ReadFailed { kind }),
        };
        let remaining = MAX_UPSTREAM_ERROR_BODY_BYTES - body.len();
        // Copy only the saved prefix; never retain a giant chunk's allocation.
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if chunk.len() > remaining {
            return (body.freeze(), ErrorBodyState::LimitExceeded);
        }
    }
    (body.freeze(), ErrorBodyState::Complete)
}

pub(super) async fn capture_upstream_error(
    phase: PdPhase,
    response: reqwest::Response,
) -> PdRequestError {
    assert!(
        !response.status().is_success(),
        "error capture requires non-2xx"
    );
    let status = response.status();
    let mut headers = HeaderMap::new();
    if let Some(value) = response.headers().get(header::CONTENT_TYPE) {
        headers.insert(header::CONTENT_TYPE, value.clone());
    }
    if let Some(value) = response
        .headers()
        .get(header::RETRY_AFTER)
        .filter(|v| valid_retry_after(v))
    {
        headers.insert(header::RETRY_AFTER, value.clone());
    }
    // reqwest's configured total timeout covers the body, including the wait
    // for EOF after exactly 64 KiB; do not reset that deadline per chunk.
    let (body, body_state) = capture_body(
        response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| classify_transport(&e))),
    )
    .await;
    metrics::counter!("vllm_router_pd_error_body_total", "phase" => phase.to_string(), "state" => body_state.label()).increment(1);
    let error = UpstreamHttpError {
        phase,
        status,
        headers,
        body,
        body_state,
    };
    tracing::debug!(details = ?error, "Upstream body omitted");
    PdRequestError::UpstreamHttp(Box::new(error))
}

/// Poll both legs, but Prefill's complete validation gates the response. If D
/// completes first its response is held (backpressure, no output buffering).
/// Dropping this future drops both sends; P failure immediately drops pending D.
pub(super) async fn prefill_first<P, D, T>(
    prefill: P,
    decode: D,
) -> Result<(T, reqwest::Response), PdRequestError>
where
    P: Future<Output = Result<T, PdRequestError>>,
    D: Future<Output = Result<reqwest::Response, reqwest::Error>>,
{
    tokio::pin!(prefill, decode);
    tokio::select! {
        biased;
        p = &mut prefill => {
            let p = p?;
            let d = decode.await.map_err(|e| PdRequestError::from_reqwest(PdPhase::Decode, e))?;
            Ok((p, d))
        }
        d = &mut decode => {
            let p = prefill.await?;
            Ok((p, d.map_err(|e| PdRequestError::from_reqwest(PdPhase::Decode, e))?))
        }
    }
}

/// Stream failures occur after commitment: terminate without synthesizing SSE.
pub(super) fn decode_stream(
    response: reqwest::Response,
) -> impl Stream<Item = Result<Bytes, PdRequestError>> {
    response.bytes_stream().map(|chunk| {
        chunk.map_err(|e| {
            let error = PdRequestError::from_reqwest(PdPhase::Decode, e);
            let (kind, _) = error.public_fields();
            // Byte streaming is used for both stream=true and ordinary direct
            // responses. New diagnostics must not change legacy SSE counters.
            metrics::counter!("vllm_router_pd_response_body_errors_total", "phase" => "decode", "error_type" => kind).increment(1);
            tracing::warn!(error = %error, "P/D stream terminated");
            error
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        pin::Pin,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        task::{Context, Poll},
    };

    #[test]
    fn internal_diagnostics_are_specific_but_public_messages_are_stable() {
        for (reason, name, records_request) in [
            (
                InternalErrorReason::MoriIoTransferModeMissing,
                "MoriIoTransferModeMissing",
                false,
            ),
            (
                InternalErrorReason::SerializePrefillRequest,
                "SerializePrefillRequest",
                false,
            ),
            (
                InternalErrorReason::SerializeDecodeRequest,
                "SerializeDecodeRequest",
                false,
            ),
            (
                InternalErrorReason::SerializeRequest,
                "SerializeRequest",
                false,
            ),
            (
                InternalErrorReason::SerializeResponse,
                "SerializeResponse",
                true,
            ),
            (InternalErrorReason::BuildResponse, "BuildResponse", true),
            (
                InternalErrorReason::RequestBuilder {
                    phase: PdPhase::Decode,
                },
                "RequestBuilder",
                true,
            ),
        ] {
            let error = PdRequestError::Internal(reason);
            assert!(format!("{error:?}").contains(name));
            assert_eq!(error.records_legacy_request(), records_request);
            assert_eq!(
                error.public_fields(),
                (
                    "internal_error",
                    "The router could not complete the request.".to_owned()
                )
            );
            assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert!(std::error::Error::source(&error).is_none());
        }
    }

    struct Chunks {
        chunks: std::collections::VecDeque<Result<Bytes, TransportErrorKind>>,
        polls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }
    impl Stream for Chunks {
        type Item = Result<Bytes, TransportErrorKind>;
        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(self.chunks.pop_front())
        }
    }
    impl Drop for Chunks {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn bounded_bytes_and_exact_eof() {
        for size in [0, 1, 65535, 65536, 65537, 1_000_000] {
            for chunk_size in [1, 8192, 2_000_000] {
                let data = vec![0xff; size];
                let polls = Arc::new(AtomicUsize::new(0));
                let drops = Arc::new(AtomicUsize::new(0));
                let stream = Chunks {
                    chunks: data
                        .chunks(chunk_size)
                        .map(|c| Ok(Bytes::copy_from_slice(c)))
                        .collect(),
                    polls: polls.clone(),
                    drops: drops.clone(),
                };
                let (body, state) = capture_body(stream).await;
                assert_eq!(body.len(), size.min(MAX_UPSTREAM_ERROR_BODY_BYTES));
                assert_eq!(
                    state,
                    if size > 65536 {
                        ErrorBodyState::LimitExceeded
                    } else {
                        ErrorBodyState::Complete
                    }
                );
                let expected_polls = if size > 65536 {
                    65536 / chunk_size + 1
                } else {
                    size.div_ceil(chunk_size) + 1
                };
                assert_eq!(polls.load(Ordering::SeqCst), expected_polls);
                assert_eq!(drops.load(Ordering::SeqCst), 1);
            }
        }
        let utf8 = "中".repeat(21846).into_bytes();
        let (body, state) =
            capture_body(futures_util::stream::iter([Ok(Bytes::from(utf8.clone()))])).await;
        assert_eq!(&body[..], &utf8[..65536]);
        assert_eq!(state, ErrorBodyState::LimitExceeded);
        assert!(std::str::from_utf8(&body).is_err());
    }
    #[tokio::test]
    async fn read_failure_keeps_prefix() {
        for kind in [
            TransportErrorKind::Timeout,
            TransportErrorKind::Connection,
            TransportErrorKind::Other,
        ] {
            let (bytes, state) = capture_body(futures_util::stream::iter([
                Ok(Bytes::from_static(b"prefix")),
                Err(kind),
                Ok(Bytes::from_static(b"never")),
            ]))
            .await;
            assert_eq!(&bytes[..], b"prefix");
            assert_eq!(state, ErrorBodyState::ReadFailed { kind });
        }
    }
    #[test]
    fn status_envelope_and_formatting_never_include_upstream_data() {
        for phase in [PdPhase::Prefill, PdPhase::Decode] {
            for code in 100..=599 {
                let status = StatusCode::from_u16(code).unwrap();
                if status.is_success() {
                    continue;
                }
                let mut headers = HeaderMap::new();
                headers.insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("secret-host"),
                );
                let error = PdRequestError::UpstreamHttp(Box::new(UpstreamHttpError {
                    phase,
                    status,
                    headers,
                    body: Bytes::from_static(b"Traceback /private/secret secret-host"),
                    body_state: ErrorBodyState::Complete,
                }));
                assert_eq!(error.status(), status);
                for formatted in [
                    format!("{error}"),
                    format!("{error:?}"),
                    crate::routers::http::pd_types::error_chain(&error),
                ] {
                    assert!(!formatted.contains("secret"));
                    assert!(formatted.len() < 1024);
                }
                assert_eq!(std::error::Error::source(&error).map(|_| ()), None);
            }
            for (kind, code) in [
                (TransportErrorKind::Timeout, 504),
                (TransportErrorKind::Connection, 502),
                (TransportErrorKind::Other, 502),
            ] {
                assert_eq!(
                    PdRequestError::Transport(TransportError { phase, kind })
                        .status()
                        .as_u16(),
                    code
                );
            }
            assert_eq!(
                PdRequestError::protocol(phase).status(),
                StatusCode::INTERNAL_SERVER_ERROR
            );
        }
        assert_eq!(
            PdRequestError::Internal(InternalErrorReason::SerializeRequest).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
    #[tokio::test]
    async fn capture_and_renderer_enforce_header_allowlist() {
        for (retry, keep) in [
            ("5", true),
            ("Sun, 06 Nov 1994 08:49:37 GMT", true),
            ("secret-host", false),
        ] {
            for size in [0, 65537] {
                let upstream = http::Response::builder()
                    .status(429)
                    .header("retry-after", retry)
                    .header("content-type", "text/html")
                    .header("content-length", "999999")
                    .header("content-encoding", "gzip")
                    .header("set-cookie", "secret")
                    .header("www-authenticate", "secret")
                    .header("connection", "keep-alive")
                    .header("x-request-id", "secret-host")
                    .body(Bytes::from(vec![0xff; size]))
                    .unwrap();
                let error =
                    capture_upstream_error(PdPhase::Decode, reqwest::Response::from(upstream))
                        .await;
                let PdRequestError::UpstreamHttp(ref e) = error else {
                    panic!("expected HTTP error");
                };
                assert_eq!(e.headers.len(), if keep { 2 } else { 1 });
                let response = error.into_response();
                assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
                assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
                assert_eq!(response.headers().contains_key(header::RETRY_AFTER), keep);
                if keep {
                    assert_eq!(response.headers()[header::RETRY_AFTER], retry);
                }
                for header in [
                    "content-length",
                    "content-encoding",
                    "set-cookie",
                    "www-authenticate",
                    "connection",
                    "x-request-id",
                ] {
                    assert!(!response.headers().contains_key(header));
                }
            }
        }
    }

    #[tokio::test]
    async fn failed_prefill_and_future_drop_release_pending_legs() {
        use futures_util::FutureExt;
        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let guard = Guard(drops.clone());
        let d = futures_util::future::pending::<Result<reqwest::Response, reqwest::Error>>().map(
            move |result| {
                drop(guard);
                result
            },
        );
        let result = prefill_first(
            async { Err::<(), _>(PdRequestError::protocol(PdPhase::Prefill)) },
            d,
        )
        .await;
        assert!(matches!(
            result,
            Err(PdRequestError::Protocol {
                phase: PdPhase::Prefill
            })
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let p_guard = Guard(drops.clone());
        let d_guard = Guard(drops.clone());
        let p = futures_util::future::pending::<Result<(), PdRequestError>>().map(move |r| {
            drop(p_guard);
            r
        });
        let d = futures_util::future::pending::<Result<reqwest::Response, reqwest::Error>>().map(
            move |r| {
                drop(d_guard);
                r
            },
        );
        let mut request = Box::pin(prefill_first(p, d));
        assert!(futures_util::poll!(&mut request).is_pending());
        drop(request);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn retry_after_is_strict() {
        for value in [
            "0",
            "5",
            "99999999999999999999",
            "Sun, 06 Nov 1994 08:49:37 GMT",
            "Sunday, 06-Nov-94 08:49:37 GMT",
            "Sun Nov  6 08:49:37 1994",
        ] {
            assert!(
                valid_retry_after(&HeaderValue::from_str(value).unwrap()),
                "{value}"
            );
        }
        for value in [
            "",
            "-1",
            "+5",
            " 5",
            "5 ",
            "1.2",
            "retry-secret-host",
            "Sun, 99 Nov 1994 08:49:37 GMT",
        ] {
            assert!(
                !valid_retry_after(&HeaderValue::from_str(value).unwrap()),
                "{value}"
            );
        }
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn raw_response(
        status: u16,
        body: Vec<u8>,
        declared: usize,
        stall: bool,
    ) -> (reqwest::Response, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                socket.read_exact(&mut byte).await.unwrap();
                request.extend_from_slice(&byte);
                assert!(request.len() < 8192);
            }
            socket.write_all(format!("HTTP/1.1 {status} Error\r\nContent-Length: {declared}\r\nRetry-After: 5\r\nContent-Type: text/html\r\nContent-Encoding: gzip\r\nSet-Cookie: secret\r\n\r\n").as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
            if stall {
                std::future::pending::<()>().await;
            }
        });
        let response = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(150))
            .build()
            .unwrap()
            .get(url)
            .send()
            .await
            .unwrap();
        (response, task)
    }

    #[tokio::test]
    async fn real_http_read_reset_timeout_and_exact_limit_deadline() {
        for status in [429, 503] {
            for (size, declared, stall, expected) in [
                (
                    6,
                    7,
                    false,
                    ErrorBodyState::ReadFailed {
                        kind: TransportErrorKind::Other,
                    },
                ),
                (
                    6,
                    7,
                    true,
                    ErrorBodyState::ReadFailed {
                        kind: TransportErrorKind::Timeout,
                    },
                ),
                (
                    65536,
                    65537,
                    true,
                    ErrorBodyState::ReadFailed {
                        kind: TransportErrorKind::Timeout,
                    },
                ),
                (65536, 65536, false, ErrorBodyState::Complete),
                (65537, 65537, false, ErrorBodyState::LimitExceeded),
            ] {
                let (response, task) =
                    raw_response(status, vec![0xff; size], declared, stall).await;
                let error = capture_upstream_error(PdPhase::Prefill, response).await;
                task.abort();
                let PdRequestError::UpstreamHttp(ref e) = error else {
                    panic!("HTTP status lost");
                };
                assert_eq!(e.body.len(), size.min(65536));
                assert_eq!(e.body_state, expected);
                assert_eq!(e.status.as_u16(), status);
                let response = error.into_response();
                assert_eq!(response.status().as_u16(), status);
                assert_eq!(response.headers()[header::RETRY_AFTER], "5");
                assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
                assert!(!response.headers().contains_key(header::CONTENT_ENCODING));
                assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
                assert!(!response.headers().contains_key(header::SET_COOKIE));
            }
        }
    }

    #[tokio::test]
    async fn real_transport_classification_and_builder_errors() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let error = client.get(&url).send().await.unwrap_err();
        assert_eq!(classify_transport(&error), TransportErrorKind::Connection);
        assert_eq!(
            PdRequestError::from_reqwest(PdPhase::Decode, error).status(),
            StatusCode::BAD_GATEWAY
        );
        let error = client.get("invalid secret path").send().await.unwrap_err();
        assert!(error.is_builder());
        assert!(matches!(
            PdRequestError::from_reqwest(PdPhase::Prefill, error),
            PdRequestError::Internal(InternalErrorReason::RequestBuilder { .. })
        ));
    }
}

/// Record at the legacy response-header/error boundary, at most once.
/// Dropping an unfinished request does not add a terminal observation.
pub(super) struct RequestMetrics<'a> {
    route: &'a str,
    start: std::time::Instant,
    recorded: bool,
    count_errors: bool,
}
impl<'a> RequestMetrics<'a> {
    pub(super) fn new(route: &'a str) -> Self {
        Self {
            route,
            start: std::time::Instant::now(),
            recorded: false,
            count_errors: true,
        }
    }
    pub(super) fn exclude_legacy_error(&mut self) {
        self.count_errors = false;
    }
    pub(super) fn record_error(&mut self) -> bool {
        self.count_errors && self.record()
    }
    /// Returns true only for the first observation. Response-body failures
    /// must not count the request or its worker error a second time.
    pub(super) fn record(&mut self) -> bool {
        if self.recorded {
            return false;
        }
        self.recorded = true;
        crate::metrics::RouterMetrics::record_pd_request(self.route);
        crate::metrics::RouterMetrics::record_pd_request_duration(self.route, self.start.elapsed());
        true
    }
}
