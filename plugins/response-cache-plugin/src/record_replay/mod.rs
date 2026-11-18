// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless model traces with live tools and fail-closed model interception.
//!
//! When `record_replay` is configured it replaces the ordinary cache. `record`
//! calls the provider and writes a trace; `verify` and `replay` serve the
//! recorded answers with their original timing and never call the provider.

pub mod config;
mod key;
mod session;
mod stream;
#[cfg(all(test, unix))]
#[path = "../../tests/unit/record_replay_tests.rs"]
mod tests;
mod tool;
mod trace;

use std::sync::Arc;
use std::time::Instant;

use nemo_relay_types::api::runtime::{LlmExecutionFn, LlmStreamExecutionFn, ToolExecutionFn};

pub(crate) const PRIORITY: i32 = i32::MAX;

/// When a stream's caller received each chunk and when it stopped reading.
#[derive(Debug, Clone)]
pub(crate) struct ConsumerReport {
    /// Times the caller received each chunk, in order.
    pub delivered: Vec<Instant>,
    /// True only when the caller read the final item.
    pub complete: bool,
}

/// Caller-side timing for one stream, from Relay's stream consumer clock.
///
/// Relay provides it for Python callers while a recorder holds the timing
/// guard. Other callers have no clock and are timed at the producer.
pub(crate) trait ConsumerClock: Send + Sync {
    /// Whether chunk `index` became available before the caller asked for it.
    fn was_prefetched(&self, index: usize) -> bool;
    /// Calls `observer` once, when the caller finishes reading.
    fn observe(self: Arc<Self>, observer: Box<dyn FnOnce(ConsumerReport) + Send>);
}

/// Captures the clock of the stream call that is running now.
pub(crate) type ClockSource = Arc<dyn Fn() -> Option<Arc<dyn ConsumerClock>> + Send + Sync>;

/// [`ConsumerClock`] backed by the Relay plugin runtime.
pub(crate) struct RuntimeClock(pub(crate) nemo_relay_plugin::StreamConsumerClock);

impl ConsumerClock for RuntimeClock {
    fn was_prefetched(&self, index: usize) -> bool {
        // An unreadable clock counts as prefetched, which the recorder treats
        // as producer timing.
        self.0.was_prefetched(index).unwrap_or(true)
    }

    fn observe(self: Arc<Self>, observer: Box<dyn FnOnce(ConsumerReport) + Send>) {
        // The stream intercept runs on the plugin's Tokio runtime.
        tokio::spawn(async move {
            if let Ok(report) = self.0.finished().await {
                observer(ConsumerReport {
                    delivered: report.delivered,
                    complete: report.complete,
                });
            }
        });
    }
}

/// Model and tool intercepts for one record/replay session.
pub(crate) struct Activation {
    pub llm: LlmExecutionFn,
    pub stream: LlmStreamExecutionFn,
    pub tool: ToolExecutionFn,
}

/// Closes the session when the last intercept holding it is dropped.
///
/// Relay releases a component's callbacks when the component closes, so this
/// runs once at deactivation.
struct CloseOnDrop(Arc<session::Session>);

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

pub(crate) fn activate(
    config: crate::config::ResponseCacheConfig,
    diagnostics: Vec<String>,
    clocks: ClockSource,
) -> Activation {
    let session = session::Session::activate_with_diagnostics(config, diagnostics);
    let close = Arc::new(CloseOnDrop(session.clone()));
    let llm = stream::make_buffered(session.clone());
    let streaming = stream::make_stream(session.clone(), clocks);
    let tools = tool::make_tool(session);
    let (llm_close, stream_close, tool_close) = (close.clone(), close.clone(), close);
    Activation {
        llm: Arc::new(move |provider, request, context, next| {
            let _close = &llm_close;
            llm(provider, request, context, next)
        }),
        stream: Arc::new(move |provider, request, context, next| {
            let _close = &stream_close;
            streaming(provider, request, context, next)
        }),
        tool: Arc::new(move |context, next| {
            let _close = &tool_close;
            tools(context, next)
        }),
    }
}
