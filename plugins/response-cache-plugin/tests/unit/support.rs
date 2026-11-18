// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host used by unit tests in place of a Relay process.
//!
//! It prices with the in-process `nemo-relay-model-pricing` resolver, which
//! the tests configure, and records marks instead of emitting them.

use std::sync::{Arc, Mutex, OnceLock};

use nemo_relay_plugin::{AnnotatedLlmResponse, CostEstimate, Json, Usage};

use nemo_relay_runtime::api::runtime::stream_consumer_clock::{
    StreamConsumerClock, StreamConsumerClockRegistration,
};

use crate::host::Host;
use crate::record_replay::{ConsumerClock, ConsumerReport};

/// One recorded mark: its name, data, and metadata.
pub(crate) type RecordedMark = (String, Json, Json);

#[derive(Default)]
pub(crate) struct TestHost {
    marks: Mutex<Vec<RecordedMark>>,
}

impl TestHost {
    /// Marks recorded so far, oldest first.
    #[allow(dead_code)]
    pub(crate) fn marks(&self) -> Vec<RecordedMark> {
        self.marks
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

impl Host for TestHost {
    fn emit_mark(&self, name: &str, data: &Json, metadata: &Json) {
        self.marks
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push((name.to_owned(), data.clone(), metadata.clone()));
    }

    fn estimate_cost_for_provider(
        &self,
        provider: Option<&str>,
        model: &str,
        usage: &Usage,
    ) -> Option<CostEstimate> {
        nemo_relay_model_pricing::estimate_cost_for_provider(provider, model, usage)
    }

    fn response_cost_for_provider(
        &self,
        response: &AnnotatedLlmResponse,
        provider: Option<&str>,
    ) -> Option<CostEstimate> {
        let mut response = response.clone();
        nemo_relay_model_pricing::attach_estimated_cost_for_provider(&mut response, provider);
        response.usage.and_then(|usage| usage.cost)
    }

    fn enable_stream_consumer_timing(&self) -> Option<Box<dyn Send + Sync>> {
        Some(Box::new(StreamConsumerClockRegistration::default()))
    }

    fn capture_stream_consumer_clock(&self) -> Option<Arc<dyn ConsumerClock>> {
        StreamConsumerClock::current()
            .map(|clock| Arc::new(TestClock(clock)) as Arc<dyn ConsumerClock>)
    }

    fn warn(&self, _event: &str, _message: &str) {}
}

/// Relay's in-process clock, standing in for the runtime operations.
pub(crate) struct TestClock(pub(crate) StreamConsumerClock);

impl ConsumerClock for TestClock {
    fn was_prefetched(&self, index: usize) -> bool {
        self.0.was_prefetched(index, std::time::Instant::now())
    }

    fn observe(self: Arc<Self>, observer: Box<dyn FnOnce(ConsumerReport) + Send>) {
        self.0.observe(move |report| {
            observer(ConsumerReport {
                delivered: report.delivered,
                complete: report.complete,
            })
        });
    }
}

/// The process-wide test host, used whenever no runtime host is installed.
pub(crate) fn host() -> Arc<TestHost> {
    static HOST: OnceLock<Arc<TestHost>> = OnceLock::new();
    Arc::clone(HOST.get_or_init(Arc::default))
}
