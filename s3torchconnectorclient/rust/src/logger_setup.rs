/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */
use std::{env};
use log::{LevelFilter};
use mountpoint_s3_crt::common::rust_log_adapter::RustLogAdapter;
use pyo3::{PyResult};
use pyo3_log::Logger;
use tracing_subscriber::{filter::EnvFilter};
use tracing_subscriber::util::{SubscriberInitExt};
use crate::exception::python_exception;

pub const ENABLE_CRT_LOGS_ENV_VAR: &str = "ENABLE_CRT_LOGS";
pub const CRT_LOGS_DIR_PATH_ENV_VAR: &str = "CRT_LOGS_DIR_PATH";

pub fn setup_logging() -> PyResult<()> {
    let enable_crt_logs = env::var(ENABLE_CRT_LOGS_ENV_VAR);

    if enable_crt_logs.is_ok() {
        enable_crt_logging()
    } else {
        enable_default_logging()
    }
}

fn enable_crt_logging() -> PyResult<()> {
    RustLogAdapter::try_init().map_err(python_exception)?;

    let filter = EnvFilter::try_from_env(ENABLE_CRT_LOGS_ENV_VAR).map_err(python_exception)?;
    let crt_logs_path = env::var(CRT_LOGS_DIR_PATH_ENV_VAR).ok();

    match crt_logs_path {
        Some(logs_path) => {
            let logfile = tracing_appender::rolling::hourly(logs_path, "s3torchconnectorclient.log");
            let subscriber_builder = tracing_subscriber::fmt()
                .with_writer(logfile)
                .with_env_filter(filter)
                .with_ansi(false);
            subscriber_builder.finish().try_init().map_err(python_exception)?;
        },
        None => {
            let subscriber_builder = tracing_subscriber::fmt()
                 .with_env_filter(filter)
                 .with_ansi(false);
            subscriber_builder.finish().try_init().map_err(python_exception)?;
        }
    }

    Ok(())
}

fn enable_default_logging() -> PyResult<()> {
    let logger = Logger::default()
        .filter_target(
            "mountpoint_s3_client::s3_crt_client::request".to_owned(),
            LevelFilter::Off,
        )
        .filter(LevelFilter::Trace);

    logger.install().map_err(python_exception)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use rusty_fork::rusty_fork_test;
    use std::{env};
    use pyo3::PyResult;
    use crate::logger_setup::{ENABLE_CRT_LOGS_ENV_VAR, setup_logging};

    fn check_valid_log_level(log_level: &str) {
        pyo3::prepare_freethreaded_python();
        env::set_var(ENABLE_CRT_LOGS_ENV_VAR, log_level);
        let result: PyResult<()> = setup_logging();
        assert!(result.is_ok());
    }

    rusty_fork_test! {
        #[test]
        fn test_environment_variable_unset() {
            pyo3::prepare_freethreaded_python();
            env::remove_var(ENABLE_CRT_LOGS_ENV_VAR);
            let result: PyResult<()> = setup_logging();
            assert!(result.is_ok());
        }

        #[test]
        fn test_logging_off() {
            check_valid_log_level("OFF");
        }

        #[test]
        fn test_logging_level_error() {
            check_valid_log_level("ERROR");
        }

        #[test]
        fn test_logging_level_warn() {
            check_valid_log_level("WARN");
        }

        #[test]
        fn test_logging_level_info() {
            check_valid_log_level("INFO");
        }

        #[test]
        fn test_logging_level_debug() {
            check_valid_log_level("debug");
        }

        #[test]
        fn test_logging_level_trace() {
            check_valid_log_level("trace");
        }
    }
}
