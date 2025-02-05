/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */
use crate::exception::python_exception;
use mountpoint_s3_client::config::RustLogAdapter;
use pyo3::PyResult;
use std::env;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::util::SubscriberInitExt;

pub const S3_TORCH_CONNECTOR_DEBUG_LOGS_ENV_VAR: &str = "S3_TORCH_CONNECTOR_DEBUG_LOGS";
pub const S3_TORCH_CONNECTOR_LOGS_DIR_PATH_ENV_VAR: &str = "S3_TORCH_CONNECTOR_LOGS_DIR_PATH";
pub const LOG_FILE_PREFIX: &str = "s3torchconnectorclient.log";

pub fn setup_logging() -> PyResult<()> {
    let enable_logs = env::var(S3_TORCH_CONNECTOR_DEBUG_LOGS_ENV_VAR);

    if enable_logs.is_ok() {
        let filter = EnvFilter::try_from_env(S3_TORCH_CONNECTOR_DEBUG_LOGS_ENV_VAR)
            .map_err(python_exception)?;
        let debug_logs_path = env::var(S3_TORCH_CONNECTOR_LOGS_DIR_PATH_ENV_VAR).ok();

        RustLogAdapter::try_init().map_err(python_exception)?;

        match debug_logs_path {
            Some(logs_path) => {
                enable_file_logging(filter, logs_path)?;
            }
            None => {
                enable_default_logging(filter)?;
            }
        }
    }

    Ok(())
}

fn enable_file_logging(filter: EnvFilter, logs_path: String) -> PyResult<()> {
    let logfile = tracing_appender::rolling::hourly(logs_path, LOG_FILE_PREFIX);
    let subscriber_builder = tracing_subscriber::fmt()
        .with_writer(logfile)
        .with_env_filter(filter)
        .with_ansi(false);
    subscriber_builder
        .finish()
        .try_init()
        .map_err(python_exception)?;

    Ok(())
}

fn enable_default_logging(filter: EnvFilter) -> PyResult<()> {
    let subscriber_builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false);
    subscriber_builder
        .finish()
        .try_init()
        .map_err(python_exception)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::logger_setup::{
        setup_logging, S3_TORCH_CONNECTOR_DEBUG_LOGS_ENV_VAR,
        S3_TORCH_CONNECTOR_LOGS_DIR_PATH_ENV_VAR,
    };
    use pyo3::PyResult;
    use rusty_fork::rusty_fork_test;
    use std::env;

    fn check_valid_log_level(log_level: &str) {
        pyo3::prepare_freethreaded_python();
        env::set_var(S3_TORCH_CONNECTOR_DEBUG_LOGS_ENV_VAR, log_level);
        let result: PyResult<()> = setup_logging();
        assert!(result.is_ok());
    }

    rusty_fork_test! {
        #[test]
        fn test_debug_log_environment_variable_unset() {
            pyo3::prepare_freethreaded_python();
            env::remove_var(S3_TORCH_CONNECTOR_DEBUG_LOGS_ENV_VAR);
            let result: PyResult<()> = setup_logging();
            assert!(result.is_ok());
        }

        #[test]
        fn test_logs_dir_environment_variable_unset() {
            pyo3::prepare_freethreaded_python();
            env::remove_var(S3_TORCH_CONNECTOR_LOGS_DIR_PATH_ENV_VAR);
            let result: PyResult<()> = setup_logging();
            assert!(result.is_ok());
        }

        #[test]
        fn test_debug_logging_off() {
            check_valid_log_level("OFF");
        }

        #[test]
        fn test_debug_logging_level_error() {
            check_valid_log_level("ERROR");
        }

        #[test]
        fn test_debug_logging_level_warn() {
            check_valid_log_level("WARN");
        }

        #[test]
        fn test_debug_logging_level_info() {
            check_valid_log_level("INFO");
        }

        #[test]
        fn test_debug_logging_level_debug() {
            check_valid_log_level("debug");
        }

        #[test]
        fn test_debug_logging_level_trace() {
            check_valid_log_level("trace");
        }

        #[test]
        fn test_invalid_logging_level() {
            pyo3::prepare_freethreaded_python();
            env::set_var(S3_TORCH_CONNECTOR_DEBUG_LOGS_ENV_VAR, "invalid123.&/?");
            let result: PyResult<()> = setup_logging();
            assert!(result.is_err());
        }
    }
}
