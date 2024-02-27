/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */
use std::{env, io::Write};
use std::str::FromStr;
use env_logger::{Builder, Target};
use log::{LevelFilter};
use mountpoint_s3_crt::common::rust_log_adapter::RustLogAdapter;
use pyo3::{PyResult};
use chrono::Local;
use pyo3_log::Logger;
use crate::exception::python_exception;

pub const ENABLE_CRT_LOGS_ENV_VAR: &str = "ENABLE_CRT_LOGS";
const DATE_FORMAT_STR: &str = "%Y-%m-%d %H:%M:%S%.3f";

pub fn setup_logging() -> PyResult<()> {
    let enable_crt_logs = env::var(ENABLE_CRT_LOGS_ENV_VAR)
        .unwrap_or(String::from("OFF"))
        .to_uppercase();

    match enable_crt_logs.as_str() {
        "OFF" => enable_default_logging(),
        level_filter_str => enable_crt_logging(level_filter_str)
    }
}

fn enable_crt_logging(level_filter_str: &str) -> PyResult<()> {
    let level_filter = LevelFilter::from_str(level_filter_str)
        .map_err(python_exception)?;

    RustLogAdapter::try_init().map_err(python_exception)?;

    let mut builder = Builder::new();
    builder
        .format(|buf, record| {
            writeln!(buf, "{} {} {} {} {} {}",
                     record.level(), record.target(),
                     Local::now().format(DATE_FORMAT_STR),
                     record.module_path().unwrap(),
                     record.line().unwrap(),
                     record.args())
        })
        .target(Target::Stdout)
        .format_module_path(false)
        .filter_module(
            "mountpoint_s3_client::s3_crt_client",
            LevelFilter::Off,
        )
        .filter_level(level_filter);

    builder.try_init().map_err(python_exception)
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

        #[test]
        fn test_invalid_values() {
            pyo3::prepare_freethreaded_python();
            let invalid_values = ["invalid", "", "\n", "123", "xyz"];
            for value in invalid_values.iter() {
                env::set_var(ENABLE_CRT_LOGS_ENV_VAR, *value);
                let error_result: PyResult<()> = setup_logging();
                assert!(error_result.is_err());
                let pyerr = error_result.err().unwrap();
                assert_eq!(pyerr.to_string(), "S3Exception: attempted to convert a string that doesn't match an existing log level");
            }
        }
    }

    fn check_valid_log_level(log_level: &str) {
        pyo3::prepare_freethreaded_python();
        env::set_var(ENABLE_CRT_LOGS_ENV_VAR, log_level);
        let result: PyResult<()> = setup_logging();
        assert!(result.is_ok());
    }
}
