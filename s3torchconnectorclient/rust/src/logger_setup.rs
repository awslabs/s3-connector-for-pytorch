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
        level_filter_str => enable_crt_logging(level_filter_str)?
    }

    Ok(())
}

fn enable_crt_logging(level_filter_str: &str) -> PyResult<()> {
    let level_filter = LevelFilter::from_str(level_filter_str)
        .map_err(python_exception)?;
    let _ = RustLogAdapter::try_init().map_err(python_exception);
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

    let _ = builder.try_init().map_err(python_exception);

    Ok(())
}

fn enable_default_logging() {
    let logger = Logger::default()
        .filter_target(
            "mountpoint_s3_client::s3_crt_client::request".to_owned(),
            LevelFilter::Off,
        )
        .filter(LevelFilter::Trace);
    let _ = logger.install().map_err(python_exception);
}

#[cfg(test)]
mod tests {
    use std::{env};
    use pyo3::PyResult;
    use crate::logger_setup::{ENABLE_CRT_LOGS_ENV_VAR, setup_logging};

    #[test]
    fn test_logging_setup() {
        pyo3::prepare_freethreaded_python();
        // Enforce serial execution as we modify the same environment variable
        check_environment_variable_unset();
        check_valid_values();
        check_invalid_values();
    }

    fn check_environment_variable_unset() {
        env::remove_var(ENABLE_CRT_LOGS_ENV_VAR);
        let result: PyResult<()> = setup_logging();
        assert!(result.is_ok());
    }

    fn check_valid_values() {
        let valid_values = ["OFF", "ERROR", "WARN", "INFO", "DEBUG", "TRACE", "debug"];
        for value in valid_values.iter() {
            env::set_var(ENABLE_CRT_LOGS_ENV_VAR, *value);
            let result: PyResult<()> = setup_logging();
            assert!(result.is_ok());
        }
    }

    fn check_invalid_values() {
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
