/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use log::error;
use std::error::Error;
use std::fmt::Write;

use pyo3::exceptions::PyException;
use pyo3::PyErr;

pyo3::create_exception!(
    s3torchconnectorclient._mountpoint_s3_client,
    S3Exception,
    PyException
);

fn log_error(message: &str) {
    error!("ERROR: {}", message);
}

pub fn python_exception(error: impl Error) -> PyErr {
    let mut s = String::new();
    let mut error: &dyn Error = &error;

    write!(&mut s, "{}", error).unwrap();
    while let Some(next) = error.source() {
        error = next;
        write!(&mut s, ": {}", error).unwrap();
    }

    let py_err = S3Exception::new_err(s);
    let py_err_str = format!("{}", py_err);
    log_error(&py_err_str);
    py_err
}

#[cfg(test)]
mod tests {
    use std::io;

    use crate::exception::python_exception;

    #[test]
    fn test_python_exception() {
        pyo3::prepare_freethreaded_python();

        let err = io::Error::new(io::ErrorKind::InvalidData, "Test message");
        let pyerr = python_exception(err);

        assert_eq!(pyerr.to_string(), "S3Exception: Test message");
    }
}
