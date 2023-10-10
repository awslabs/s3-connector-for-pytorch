use std::error::Error;
use std::fmt::Write;
use std::io;

use pyo3::exceptions::PyException;
use pyo3::PyErr;

pyo3::create_exception!(_s3dataset, S3DatasetException, PyException);

pub fn python_exception(error: impl Error) -> PyErr {
    let mut s = String::new();
    let mut error: &dyn Error = &error;

    write!(&mut s, "{}", error).unwrap();
    while let Some(next) = error.source() {
        error = next;
        write!(&mut s, ": {}", error).unwrap();
    }

    S3DatasetException::new_err(s)
}

#[test]
fn test_python_exception() {
    pyo3::prepare_freethreaded_python();

    let err = io::Error::new(io::ErrorKind::InvalidData,"Test message");
    let pyerr = python_exception(err);

    assert_eq!(pyerr.to_string(), "S3DatasetException: Test message");
}
