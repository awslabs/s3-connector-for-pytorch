use futures::executor::block_on;
use futures::TryStreamExt;
use mountpoint_s3_client::S3GetObjectRequest;
use pyo3::{py_run, pyclass, PyErr, pymethods, PyRef, PyRefMut, PyResult, Python};
use pyo3::types::PyBytes;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::exception::{python_exception, S3DatasetException};
use crate::mountpoint_s3_client::MountpointS3Client;

#[pyclass(name = "GetObjectStream", module="_s3dataset")]
#[derive(Debug)]
pub struct GetObjectStream {
    request: S3GetObjectRequest,
    offset: u64,
    #[pyo3(get)]
    bucket: String,
    #[pyo3(get)]
    key: String
}

impl GetObjectStream {
    pub(crate) fn new(request: S3GetObjectRequest, bucket: String, key: String) -> Self {
        Self {
            request,
            offset: 0,
            bucket,
            key,
        }
    }
}

#[pymethods]
impl GetObjectStream {
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<&PyBytes>> {
        let py = slf.py();
        let request = &mut slf.request;

        let body_part = py.allow_threads(|| {
            block_on(request.try_next()).map_err(python_exception)
        })?;
        match body_part {
            None => Ok(None),
            Some((offset, data)) => {
                if offset != slf.offset {
                    return Err(S3DatasetException::new_err("Data from S3 was returned out of order!"));
                }
                slf.offset += data.len() as u64;
                let data = PyBytes::new(slf.py(), data.as_ref());
                Ok::<Option<&PyBytes>, PyErr>(Some(data))
            }
        }
    }
}

#[test]
fn test_get_object() -> PyResult<()> {
    let layer = tracing_subscriber::fmt::layer().with_ansi(true);
    let registry = tracing_subscriber::registry().with(layer);
    let _ = registry.try_init();

    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let crt_client = py.get_type::<MountpointS3Client>();
        py_run!(py, crt_client, r#"
client = crt_client("us-east-1")
stream = client.get_object("s3dataset-testing", "hello_world.txt")
full_data = b''.join(stream)
assert full_data == b"Hello, World!\n"
"#);
    });

    Ok(())
}
