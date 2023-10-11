use std::sync::Arc;

use futures::executor::block_on;
use mountpoint_s3_client::{ObjectClient, S3CrtClient};
use mountpoint_s3_client::types::ListObjectsResult;
use pyo3::{py_run, pyclass, pymethods, PyRef, PyRefMut, PyResult, Python};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::exception::python_exception;
use crate::mountpoint_s3_client::MountpointS3Client;
use crate::python_structs::py_list_object_result::PyListObjectResult;
use crate::python_structs::py_object_info::PyObjectInfo;

#[pyclass(name = "ListObjectStream", module="_s3dataset")]
#[derive(Debug)]
pub struct ListObjectStream {
    client: Arc<S3CrtClient>,
    continuation_token: Option<String>,
    complete: bool,
    #[pyo3(get)]
    bucket: String,
    #[pyo3(get)]
    prefix: String,
    #[pyo3(get)]
    delimiter: String,
    #[pyo3(get)]
    max_keys: usize,
}

impl ListObjectStream {
    pub(crate) fn new(client: Arc<S3CrtClient>, bucket: String, prefix: String, delimiter: String, max_keys: usize) -> Self {
        Self { client, bucket, prefix, delimiter, max_keys, continuation_token: None, complete: false }
    }
}

#[pymethods]
impl ListObjectStream {
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyListObjectResult>> {
        if slf.complete {
            return Ok(None);
        }
        let results = make_request(&slf)?;

        slf.continuation_token = results.next_continuation_token;
        if slf.continuation_token.is_none() {
            slf.complete = true;
        }

        let objects = results.objects.into_iter().map(|obj| PyObjectInfo::new(obj)).collect();

        Ok(Some(PyListObjectResult::new(objects, results.common_prefixes)))
    }
}

fn make_request(slf: &PyRefMut<'_, ListObjectStream>) -> PyResult<ListObjectsResult> {
    let client = &slf.client;
    let bucket = &slf.bucket;
    let continuation_token = slf.continuation_token.as_deref();
    let delimiter = &slf.delimiter;
    let prefix = &slf.prefix;
    let max_keys = slf.max_keys;

    slf.py().allow_threads(|| {
        block_on(
            client.list_objects(bucket, continuation_token, delimiter, max_keys, prefix)
        ).map_err(python_exception)
    })
}

#[test]
fn test_list_objects() -> PyResult<()> {
    let layer = tracing_subscriber::fmt::layer().with_ansi(true);
    let registry = tracing_subscriber::registry().with(layer);
    let _ = registry.try_init();

    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let crt_client = py.get_type::<MountpointS3Client>();
        py_run!(py, crt_client, r#"
client = crt_client("us-east-1")
stream = client.list_objects("s3dataset-testing")

object_infos = [object_info for page in stream for object_info in page.object_info]
keys = {object_info.key for object_info in object_infos}
assert keys == {"hello_world.txt"}
"#);
    });

    Ok(())
}
