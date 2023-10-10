use std::sync::Arc;

use futures::executor::block_on;
use mountpoint_s3_client::ObjectClient;
use mountpoint_s3_client::types::ListObjectsResult;
use pyo3::{py_run, pyclass, pymethods, PyRef, PyRefMut, PyResult, Python};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::exception::python_exception;
use crate::mock_mountpoint_s3_client::MockMountpointS3Client;
use crate::python_structs::py_list_object_result::PyListObjectResult;
use crate::python_structs::py_object_info::PyObjectInfo;

#[pyclass(name = "ListObjectStream", module="_s3dataset")]
pub struct ListObjectStream {
    list_objects: Box<dyn Fn(&str, Option<&str>, &str, usize, &str) -> PyResult<ListObjectsResult> + Send + Sync>,
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
    pub(crate) fn new(client: Arc<impl ObjectClient + Send + Sync + 'static>, bucket: String, prefix: String, delimiter: String, max_keys: usize) -> Self {
        let list_objects = move |bucket: &_, continuation_token: Option<&str>, delimiter: &_, max_keys, prefix: &_| {
            block_on(
                client.list_objects(bucket, continuation_token, delimiter, max_keys, prefix)
            ).map_err(python_exception)
        };
        Self { list_objects: Box::new(list_objects), bucket, prefix, delimiter, max_keys, continuation_token: None, complete: false }
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
    let bucket = &slf.bucket;
    let continuation_token = slf.continuation_token.as_deref();
    let delimiter = &slf.delimiter;
    let prefix = &slf.prefix;
    let max_keys = slf.max_keys;

    let list_objects = &slf.list_objects;

    slf.py().allow_threads(|| {
        list_objects(bucket, continuation_token, delimiter, max_keys, prefix)
    })
}

#[test]
fn test_list_objects() -> PyResult<()> {
    let layer = tracing_subscriber::fmt::layer().with_ansi(true);
    let registry = tracing_subscriber::registry().with(layer);
    let _ = registry.try_init();

    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let crt_client = py.get_type::<MockMountpointS3Client>();
        py_run!(py, crt_client, r#"
expected_keys = {"test"}

client = crt_client("us-east-1", "mock-bucket")
for key in expected_keys:
    client.add_object(key, b"")

stream = client.list_objects("mock-bucket")

object_infos = [object_info for page in stream for object_info in page.object_info]
keys = {object_info.key for object_info in object_infos}
assert keys == expected_keys
"#);
    });

    Ok(())
}
