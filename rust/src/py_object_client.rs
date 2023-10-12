use std::sync::Arc;

use futures::executor::block_on;
use futures::TryStreamExt;
use mountpoint_s3_client::types::ListObjectsResult;
use mountpoint_s3_client::ObjectClient;
use pyo3::{PyResult, Python};

use crate::exception::python_exception;

pub type MPGetObjectClosure =
    Box<dyn FnMut(Python) -> PyResult<Option<(u64, Box<[u8]>)>> + Send + Sync>;

pub trait PyObjectClient {
    fn get_object(&self, py: Python, bucket: &str, key: &str) -> PyResult<MPGetObjectClosure>;
    fn list_objects(
        &self,
        bucket: &str,
        continuation_token: Option<&str>,
        delimiter: &str,
        max_keys: usize,
        prefix: &str,
    ) -> PyResult<ListObjectsResult>;
}

pub struct ObjectClientWrapper<T: ObjectClient> {
    client: Arc<T>,
}

impl<T: ObjectClient> ObjectClientWrapper<T> {
    pub fn new(client: Arc<T>) -> ObjectClientWrapper<T> {
        ObjectClientWrapper { client }
    }
}

impl<T> PyObjectClient for ObjectClientWrapper<T>
where
    T: ObjectClient,
    <T as ObjectClient>::GetObjectResult: Unpin,
    <T as ObjectClient>::GetObjectResult: Sync,
    <T as ObjectClient>::GetObjectResult: Send,
    <T as ObjectClient>::GetObjectResult: 'static,
{
    fn get_object(&self, py: Python, bucket: &str, key: &str) -> PyResult<MPGetObjectClosure> {
        let request = self.client.get_object(bucket, key, None, None);

        // TODO - Look at use of `block_on` and see if we can future this.
        let mut request = py.allow_threads(|| block_on(request).map_err(python_exception))?;

        Ok(Box::new(move |py: Python| {
            py.allow_threads(|| block_on(request.try_next()).map_err(python_exception))
        }))
    }

    fn list_objects(
        &self,
        bucket: &str,
        continuation_token: Option<&str>,
        delimiter: &str,
        max_keys: usize,
        prefix: &str,
    ) -> PyResult<ListObjectsResult> {
        block_on(
            self.client
                .list_objects(bucket, continuation_token, delimiter, max_keys, prefix),
        )
        .map_err(python_exception)
    }
}
