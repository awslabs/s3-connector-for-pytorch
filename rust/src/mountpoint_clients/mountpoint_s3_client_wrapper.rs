use mountpoint_s3_client::types::ListObjectsResult;
use pyo3::{PyResult, Python};

pub type MPGetObjectClosure =
    Box<dyn FnMut(Python) -> PyResult<Option<(u64, Box<[u8]>)>> + Send + Sync>;

pub trait MountpointS3ClientWrapper {
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
