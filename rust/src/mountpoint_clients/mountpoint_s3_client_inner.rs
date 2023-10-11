use std::sync::Arc;

use futures::executor::block_on;
use futures::TryStreamExt;
use mountpoint_s3_client::types::ListObjectsResult;
use mountpoint_s3_client::{ObjectClient, S3CrtClient};
use pyo3::{PyResult, Python};

use crate::exception::python_exception;
use crate::mountpoint_clients::mountpoint_s3_client_wrapper::{
    MPGetObjectClosure, MountpointS3ClientWrapper,
};

pub struct MountpointS3ClientInner {
    client: Arc<S3CrtClient>,
}

impl MountpointS3ClientInner {
    pub fn new(client: S3CrtClient) -> MountpointS3ClientInner {
        MountpointS3ClientInner {
            client: Arc::new(client),
        }
    }
}

impl MountpointS3ClientWrapper for MountpointS3ClientInner {
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
