use std::sync::Arc;

use futures::executor::block_on;
use futures::TryStreamExt;
use mountpoint_s3_client::types::ListObjectsResult;
use mountpoint_s3_client::ObjectClient;
use pyo3::{PyResult, Python};

use crate::exception::python_exception;
use crate::get_object_stream::GetObjectStream;

pub type MPGetObjectClosure =
    Box<dyn FnMut(Python) -> PyResult<Option<(u64, Box<[u8]>)>> + Send + Sync>;

/// We need an extra trait here, as pyo3 doesn't support structs with generics.
/// However, it does allow dynamic traits. We can't ues `ObjectClient` itself, as that requires
/// us to specify the additional types (GetObjectResult, PutObjectResult, ClientError), which
/// we don't want to do.
pub(crate) trait MountpointS3ClientInner {
    fn get_object(&self, py: Python, bucket: String, key: String) -> PyResult<GetObjectStream>;
    fn list_objects(
        &self,
        bucket: &str,
        continuation_token: Option<&str>,
        delimiter: &str,
        max_keys: usize,
        prefix: &str,
    ) -> PyResult<ListObjectsResult>;
}

pub(crate) struct MountpointS3ClientInnerImpl<T: ObjectClient> {
    client: Arc<T>,
}

impl<T: ObjectClient> MountpointS3ClientInnerImpl<T> {
    pub fn new(client: Arc<T>) -> MountpointS3ClientInnerImpl<T> {
        MountpointS3ClientInnerImpl { client }
    }
}

impl<Client> MountpointS3ClientInner for MountpointS3ClientInnerImpl<Client>
where
    Client: ObjectClient,
    <Client as ObjectClient>::GetObjectResult: Sync + Send + Unpin + 'static,
{
    fn get_object(&self, py: Python, bucket: String, key: String) -> PyResult<GetObjectStream> {
        let request = self.client.get_object(&bucket, &key, None, None);

        // TODO - Look at use of `block_on` and see if we can future this.
        let mut request = py.allow_threads(|| block_on(request).map_err(python_exception))?;

        let closure = Box::new(move |py: Python| {
            py.allow_threads(|| block_on(request.try_next()).map_err(python_exception))
        });

        Ok(GetObjectStream::new(closure, bucket, key))
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
