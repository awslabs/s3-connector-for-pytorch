use std::sync::Arc;

use futures::executor::block_on;
use futures::TryStreamExt;
use mountpoint_s3_client::mock_client::{MockClient, MockClientConfig, MockObject};
use mountpoint_s3_client::types::ListObjectsResult;
use mountpoint_s3_client::ObjectClient;
use pyo3::{pyclass, pymethods, PyResult, Python};

use crate::exception::python_exception;
use crate::mountpoint_clients::mountpoint_s3_client_wrapper::{
    MPGetObjectClosure, MountpointS3ClientWrapper,
};

#[derive(Clone)]
#[pyclass(name = "MountpointS3ClientMock", module = "_s3dataset", frozen)]
pub struct MountpointS3ClientMock {
    client: Arc<MockClient>,
    pub region: String,
    pub throughput_target_gbps: f64,
    pub part_size: usize,
}

#[pymethods]
impl MountpointS3ClientMock {
    #[new]
    #[pyo3(signature = (region, bucket, throughput_target_gbps = 10.0, part_size = 8 * 1024 * 1024))]
    pub fn new(
        region: String,
        bucket: String,
        throughput_target_gbps: f64,
        part_size: usize,
    ) -> Self {
        let config = MockClientConfig { bucket, part_size };
        let client = Arc::new(MockClient::new(config));

        Self {
            client,
            region,
            throughput_target_gbps,
            part_size,
        }
    }

    pub fn add_object(&self, key: String, data: Vec<u8>) {
        self.client.add_object(&key, MockObject::from(data));
    }

    pub fn remove_object(&self, key: String) {
        self.client.remove_object(&key);
    }
}

impl MountpointS3ClientWrapper for MountpointS3ClientMock {
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
