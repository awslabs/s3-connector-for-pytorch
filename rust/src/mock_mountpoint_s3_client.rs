use std::sync::Arc;

use futures::executor::block_on;
use futures::TryStreamExt;
use mountpoint_s3_client::mock_client::{MockClient, MockClientConfig, MockObject};
use mountpoint_s3_client::ObjectClient;
use pyo3::{pyclass, pymethods, PyRef, PyResult, Python};

use crate::exception::python_exception;
use crate::get_object_stream::GetObjectStream;
use crate::list_object_stream::ListObjectStream;

#[pyclass(name = "MockMountpointS3Client", module="_s3dataset", frozen)]
#[derive(Debug)]
pub struct MockMountpointS3Client {
    client: Arc<MockClient>,
    #[pyo3(get)]
    throughput_target_gbps: f64,
    #[pyo3(get)]
    region: String,
    #[pyo3(get)]
    part_size: usize
}

#[pymethods]
impl MockMountpointS3Client {
    #[new]
    #[pyo3(signature = (region, bucket, throughput_target_gbps=10.0, part_size=8*1024*1024))]
    pub fn new(region: String, bucket: String, throughput_target_gbps: f64, part_size: usize) -> PyResult<Self> {
        let config = MockClientConfig { bucket, part_size };
        let client = Arc::new(MockClient::new(config));
        Ok(Self { client, throughput_target_gbps, part_size, region })
    }

    pub fn get_object(slf: PyRef<'_, Self>, bucket: String, key: String) -> PyResult<GetObjectStream> {
        let request = slf.client.get_object(&bucket, &key, None, None);

        let mut request = slf.py().allow_threads(|| {block_on(request).map_err(python_exception)})?;
        let next_part = move |py: Python| {
            py.allow_threads(|| {
                block_on(request.try_next()).map_err(python_exception)
            })
        };

        Ok(GetObjectStream::new(Box::new(next_part), bucket, key))
    }

    #[pyo3(signature = (bucket, prefix=String::from(""), delimiter=String::from(""), max_keys=1000))]
    pub fn list_objects(&self, bucket: String, prefix: String, delimiter: String, max_keys: usize) -> PyResult<ListObjectStream> {
        Ok(ListObjectStream::new(self.client.clone(), bucket, prefix, delimiter, max_keys))
    }

    pub fn add_object(&self, key: String, data: Vec<u8>) {
        self.client.add_object(&key, MockObject::from(data));
    }

    pub fn remove_object(&self, key: String) {
        self.client.remove_object(&key);
    }
}
