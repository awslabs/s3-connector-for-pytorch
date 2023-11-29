/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use std::sync::Arc;

use mountpoint_s3_client::mock_client::{MockClient, MockClientConfig, MockObject};
use pyo3::{pyclass, pymethods};

use crate::MountpointS3Client;

#[derive(Clone)]
#[pyclass(
    name = "MockMountpointS3Client",
    module = "s3torchconnectorclient._mountpoint_s3_client",
    frozen
)]
pub struct PyMockClient {
    mock_client: Arc<MockClient>,
    #[pyo3(get)]
    pub(crate) throughput_target_gbps: f64,
    #[pyo3(get)]
    pub(crate) region: String,
    #[pyo3(get)]
    pub(crate) part_size: usize,
}

#[pymethods]
impl PyMockClient {
    #[new]
    #[pyo3(signature = (region, bucket, throughput_target_gbps = 10.0, part_size = 8 * 1024 * 1024))]
    pub fn new(
        region: String,
        bucket: String,
        throughput_target_gbps: f64,
        part_size: usize,
    ) -> PyMockClient {
        let unordered_list_seed: Option<u64> = None;
        let config = MockClientConfig { bucket, part_size, unordered_list_seed };
        let mock_client = Arc::new(MockClient::new(config));

        PyMockClient {
            mock_client,
            region,
            throughput_target_gbps,
            part_size,
        }
    }

    fn create_mocked_client(&self) -> MountpointS3Client {
        MountpointS3Client::new(
            self.region.clone(),
            "mock-client".to_string(),
            self.throughput_target_gbps,
            self.part_size,
            None,
            false,
            self.mock_client.clone(),
        )
    }

    fn add_object(&self, key: String, data: Vec<u8>) {
        self.mock_client.add_object(&key, MockObject::from(data));
    }

    fn remove_object(&self, key: String) {
        self.mock_client.remove_object(&key);
    }
}
