/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use std::sync::Arc;

use mountpoint_s3_client::types::ListObjectsResult;
use pyo3::{pyclass, pymethods, PyRef, PyRefMut, PyResult, Python};

use crate::mountpoint_s3_client::MountpointS3Client;
use crate::mountpoint_s3_client_inner::MountpointS3ClientInner;
use crate::python_structs::py_list_object_result::PyListObjectResult;
use crate::python_structs::py_object_info::PyObjectInfo;

#[pyclass(
    name = "ListObjectStream",
    module = "s3torchconnectorclient._mountpoint_s3_client"
)]
pub struct ListObjectStream {
    client: Arc<dyn MountpointS3ClientInner + Send + Sync + 'static>,
    #[pyo3(get)]
    continuation_token: Option<String>,
    #[pyo3(get)]
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
    pub(crate) fn new(
        client: Arc<dyn MountpointS3ClientInner + Send + Sync + 'static>,
        bucket: String,
        prefix: String,
        delimiter: String,
        max_keys: usize,
    ) -> Self {
        Self {
            client,
            bucket,
            prefix,
            delimiter,
            max_keys,
            continuation_token: None,
            complete: false,
        }
    }

    fn make_request(&self, py: Python) -> PyResult<ListObjectsResult> {
        py.allow_threads(|| {
            let client = &self.client;
            client.list_objects(
                &self.bucket,
                self.continuation_token.as_deref(),
                &self.delimiter,
                self.max_keys,
                &self.prefix,
            )
        })
    }
}

#[pymethods]
impl ListObjectStream {
    #[staticmethod]
    #[pyo3(signature=(client, bucket, prefix, delimiter, max_keys, continuation_token, complete))]
    pub fn _from_state(
        client: &MountpointS3Client,
        bucket: String,
        prefix: String,
        delimiter: String,
        max_keys: usize,
        continuation_token: Option<String>,
        complete: bool,
    ) -> Self {
        Self {
            client: client.client.clone(),
            bucket,
            prefix,
            delimiter,
            max_keys,
            continuation_token,
            complete,
        }
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyListObjectResult>> {
        if slf.complete {
            return Ok(None);
        }
        let results = slf.make_request(slf.py())?;

        slf.continuation_token = results.next_continuation_token;
        if slf.continuation_token.is_none() {
            slf.complete = true;
        }

        let objects = results
            .objects
            .into_iter()
            .map(PyObjectInfo::from_object_info)
            .collect();

        Ok(Some(PyListObjectResult::new(
            objects,
            results.common_prefixes,
        )))
    }
}

#[cfg(test)]
mod tests {
    use pyo3::types::IntoPyDict;
    use pyo3::{py_run, PyResult, Python};
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    use crate::mock_client::PyMockClient;
    use crate::mountpoint_s3_client::MountpointS3Client;

    #[test]
    fn test_list_objects() -> PyResult<()> {
        let layer = tracing_subscriber::fmt::layer().with_ansi(true);
        let registry = tracing_subscriber::registry().with(layer);
        let _ = registry.try_init();

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let locals = [
                (
                    "MountpointS3Client",
                    py.get_type::<MountpointS3Client>(),
                ),
                (
                    "MockMountpointS3Client",
                    py.get_type::<PyMockClient>(),
                ),
            ];

            py_run!(
                py,
                *locals.into_py_dict(py).unwrap(),
                r#"
                expected_keys = {"test"}
                
                mock_client = MockMountpointS3Client("us-east-1", "mock-bucket")
                client = mock_client.create_mocked_client()
                for key in expected_keys:
                    mock_client.add_object(key, b"")
                
                stream = client.list_objects("mock-bucket")
                
                object_infos = [object_info for page in stream for object_info in page.object_info]
                keys = {object_info.key for object_info in object_infos}
                assert keys == expected_keys
                "#
            );
        });

        Ok(())
    }
}
