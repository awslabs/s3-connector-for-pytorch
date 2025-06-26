/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Bound, PyRef, PyRefMut, PyResult};
use mountpoint_s3_client::types::GetBodyPart;

use crate::exception::S3Exception;
use crate::mountpoint_s3_client_inner::MPGetObjectClosure;

#[pyclass(
    name = "GetObjectStream",
    module = "s3torchconnectorclient._mountpoint_s3_client"
)]
pub struct GetObjectStream {
    next_part: MPGetObjectClosure,
    offset: u64,
    #[pyo3(get)]
    bucket: String,
    #[pyo3(get)]
    key: String,
}

impl GetObjectStream {
    pub(crate) fn new(next_part: MPGetObjectClosure, bucket: String, key: String, start_offset: Option<u64>) -> Self {
        Self {
            next_part,
            offset: start_offset.unwrap_or(0),
            bucket,
            key,
        }
    }
}

#[pymethods]
impl GetObjectStream {
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyBytes>>> {
        let py = slf.py();

        let body_part = (slf.next_part)(py)?;
        match body_part {
            None => Ok(None),
            Some(GetBodyPart { offset, data }) => {
                if offset != slf.offset {
                    return Err(S3Exception::new_err(
                        "Data from S3 was returned out of order!",
                    ));
                }
                slf.offset += data.len() as u64;
                let data = PyBytes::new(py, data.as_ref());
                Ok(Some(data))
            }
        }
    }

    pub fn tell(slf: PyRef<'_, Self>) -> u64 {
        slf.offset
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
    fn test_get_object() -> PyResult<()> {
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
                mock_client = MockMountpointS3Client("us-east-1", "mock-bucket")
                client = mock_client.create_mocked_client()

                mock_client.add_object("key", b"data")
                stream = client.get_object("mock-bucket", "key")

                returned_data = b''.join(stream)
                assert returned_data == b"data"
                "#
            );
        });

        Ok(())
    }
}
