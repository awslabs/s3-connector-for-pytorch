/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use futures::executor::block_on;
use mountpoint_s3_client::PutObjectRequest;
use pyo3::{pyclass, pymethods, PyRefMut, PyResult, Python};

use crate::exception::{python_exception, S3Exception};

#[pyclass(
    name = "PutObjectStream",
    module = "s3torchconnectorclient._mountpoint_s3_client"
)]
pub struct PutObjectStream {
    request: Box<dyn PutObjectRequestWrapper + Send + Sync>,
    #[pyo3(get)]
    bucket: String,
    #[pyo3(get)]
    key: String,
}

impl PutObjectStream {
    pub(crate) fn new<T: PutObjectRequest + Sync + Sync + 'static>(
        request: T,
        bucket: String,
        key: String,
    ) -> Self {
        let request = Box::new(PutObjectRequestWrapperImpl::new(request));
        Self {
            request,
            bucket,
            key,
        }
    }
}

#[pymethods]
impl PutObjectStream {
    pub fn write(mut slf: PyRefMut<'_, Self>, data: &[u8]) -> PyResult<()> {
        let py = slf.py();
        slf.request.write(py, data)
    }

    pub fn close(mut slf: PyRefMut<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        slf.request.complete(py)
    }
}

pub trait PutObjectRequestWrapper {
    fn write(&mut self, py: Python, data: &[u8]) -> PyResult<()>;
    fn complete(&mut self, py: Python) -> PyResult<()>;
}

pub struct PutObjectRequestWrapperImpl<T: PutObjectRequest> {
    request: Option<T>,
}

impl<T: PutObjectRequest> PutObjectRequestWrapperImpl<T> {
    pub fn new(request: T) -> PutObjectRequestWrapperImpl<T> {
        PutObjectRequestWrapperImpl {
            request: Some(request),
        }
    }
}

impl<T: PutObjectRequest + Send + Sync> PutObjectRequestWrapper for PutObjectRequestWrapperImpl<T> {
    fn write(&mut self, py: Python, data: &[u8]) -> PyResult<()> {
        if let Some(request) = self.request.as_mut() {
            py.allow_threads(|| block_on(request.write(data)).map_err(python_exception))
        } else {
            Err(S3Exception::new_err("Cannot write to closed object"))
        }
    }

    fn complete(&mut self, py: Python) -> PyResult<()> {
        if let Some(request) = self.request.take() {
            py.allow_threads(|| block_on(request.complete()).map_err(python_exception))?;
            Ok(())
        } else {
            Err(S3Exception::new_err("Cannot close object more than once"))
        }
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
    fn test_put_object() -> PyResult<()> {
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
                data_to_write = b"Hello!"
                
                mock_client = MockMountpointS3Client("us-east-1", "mock-bucket")
                client = mock_client.create_mocked_client()
                
                put_stream = client.put_object("mock-bucket", "key")
                put_stream.write(data_to_write)
                put_stream.close()
                
                get_stream = client.get_object("mock-bucket", "key")
                assert b''.join(get_stream) == data_to_write
                "#
            );
        });

        Ok(())
    }
}
