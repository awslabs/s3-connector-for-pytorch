/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use mountpoint_s3_client::config::{
    AddressingStyle, EndpointConfig, S3ClientAuthConfig, S3ClientConfig,
};
use mountpoint_s3_client::config::{Allocator, Uri};
use mountpoint_s3_client::types::{GetObjectParams, HeadObjectParams, PutObjectParams};
use mountpoint_s3_client::user_agent::UserAgent;
use mountpoint_s3_client::{ObjectClient, S3CrtClient};
use mountpoint_s3_crt_sys::{aws_thread_join_all_managed, aws_thread_set_managed_join_timeout_ns};
use pyo3::marker::Python;
use pyo3::types::PyTuple;
use pyo3::{pyclass, pyfunction, pymethods, Bound, PyErr, PyRef, PyResult, IntoPyObject, IntoPyObjectExt};
use std::sync::Arc;
use std::num::NonZeroUsize;

use crate::exception::python_exception;
use crate::get_object_stream::GetObjectStream;
use crate::list_object_stream::ListObjectStream;
use crate::mountpoint_s3_client_inner::{MountpointS3ClientInner, MountpointS3ClientInnerImpl};
use crate::put_object_stream::PutObjectStream;

use crate::build_info;
use crate::python_structs::py_head_object_result::PyHeadObjectResult;

#[pyclass(
    name = "MountpointS3Client",
    module = "s3torchconnectorclient._mountpoint_s3_client",
    frozen
)]
pub struct MountpointS3Client {
    pub(crate) client: Arc<dyn MountpointS3ClientInner + Send + Sync + 'static>,

    #[pyo3(get)]
    throughput_target_gbps: f64,
    #[pyo3(get)]
    region: String,
    #[pyo3(get)]
    part_size: usize,
    #[pyo3(get)]
    profile: Option<String>,
    #[pyo3(get)]
    unsigned: bool,
    #[pyo3(get)]
    force_path_style: bool,
    #[pyo3(get)]
    user_agent_prefix: String,
    #[pyo3(get)]
    endpoint: Option<String>,
    #[pyo3(get)]
    max_attempts: usize,
}

/// Waits for all managed CRT threads to complete, with a specified timeout.
///
/// This function blocks the calling thread until all CRT-managed threads have
/// completed execution or until the timeout expires.
///
/// Args:
///     timeout_secs (float): Maximum time to wait for threads to join, in seconds.
///                          Use 0.0 for no timeout.
///
/// Returns:
///     None: On successful completion when all threads have joined.
///
/// Raises:
///     RuntimeError: If threads failed to join within the timeout period.
///
/// Note:
///     This function must only be called from the main thread or a non-managed thread.
///     Calling it from a managed thread may result in deadlock or other undefined behavior.
///
/// Example:
///     >>> join_all_managed_threads(0.5)  # Wait up to 0.5 seconds for threads to join
#[pyfunction]
pub fn join_all_managed_threads(py: Python<'_>, timeout_secs: f64) -> PyResult<()> {
    unsafe {
        // Convert seconds to nanoseconds (1 second = 1_000_000_000 nanoseconds)
        let timeout_ns = (timeout_secs * 1_000_000_000.0) as u64;

        aws_thread_set_managed_join_timeout_ns(timeout_ns);

        // Release the GIL while waiting for other threads to join, which may acquire GIL, to avoid deadlock
        let result = py.allow_threads(|| aws_thread_join_all_managed());

        if result != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to join managed threads in {} secs",
                timeout_secs
            )));
        }
    }
    Ok(())
}

#[pymethods]
impl MountpointS3Client {
    #[new]
    #[pyo3(signature = (region, user_agent_prefix="".to_string(), throughput_target_gbps=10.0, part_size=8*1024*1024, profile=None, unsigned=false, endpoint=None, force_path_style=false, max_attempts=10))]
    #[allow(clippy::too_many_arguments)]
    pub fn new_s3_client(
        region: String,
        user_agent_prefix: String,
        throughput_target_gbps: f64,
        part_size: usize,
        profile: Option<String>,
        unsigned: bool,
        endpoint: Option<String>,
        force_path_style: bool,
        max_attempts: usize,
    ) -> PyResult<Self> {
        // TODO: Mountpoint has logic for guessing based on instance type. It may be worth having
        // similar logic if we want to exceed 10Gbps reading for larger instances

        let endpoint_str = endpoint.as_deref().unwrap_or("");
        let mut endpoint_config = if endpoint_str.is_empty() {
            EndpointConfig::new(&region)
        } else {
            EndpointConfig::new(&region)
                .endpoint(Uri::new_from_str(&Allocator::default(), endpoint_str).unwrap())
        };
        if force_path_style {
            endpoint_config = endpoint_config.addressing_style(AddressingStyle::Path);
        }
        let auth_config = auth_config(profile.as_deref(), unsigned);

        let user_agent_suffix =
            &format!("{}/{}", build_info::PACKAGE_NAME, build_info::FULL_VERSION);
        let mut user_agent_string = &format!("{} {}", &user_agent_prefix, &user_agent_suffix);
        if user_agent_prefix.ends_with(user_agent_suffix) {
            // If we unpickle a client, we should not append the suffix again
            user_agent_string = &user_agent_prefix;
        }

        let config = S3ClientConfig::new()
            .user_agent(UserAgent::new(Some(user_agent_string.to_owned())))
            .throughput_target_gbps(throughput_target_gbps)
            .part_size(part_size)
            .auth_config(auth_config)
            .endpoint_config(endpoint_config)
            .max_attempts(NonZeroUsize::try_from(max_attempts).expect("max_attempts must be > 0"));
        let crt_client = Arc::new(S3CrtClient::new(config).map_err(python_exception)?);

        Ok(MountpointS3Client::new(
            region,
            user_agent_prefix.to_string(),
            throughput_target_gbps,
            part_size,
            profile,
            unsigned,
            force_path_style,
            max_attempts,
            crt_client,
            endpoint,
        ))
    }

    #[pyo3(signature = (bucket, key, start=None, end=None))]
    pub fn get_object(
        slf: PyRef<'_, Self>,
        bucket: String,
        key: String,
        start: Option<u64>,
        end: Option<u64>,
    ) -> PyResult<GetObjectStream> {
        let mut params = GetObjectParams::default();

        // Configure byte range for the request if start is provided
        if let Some(start_val) = start {
            // Use end_val if provided, otherwise use u64::MAX
            let end_val = end.unwrap_or(u64::MAX);
            params = params.range(Some(start_val..end_val));
        }

        slf.client.get_object(slf.py(), bucket, key, params)
    }

    #[pyo3(signature = (bucket, prefix=String::from(""), delimiter=String::from(""), max_keys=1000))]
    pub fn list_objects(
        &self,
        bucket: String,
        prefix: String,
        delimiter: String,
        max_keys: usize,
    ) -> ListObjectStream {
        ListObjectStream::new(self.client.clone(), bucket, prefix, delimiter, max_keys)
    }

    #[pyo3(signature = (bucket, key, storage_class=None))]
    pub fn put_object(
        slf: PyRef<'_, Self>,
        bucket: String,
        key: String,
        storage_class: Option<String>,
    ) -> PyResult<PutObjectStream> {
        let mut params = PutObjectParams::default();
        params.storage_class = storage_class;

        slf.client.put_object(slf.py(), bucket, key, params)
    }

    pub fn head_object(
        slf: PyRef<'_, Self>,
        bucket: String,
        key: String,
    ) -> PyResult<PyHeadObjectResult> {
        let params = HeadObjectParams::default();
        slf.client.head_object(slf.py(), bucket, key, params)
    }

    pub fn delete_object(slf: PyRef<'_, Self>, bucket: String, key: String) -> PyResult<()> {
        slf.client.delete_object(slf.py(), bucket, key)
    }

    pub fn copy_object(
        slf: PyRef<'_, Self>,
        src_bucket: String,
        src_key: String,
        dst_bucket: String,
        dst_key: String,
    ) -> PyResult<()> {
        slf.client
            .copy_object(slf.py(), src_bucket, src_key, dst_bucket, dst_key)
    }

    pub fn __getnewargs__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyTuple>> {
        let py = slf.py();
        let state = [
            slf.region.clone().into_pyobject(py)?.into_any(),
            slf.user_agent_prefix.clone().into_pyobject(py)?.into_any(),
            slf.throughput_target_gbps.into_pyobject(py)?.into_any(),
            slf.part_size.into_pyobject(py)?.into_any(),
            slf.profile.clone().into_pyobject(py)?.into_any(),
            slf.unsigned.into_py_any(py)?.bind(py).to_owned(),
            slf.endpoint.clone().into_pyobject(py)?.into_any(),
            slf.force_path_style.into_py_any(py)?.bind(py).to_owned(),
            slf.max_attempts.into_pyobject(py)?.into_any(),
        ];
        PyTuple::new(py, state)
    }
}

#[allow(clippy::too_many_arguments)]
impl MountpointS3Client {
    pub(crate) fn new<Client>(
        region: String,
        user_agent_prefix: String,
        throughput_target_gbps: f64,
        part_size: usize,
        profile: Option<String>,
        // no_sign_request on mountpoint-s3-client
        unsigned: bool,
        force_path_style: bool,
        max_attempts: usize,
        client: Arc<Client>,
        endpoint: Option<String>,
    ) -> Self
    where
        Client: ObjectClient + Sync + Send + 'static,
        <Client as ObjectClient>::GetObjectResponse: Unpin + Sync,
        <Client as ObjectClient>::PutObjectRequest: Sync,
    {
        Self {
            throughput_target_gbps,
            part_size,
            region,
            profile,
            unsigned,
            force_path_style,
            max_attempts,
            client: Arc::new(MountpointS3ClientInnerImpl::new(client)),
            user_agent_prefix,
            endpoint,
        }
    }
}

fn auth_config(profile: Option<&str>, unsigned: bool) -> S3ClientAuthConfig {
    if unsigned {
        S3ClientAuthConfig::NoSigning
    } else if let Some(profile_name) = profile {
        S3ClientAuthConfig::Profile(profile_name.to_string())
    } else {
        S3ClientAuthConfig::Default
    }
}
