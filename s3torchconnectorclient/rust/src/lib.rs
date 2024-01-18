/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use log::LevelFilter;
use mountpoint_s3_crt::common::rust_log_adapter::RustLogAdapter;
use pyo3::prelude::*;
use pyo3_log::Logger;

use crate::exception::{python_exception, S3Exception};
use crate::get_object_stream::GetObjectStream;
use crate::list_object_stream::ListObjectStream;
use crate::mock_client::PyMockClient;
use crate::mountpoint_s3_client::MountpointS3Client;
use crate::put_object_stream::PutObjectStream;
use crate::python_structs::py_list_object_result::PyListObjectResult;
use crate::python_structs::py_object_info::PyObjectInfo;
use crate::python_structs::py_restore_status::PyRestoreStatus;

mod exception;
mod get_object_stream;
mod list_object_stream;
mod mock_client;
mod mountpoint_s3_client;
mod mountpoint_s3_client_inner;
mod put_object_stream;
mod python_structs;
mod build_info;


/// Experimental method to enable CRT logs for debugging purposes.
/// Please do NOT call this method unless otherwise instructed.
#[pyfunction]
#[pyo3(name = "_enable_crt_logging")]
fn _enable_crt_logging() {
    let _ = RustLogAdapter::try_init().map_err(python_exception);
}


/// Experimental method to prevent deadlocks when enabling verbose CRT logging:
/// https://docs.rs/pyo3-log/latest/pyo3_log/#interaction-with-python-gil
#[pyfunction]
#[pyo3(name = "_disable_logging")]
fn _disable_logging() {
    Logger::default().reset_handle();
}

#[pymodule]
#[pyo3(name = "_mountpoint_s3_client")]
fn make_lib(py: Python, mountpoint_s3_client: &PyModule) -> PyResult<()> {
    // Filter out events for request spans, which are at warn level
    let logger = Logger::default()
        .filter_target(
            "mountpoint_s3_client::s3_crt_client::request".to_owned(),
            LevelFilter::Off,
        )
        .filter(LevelFilter::Trace);
    logger.install().map_err(python_exception)?;

    mountpoint_s3_client.add_class::<MountpointS3Client>()?;
    mountpoint_s3_client.add_class::<PyMockClient>()?;
    mountpoint_s3_client.add_class::<GetObjectStream>()?;
    mountpoint_s3_client.add_class::<ListObjectStream>()?;
    mountpoint_s3_client.add_class::<PutObjectStream>()?;
    mountpoint_s3_client.add_class::<PyListObjectResult>()?;
    mountpoint_s3_client.add_class::<PyObjectInfo>()?;
    mountpoint_s3_client.add_class::<PyRestoreStatus>()?;
    mountpoint_s3_client.add("S3Exception", py.get_type::<S3Exception>())?;
    mountpoint_s3_client.add("__version__", build_info::FULL_VERSION)?;
    mountpoint_s3_client.add_function(wrap_pyfunction!(_enable_crt_logging, mountpoint_s3_client)?)?;
    mountpoint_s3_client.add_function(wrap_pyfunction!(_disable_logging, mountpoint_s3_client)?)?;
    Ok(())
}
