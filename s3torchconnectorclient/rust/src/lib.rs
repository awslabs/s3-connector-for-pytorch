/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use crate::exception::S3Exception;
use crate::get_object_stream::GetObjectStream;
use crate::list_object_stream::ListObjectStream;
use crate::mock_client::PyMockClient;
use crate::mountpoint_s3_client::join_all_managed_threads;
use crate::mountpoint_s3_client::MountpointS3Client;
use crate::put_object_stream::PutObjectStream;
use crate::python_structs::py_head_object_result::PyHeadObjectResult;
use crate::python_structs::py_list_object_result::PyListObjectResult;
use crate::python_structs::py_object_info::PyObjectInfo;
use crate::python_structs::py_restore_status::PyRestoreStatus;
use pyo3::prelude::*;

mod build_info;
mod exception;
mod get_object_stream;
mod list_object_stream;
mod logger_setup;
mod mock_client;
mod mountpoint_s3_client;
mod mountpoint_s3_client_inner;
mod put_object_stream;
mod python_structs;

#[pymodule]
#[pyo3(name = "_mountpoint_s3_client")]
fn make_lib(py: Python, mountpoint_s3_client: &Bound<'_, PyModule>) -> PyResult<()> {
    logger_setup::setup_logging()?;
    mountpoint_s3_client.add_class::<MountpointS3Client>()?;
    mountpoint_s3_client.add_class::<PyMockClient>()?;
    mountpoint_s3_client.add_class::<GetObjectStream>()?;
    mountpoint_s3_client.add_class::<ListObjectStream>()?;
    mountpoint_s3_client.add_class::<PutObjectStream>()?;
    mountpoint_s3_client.add_class::<PyListObjectResult>()?;
    mountpoint_s3_client.add_class::<PyObjectInfo>()?;
    mountpoint_s3_client.add_class::<PyHeadObjectResult>()?;
    mountpoint_s3_client.add_class::<PyRestoreStatus>()?;
    mountpoint_s3_client.add("S3Exception", py.get_type::<S3Exception>())?;
    mountpoint_s3_client.add("__version__", build_info::FULL_VERSION)?;
    mountpoint_s3_client.add_function(wrap_pyfunction!(
        join_all_managed_threads,
        mountpoint_s3_client
    )?)?;

    Ok(())
}
