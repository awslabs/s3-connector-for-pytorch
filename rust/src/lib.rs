use log::LevelFilter;
use pyo3::prelude::*;
use pyo3_log::Logger;

use crate::exception::{python_exception, S3DatasetException};
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

#[pymodule]
#[pyo3(name = "_s3dataset")]
fn make_lib(py: Python, s3dataset: &PyModule) -> PyResult<()> {
    let logger = Logger::default().filter(LevelFilter::Trace);
    logger.install().map_err(python_exception)?;

    s3dataset.add_class::<MountpointS3Client>()?;
    s3dataset.add_class::<PyMockClient>()?;
    s3dataset.add_class::<GetObjectStream>()?;
    s3dataset.add_class::<ListObjectStream>()?;
    s3dataset.add_class::<PutObjectStream>()?;
    s3dataset.add_class::<PyListObjectResult>()?;
    s3dataset.add_class::<PyObjectInfo>()?;
    s3dataset.add_class::<PyRestoreStatus>()?;
    s3dataset.add("S3DatasetException", py.get_type::<S3DatasetException>())?;
    Ok(())
}
