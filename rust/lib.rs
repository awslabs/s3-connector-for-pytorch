use log::LevelFilter;
use pyo3::prelude::*;
use pyo3_log::Logger;

use crate::exception::{python_exception, S3DatasetException};
use crate::mountpoint_s3_client::MountpointS3Client;

mod mountpoint_s3_client;
mod exception;
mod get_object_stream;

#[pymodule]
#[pyo3(name = "_s3dataset")]
fn make_lib(py: Python, s3dataset: &PyModule) -> PyResult<()> {
    let logger = Logger::default().filter(LevelFilter::Trace);
    logger.install().map_err(python_exception)?;

    s3dataset.add_class::<MountpointS3Client>()?;
    s3dataset.add("S3DatasetException", py.get_type::<S3DatasetException>())?;
    Ok(())
}
