use mountpoint_s3_client::types::ObjectInfo;
use pyo3::{pyclass, pymethods};

use crate::python_structs::py_restore_status::PyRestoreStatus;

#[pyclass(name = "ObjectInfo", module="_s3dataset")]
#[derive(Debug, Clone)]
pub struct PyObjectInfo {
    #[pyo3(get)]
    key: String,
    #[pyo3(get)]
    etag: String,
    #[pyo3(get)]
    size: u64,
    #[pyo3(get)]
    last_modified: i64,
    #[pyo3(get)]
    storage_class: Option<String>,
    #[pyo3(get)]
    restore_status: Option<PyRestoreStatus>
}

impl PyObjectInfo {
    pub(crate) fn new(object_info: ObjectInfo) -> Self {
        Self {
            key: object_info.key,
            etag: object_info.etag,
            size: object_info.size,
            last_modified: object_info.last_modified.unix_timestamp(),
            storage_class: object_info.storage_class,
            restore_status: object_info.restore_status.map(PyRestoreStatus::new),
        }
    }
}

#[pymethods]
impl PyObjectInfo {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

