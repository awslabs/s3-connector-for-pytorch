/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use mountpoint_s3_client::types::ObjectInfo;
use pyo3::types::PyTuple;
use pyo3::{pyclass, pymethods};
use pyo3::{Bound};
use pyo3::{IntoPyObject, PyResult};

use crate::python_structs::py_restore_status::PyRestoreStatus;
use crate::PyRef;

#[pyclass(
    name = "ObjectInfo",
    module = "s3torchconnectorclient._mountpoint_s3_client",
    frozen
)]
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
    restore_status: Option<PyRestoreStatus>,
}

impl PyObjectInfo {
    pub(crate) fn from_object_info(object_info: ObjectInfo) -> Self {
        PyObjectInfo::new(
            object_info.key,
            object_info.etag,
            object_info.size,
            object_info.last_modified.unix_timestamp(),
            object_info.storage_class,
            object_info
                .restore_status
                .map(PyRestoreStatus::from_restore_status),
        )
    }
}

#[pymethods]
impl PyObjectInfo {
    #[new]
    #[pyo3(signature = (key, etag, size, last_modified, storage_class=None, restore_status=None))]
    pub fn new(
        key: String,
        etag: String,
        size: u64,
        last_modified: i64,
        storage_class: Option<String>,
        restore_status: Option<PyRestoreStatus>,
    ) -> Self {
        Self {
            key,
            etag,
            size,
            last_modified,
            storage_class,
            restore_status,
        }
    }

    pub fn __getnewargs__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyTuple>> {
        let py = slf.py();
        let state = [
            slf.key.clone().into_pyobject(py)?.into_any(),
            slf.etag.clone().into_pyobject(py)?.into_any(),
            slf.size.into_pyobject(py)?.into_any(),
            slf.last_modified.into_pyobject(py)?.into_any(),
            slf.storage_class.clone().into_pyobject(py)?.into_any(),
            slf.restore_status.clone().into_pyobject(py)?.into_any(),
        ];
        PyTuple::new(py, state)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}
