/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use mountpoint_s3_client::types::RestoreStatus;
use pyo3::types::PyTuple;
use pyo3::PyResult;
use pyo3::ToPyObject;
use pyo3::{pyclass, pymethods};

use crate::PyRef;

#[pyclass(name = "RestoreStatus", module = "s3torchconnectorclient._mountpoint_s3_client")]
#[derive(Debug, Clone)]
pub struct PyRestoreStatus {
    #[pyo3(get)]
    in_progress: bool,
    #[pyo3(get)]
    expiry: Option<u128>,
}

impl PyRestoreStatus {
    pub(crate) fn from_restore_status(restore_status: RestoreStatus) -> Self {
        match restore_status {
            RestoreStatus::InProgress => PyRestoreStatus::new(true, None),
            RestoreStatus::Restored { expiry } => {
                let expiry = expiry
                    .duration_since(expiry)
                    .expect("Expired before unix epoch!")
                    .as_millis();
                PyRestoreStatus::new(false, Some(expiry))
            }
        }
    }
}

#[pymethods]
impl PyRestoreStatus {
    #[new]
    pub fn new(in_progress: bool, expiry: Option<u128>) -> Self {
        Self {
            in_progress,
            expiry,
        }
    }

    pub fn __getnewargs__(slf: PyRef<'_, Self>) -> PyResult<&PyTuple> {
        let py = slf.py();
        let state = [slf.in_progress.to_object(py), slf.expiry.to_object(py)];
        Ok(PyTuple::new(py, state))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}
