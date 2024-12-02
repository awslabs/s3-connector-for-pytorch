/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use pyo3::{pyclass, pymethods};

use crate::python_structs::py_object_info::PyObjectInfo;

#[pyclass(
    name = "ListObjectResult",
    module = "s3torchconnectorclient._mountpoint_s3_client"
)]
#[derive(Debug)]
pub struct PyListObjectResult {
    #[pyo3(get)]
    object_info: Vec<PyObjectInfo>,
    #[pyo3(get)]
    common_prefixes: Vec<String>,
}

impl PyListObjectResult {
    pub(crate) fn new(object_info: Vec<PyObjectInfo>, common_prefixes: Vec<String>) -> Self {
        Self {
            object_info,
            common_prefixes,
        }
    }
}

#[pymethods]
impl PyListObjectResult {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}
