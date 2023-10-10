use mountpoint_s3_client::types::RestoreStatus;
use pyo3::{pyclass, pymethods};

#[pyclass(name = "RestoreStatus", module="_s3dataset")]
#[derive(Debug, Clone)]
pub struct PyRestoreStatus {
    #[pyo3(get)]
    in_progress: bool,
    #[pyo3(get)]
    expiry: Option<u128>
}

impl PyRestoreStatus {
    pub(crate) fn new(restore_status: RestoreStatus) -> Self {
        match restore_status {
            RestoreStatus::InProgress => PyRestoreStatus { in_progress: true, expiry: None },
            RestoreStatus::Restored { expiry } => {
                let expiry = expiry.duration_since(expiry).expect("Expired before unix epoch!").as_millis();
                PyRestoreStatus { in_progress: false, expiry: Some(expiry) }
            }
        }
    }
}

#[pymethods]
impl PyRestoreStatus {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

