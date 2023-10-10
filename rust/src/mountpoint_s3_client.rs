use std::sync::Arc;

use futures::executor::block_on;
use futures::TryStreamExt;
use mountpoint_s3_client::{ObjectClient, S3CrtClient};
use mountpoint_s3_client::config::{EndpointConfig, S3ClientAuthConfig, S3ClientConfig};
use pyo3::{pyclass, pymethods, PyRef, PyResult, Python};

use crate::exception::python_exception;
use crate::get_object_stream::GetObjectStream;
use crate::list_object_stream::ListObjectStream;

#[pyclass(name = "MountpointS3Client", module="_s3dataset", frozen)]
#[derive(Debug)]
pub struct MountpointS3Client {
    client: Arc<S3CrtClient>,
    #[pyo3(get)]
    throughput_target_gbps: f64,
    #[pyo3(get)]
    region: String,
    #[pyo3(get)]
    part_size: usize,
}

#[pymethods]
impl MountpointS3Client {
    #[new]
    #[pyo3(signature = (region, throughput_target_gbps=10.0, part_size=8*1024*1024, profile=None, no_sign_request=false))]
    pub fn new(region: String, throughput_target_gbps: f64, part_size: usize, profile: Option<String>, no_sign_request: bool) -> PyResult<Self> {
        /*
        TODO - Mountpoint has logic for guessing based on instance type.
         It may be worth having similar logic if we want to exceed 10Gbps reading for larger instances
        */

        let endpoint_config = EndpointConfig::new(&region);
        let auth_config = auth_config(profile, no_sign_request);

        let config = S3ClientConfig::new()
            /*
             TODO - Add version number here
              https://github.com/awslabs/mountpoint-s3/blob/73328cc64a2dbca78e879730d4d264aedd881c60/mountpoint-s3/src/main.rs#L427
             */
            .user_agent_prefix("pytorch-loader;mountpoint")
            .throughput_target_gbps(throughput_target_gbps)
            .part_size(part_size)
            .auth_config(auth_config)
            .endpoint_config(endpoint_config);
        let client = Arc::new(S3CrtClient::new(config).map_err(python_exception)?);
        Ok(Self { client, throughput_target_gbps, part_size, region })
    }

    pub fn get_object(slf: PyRef<'_, Self>, bucket: String, key: String) -> PyResult<GetObjectStream> {
        let request = slf.client.get_object(&bucket, &key, None, None);

        // TODO - Look at use of `block_on` and see if we can future this.
        let mut request = slf.py().allow_threads(|| {block_on(request).map_err(python_exception)})?;
        let next_part = move |py: Python| {
            py.allow_threads(|| {
                block_on(request.try_next()).map_err(python_exception)
            })
        };

        Ok(GetObjectStream::new(Box::new(next_part), bucket, key))
    }

    #[pyo3(signature = (bucket, prefix=String::from(""), delimiter=String::from(""), max_keys=1000))]
    pub fn list_objects(&self, bucket: String, prefix: String, delimiter: String, max_keys: usize) -> PyResult<ListObjectStream> {
        Ok(ListObjectStream::new(self.client.clone(), bucket, prefix, delimiter, max_keys))
    }
}

fn auth_config(profile: Option<String>, no_sign_request: bool) -> S3ClientAuthConfig {
    if no_sign_request {
        S3ClientAuthConfig::NoSigning
    } else if let Some(profile_name) = profile {
        S3ClientAuthConfig::Profile(profile_name)
    } else {
        S3ClientAuthConfig::Default
    }
}
