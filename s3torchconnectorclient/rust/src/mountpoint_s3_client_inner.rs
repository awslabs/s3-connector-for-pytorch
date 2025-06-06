/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use std::sync::Arc;

use futures::executor::block_on;
use futures::TryStreamExt;
use mountpoint_s3_client::types::{GetBodyPart, CopyObjectParams, GetObjectParams, HeadObjectParams, ListObjectsResult, PutObjectParams};
use mountpoint_s3_client::ObjectClient;
use pyo3::{PyResult, Python};

use crate::exception::python_exception;
use crate::get_object_stream::GetObjectStream;
use crate::put_object_stream::PutObjectStream;
use crate::python_structs::py_head_object_result::PyHeadObjectResult;

pub type MPGetObjectClosure =
    Box<dyn FnMut(Python) -> PyResult<Option<GetBodyPart>> + Send + Sync>;

/// We need an extra trait here, as pyo3 doesn't support structs with generics.
/// However, it does allow dynamic traits. We can't ues `ObjectClient` itself, as that requires
/// us to specify the additional types (GetObjectResult, PutObjectResult, ClientError), which
/// we don't want to do.
pub(crate) trait MountpointS3ClientInner {
    fn get_object(&self, py: Python, bucket: String, key: String, params: GetObjectParams) -> PyResult<GetObjectStream>;
    fn list_objects(
        &self,
        bucket: &str,
        continuation_token: Option<&str>,
        delimiter: &str,
        max_keys: usize,
        prefix: &str,
    ) -> PyResult<ListObjectsResult>;
    fn put_object(
        &self,
        py: Python,
        bucket: String,
        key: String,
        params: PutObjectParams,
    ) -> PyResult<PutObjectStream>;
    fn head_object(&self, py: Python, bucket: String, key: String, params: HeadObjectParams) -> PyResult<PyHeadObjectResult>;
    fn delete_object(&self, py: Python, bucket: String, key: String) -> PyResult<()>;
    fn copy_object(
        &self,
        py: Python,
        source_bucket: String,
        source_key: String,
        destination_bucket: String,
        destination_key: String,
    ) -> PyResult<()>;
}

pub(crate) struct MountpointS3ClientInnerImpl<T: ObjectClient> {
    client: Arc<T>,
}

impl<T: ObjectClient> MountpointS3ClientInnerImpl<T> {
    pub fn new(client: Arc<T>) -> MountpointS3ClientInnerImpl<T> {
        MountpointS3ClientInnerImpl { client }
    }
}

impl<Client> MountpointS3ClientInner for MountpointS3ClientInnerImpl<Client>
where
    Client: ObjectClient,
    <Client as ObjectClient>::GetObjectResponse: Sync + Unpin + 'static,
    <Client as ObjectClient>::PutObjectRequest: Sync + 'static,
{
    fn get_object(&self, py: Python, bucket: String, key: String, params: GetObjectParams) -> PyResult<GetObjectStream> {
        let request = self.client.get_object(&bucket, &key, &params);

        // TODO - Look at use of `block_on` and see if we can future this.
        let mut request = py.allow_threads(|| block_on(request).map_err(python_exception))?;

        let closure = Box::new(move |py: Python| {
            py.allow_threads(|| block_on(request.try_next()).map_err(python_exception))
        });

        // Extract start_offset from params if needed for RangedS3Reader
        let start_offset = params.range.as_ref().map(|range| range.start);

        Ok(GetObjectStream::new(closure, bucket, key, start_offset))
    }

    fn list_objects(
        &self,
        bucket: &str,
        continuation_token: Option<&str>,
        delimiter: &str,
        max_keys: usize,
        prefix: &str,
    ) -> PyResult<ListObjectsResult> {
        block_on(
            self.client
                .list_objects(bucket, continuation_token, delimiter, max_keys, prefix),
        )
        .map_err(python_exception)
    }

    fn put_object(
        &self,
        py: Python,
        bucket: String,
        key: String,
        params: PutObjectParams,
    ) -> PyResult<PutObjectStream> {
        let request = self.client.put_object(&bucket, &key, &params);
        // TODO - Look at use of `block_on` and see if we can future this.
        let request = py.allow_threads(|| block_on(request).map_err(python_exception))?;

        Ok(PutObjectStream::new(request, bucket, key))
    }

    fn head_object(&self, py: Python, bucket: String, key: String, params: HeadObjectParams) -> PyResult<PyHeadObjectResult> {
        let request = self.client.head_object(&bucket, &key, &params);

        // TODO - Look at use of `block_on` and see if we can future this.
        let request = py.allow_threads(|| block_on(request).map_err(python_exception))?;
        Ok(PyHeadObjectResult::from_head_object_result(request))
    }

    fn delete_object(&self, py: Python, bucket: String, key: String) -> PyResult<()> {
        let request = self.client.delete_object(&bucket, &key);

        // TODO - Look at use of `block_on` and see if we can future this.
        py.allow_threads(|| block_on(request).map_err(python_exception))?;
        Ok(())
    }

    fn copy_object(
        &self,
        py: Python,
        source_bucket: String,
        source_key: String,
        destination_bucket: String,
        destination_key: String,
    ) -> PyResult<()> {
        // [Oct. 2024] `CopyObjectParams` is omitted from the function signature, as it anyway contains no fields.
        let params = CopyObjectParams::new();
        let request = self.client.copy_object(
            &source_bucket,
            &source_key,
            &destination_bucket,
            &destination_key,
            &params,
        );

        py.allow_threads(|| block_on(request).map_err(python_exception))?;
        Ok(())
    }
}
