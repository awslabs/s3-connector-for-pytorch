/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

// Information from build, made available by built crate.
mod built {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

pub const PACKAGE_NAME: &str = built::PKG_NAME;
pub const FULL_VERSION: &str = built::PKG_VERSION;
