/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

fn main() {
    built::write_built_file().expect("Failed to acquire build-time information");
}
