#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import random

filename = input("Enter the file name: ")
filesize = int(input("Enter the file size in mb: ")) * 1024 * 1024
with open(filename, "wb") as f:
    for i in range(filesize):
        #convert int to 8bit value
        value = bytes([random.randint(0, 255)])
        f.write(value)